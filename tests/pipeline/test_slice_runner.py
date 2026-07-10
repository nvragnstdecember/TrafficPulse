"""P1-U12 slice runner: real-video end-to-end verification + demo hardening.

Exercises the full offline slice on a **real encoded clip** decoded through the
P1-U5 ingestion, with a scripted ``StubDetector`` standing in for perception (the
COCO RT-DETR does not fire the vehicle class on synthetic pixels -- the real
backend is proven separately in ``test_slice_e2e_rtdetr.py``). Everything else is
real: PTS-accurate ingestion, the real ``IouTracker``, the existing P1-U4 heading
derivation + wrong-way reasoner, and P1-U11 persistence + evidence manifests. Also
covers CLI wiring, determinism/replay, and fail-fast typed errors.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _slice_fixtures import (
    FRAME_COUNT,
    scripted_down_detector,
    write_wrong_way_clip,
)

from trafficpulse.contracts import ObjectClass, SceneConfig, ViolationType
from trafficpulse.detector import DetectorConfig
from trafficpulse.ingestion import SourceNotFoundError
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline import SceneConfigurationError
from trafficpulse.pipeline.runner import (
    SliceRunReport,
    _load_scene_config,
    _parse_label_map,
    main,
    run_wrong_way_slice,
)
from trafficpulse.tracking import IouTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
SCENE: SceneConfig = SceneConfig.model_validate(
    yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
)
NORTH = "dir-north"  # legal north = (0, -1); the clip's box moves down => wrong-way
DETECTOR_CONFIG = DetectorConfig(label_map={"car": ObjectClass.CAR})


def _run(clip: Path, output_dir: Path, run_id: str = "slice-1") -> SliceRunReport:
    return run_wrong_way_slice(
        clip=clip,
        scene=SCENE,
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=DETECTOR_CONFIG,
        output_dir=output_dir,
        run_id=run_id,
        direction_id=NORTH,
    )


# --- the core end-to-end assertion -------------------------------------------
def test_slice_produces_one_wrong_way_event_from_a_real_clip(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    report = _run(clip, tmp_path / "runs")

    assert report.frames_processed == FRAME_COUNT  # every decoded frame processed
    assert report.event_count == 1
    assert report.manifest_count == 1
    assert report.unique_tracks == 1
    assert report.detector_kind == "StubDetector"  # honest: not real RT-DETR
    assert report.tracker_kind == "IouTracker"  # the real backend ran
    assert report.checkpoint is None  # no real checkpoint on the stub path
    assert report.scene_config_hash  # provenance stamped
    assert report.width == 320 and report.height == 240


def test_event_is_persisted_with_a_linked_valid_manifest(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    _run(clip, tmp_path / "runs")

    stored = EventStore(tmp_path / "runs").load("slice-1")
    assert len(stored) == 1
    event, manifest = stored[0].event, stored[0].manifest
    assert event.violation_type is ViolationType.WRONG_WAY
    assert event.track_ids == ("iou-1",)  # id survived persistence from the real tracker
    assert manifest.event_id == event.event_id  # correct linkage
    assert manifest.trigger_frame is not None
    assert event.event_id in manifest.trigger_frame.locator  # relative trigger locator
    assert manifest.scene_config_hash == event.scene_config_hash  # hash survived


def test_frame_timestamps_come_from_media_pts_not_fps_assumption(tmp_path: Path) -> None:
    # Timestamps must come from real decoded PTS (10 fps => 0.1 s steps), anchored at
    # the pipeline's fixed media-time epoch -- never wall-clock. The run is a
    # contradiction from the first step, so it confirms exactly one min_persistence
    # (1.0 s of media time) after the run opens.
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    _run(clip, tmp_path / "runs")
    event = EventStore(tmp_path / "runs").load("slice-1")[0].event
    span = (event.trigger_at - event.start_at).total_seconds()
    assert span == pytest.approx(1.0, abs=0.05)  # == the scene's min_persistence, from PTS
    assert event.start_at.year == 1970  # media epoch anchor, not the 2026 wall-clock
    assert event.start_at.tzinfo is not None


# --- determinism / replay -----------------------------------------------------
def test_replay_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    first = _run(clip, tmp_path / "runs-a")
    second = _run(clip, tmp_path / "runs-b")

    a = EventStore(tmp_path / "runs-a").load("slice-1")[0].event
    b = EventStore(tmp_path / "runs-b").load("slice-1")[0].event
    assert a.event_id == b.event_id  # content-derived id is replay-stable
    # Reports agree on everything except the (intentionally different) output roots.
    assert (first.event_count, first.scene_config_hash) == (
        second.event_count,
        second.scene_config_hash,
    )

    # Byte-identical persisted files (write-once idempotent re-persist).
    file_a = (tmp_path / "runs-a" / "slice-1" / "events" / f"{a.event_id}.json").read_bytes()
    file_b = (tmp_path / "runs-b" / "slice-1" / "events" / f"{b.event_id}.json").read_bytes()
    assert file_a == file_b
    # Re-running into the same root is an idempotent no-op (no conflict raised).
    _run(clip, tmp_path / "runs-a")


_FRESH_SCRIPT = '''
import sys
from pathlib import Path

import yaml

from _slice_fixtures import scripted_down_detector
from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.detector import DetectorConfig
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline.runner import run_wrong_way_slice
from trafficpulse.tracking import IouTracker

fixtures_dir, scene_path, clip, out = sys.argv[1:5]
sys.path.insert(0, fixtures_dir)
scene = SceneConfig.model_validate(yaml.safe_load(Path(scene_path).read_text("utf-8")))
run_wrong_way_slice(
    clip=clip, scene=scene, detector=scripted_down_detector(), tracker=IouTracker(),
    detector_config=DetectorConfig(label_map={"car": ObjectClass.CAR}),
    output_dir=out, run_id="fresh", direction_id="dir-north",
)
print(EventStore(out).load("fresh")[0].event.event_id)
'''


def test_fresh_process_determinism(tmp_path: Path) -> None:
    # Prove the event id is process-independent (no PYTHONHASHSEED / wall-clock in
    # the identity path): a fresh interpreter yields the same id as in-process.
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    assert _run(clip, tmp_path / "runs").event_count == 1
    in_process_id = EventStore(tmp_path / "runs").load("slice-1")[0].event.event_id

    script_file = tmp_path / "fresh_runner.py"
    script_file.write_text(_FRESH_SCRIPT, encoding="utf-8")
    fixtures_dir = str(Path(__file__).parent)
    proc = subprocess.run(
        [
            sys.executable, str(script_file),
            fixtures_dir, str(SCENE_PATH), str(clip), str(tmp_path / "fresh-runs"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": fixtures_dir},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == in_process_id


# --- output containment -------------------------------------------------------
def test_output_stays_under_the_given_root(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    runs = tmp_path / "runs"
    _run(clip, runs)
    written = {p.relative_to(runs).parts[0] for p in runs.rglob("*") if p.is_file()}
    assert written == {"slice-1"}  # nothing escapes output_dir/run_id


# --- fail-fast typed errors ---------------------------------------------------
def test_missing_clip_fails_with_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        _run(tmp_path / "does-not-exist.mp4", tmp_path / "runs")


def test_ambiguous_scene_without_direction_id_fails(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    with pytest.raises(SceneConfigurationError):
        run_wrong_way_slice(
            clip=clip,
            scene=SCENE,  # declares two legal directions
            detector=scripted_down_detector(),
            tracker=IouTracker(),
            detector_config=DETECTOR_CONFIG,
            output_dir=tmp_path / "runs",
            run_id="slice-1",
            direction_id=None,  # ambiguous
        )


# --- CLI helpers + wiring -----------------------------------------------------
def test_load_scene_config_reads_the_committed_yaml_scene() -> None:
    scene = _load_scene_config(SCENE_PATH)
    assert scene.scene.camera_id == SCENE.scene.camera_id


def test_load_scene_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SceneConfigurationError):
        _load_scene_config(tmp_path / "nope.yaml")


def test_parse_label_map_default_and_explicit() -> None:
    assert _parse_label_map(None) == {"car": ObjectClass.CAR}
    assert _parse_label_map(["truck=truck", "bus=bus"]) == {
        "truck": ObjectClass.TRUCK,
        "bus": ObjectClass.BUS,
    }


def test_parse_label_map_rejects_bad_pairs() -> None:
    with pytest.raises(ValueError):
        _parse_label_map(["car"])  # no '='
    with pytest.raises(ValueError):
        _parse_label_map(["car=not_a_class"])


def test_cli_success_path_with_injected_detector(tmp_path, capsys, monkeypatch) -> None:
    # Drive the real CLI (arg parsing -> scene load -> run -> persist -> JSON out),
    # substituting the scripted detector for the real RT-DETR build so the default
    # suite needs no torch/checkpoint. Everything else is the production code path.
    import json

    import trafficpulse.pipeline.runner as runner

    monkeypatch.setattr(
        runner, "_build_rtdetr_detector", lambda **_: scripted_down_detector()
    )
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    code = main(
        [
            "--clip", str(clip),
            "--scene", str(SCENE_PATH),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "cli-1",
            "--checkpoint", "unused-because-injected",
            "--direction-id", NORTH,
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["event_count"] == 1
    assert report["manifest_count"] == 1
    assert report["run_id"] == "cli-1"


def test_cli_composition_root_stamps_truthful_model_provenance(
    tmp_path, capsys, monkeypatch
) -> None:
    # The composition root (P2-U1) must construct honest ModelRefs and wire them so
    # the persisted event + manifest carry truthful detector/tracker provenance:
    # detector name = the --checkpoint actually passed, tracker = the in-repo IoU
    # associator, weights_hash None everywhere (nothing hashed). The injected stub
    # replaces only real inference; the provenance wiring is the production path.
    import trafficpulse.pipeline.runner as runner

    monkeypatch.setattr(runner, "_build_rtdetr_detector", lambda **_: scripted_down_detector())
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    code = main(
        [
            "--clip", str(clip),
            "--scene", str(SCENE_PATH),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "prov-1",
            "--checkpoint", "cp-test-checkpoint",
            "--direction-id", NORTH,
        ]
    )
    assert code == 0
    stored = EventStore(tmp_path / "runs").load("prov-1")[0]
    expected = (
        runner._IOU_TRACKER_MODEL_REF,
        runner._rtdetr_model_ref("cp-test-checkpoint"),
    )
    # Sorted/de-duped: "cp-test-checkpoint" sorts before "iou-tracker".
    from trafficpulse.pipeline import normalize_model_refs

    assert stored.event.models == normalize_model_refs(expected)
    assert stored.manifest.models == stored.event.models
    assert all(ref.weights_hash is None for ref in stored.event.models)
    assert {ref.name for ref in stored.event.models} == {"iou-tracker", "cp-test-checkpoint"}


def test_cli_reports_typed_error_for_missing_scene(tmp_path, capsys) -> None:
    code = main(
        [
            "--clip", str(tmp_path / "clip.mp4"),
            "--scene", str(tmp_path / "missing-scene.yaml"),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "x",
            "--checkpoint", "unused",
        ]
    )
    assert code == 2
    assert "SceneConfigurationError" in capsys.readouterr().err

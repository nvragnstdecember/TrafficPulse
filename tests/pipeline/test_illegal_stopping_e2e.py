"""P2-U6 illegal-stopping slice: recorded-clip end-to-end verification + demo.

Exercises the full offline illegal-stopping slice on a **real encoded clip**
decoded through the P1-U5 ingestion, with a scripted ``StubDetector`` standing in
for perception (a COCO RT-DETR does not fire the vehicle class on synthetic pixels;
the real backend is proven separately, opt-in, at the bottom of this file).
Everything else is real: PTS-accurate ingestion, the real ``IouTracker``, the
P2-U2/U3 derivations + P2-U4 illegal-stopping reasoner, and P1-U11 persistence +
evidence manifests. Also covers the sibling runner's report, determinism/replay,
no-event / empty runs, the CLI wiring, truthful provenance, and a wrong-way
non-regression check.

Parity with the P1-U12 wrong-way honesty bar: a real decoded clip, real PTS, real
tracker, real persistence -- the event is driven by injected detections, never by a
false claim that RT-DETR fired.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from _stopping_fixtures import (
    FRAME_COUNT,
    STATIONARY_DURATION_S,
    illegal_stopping_test_scene,
    scripted_drive_through_detector,
    scripted_stopping_detector,
    stopping_detector_config,
    write_drive_through_clip,
    write_illegal_stopping_clip,
)

from trafficpulse.contracts import ModelRef, SceneConfig, ViolationType
from trafficpulse.contracts import scene_config_hash as _scene_hash
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline import normalize_model_refs
from trafficpulse.pipeline.illegal_stopping_runner import (
    IllegalStoppingSliceRunReport,
    main,
    run_illegal_stopping_slice,
)
from trafficpulse.tracking import IouTracker, TrackerConfig

SCENE: SceneConfig = illegal_stopping_test_scene()
SCH = _scene_hash(SCENE)


def _run(
    clip: Path,
    output_dir: Path,
    run_id: str = "stop-1",
    *,
    frames: int = FRAME_COUNT,
) -> IllegalStoppingSliceRunReport:
    return run_illegal_stopping_slice(
        clip=clip,
        scene=SCENE,
        detector=scripted_stopping_detector(frames),
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=output_dir,
        run_id=run_id,
    )


# --- the core end-to-end assertion -------------------------------------------
def test_slice_produces_one_illegal_stopping_event_from_a_real_clip(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    report = _run(clip, tmp_path / "runs")

    assert report.frames_processed == FRAME_COUNT  # every decoded frame processed
    assert report.event_count == 1
    assert report.manifest_count == 1
    assert report.unique_tracks == 1
    assert report.detector_kind == "StubDetector"  # honest: not real RT-DETR
    assert report.tracker_kind == "IouTracker"  # the real backend ran
    assert report.checkpoint is None  # no real checkpoint on the stub path
    assert report.no_stopping_zone_ids == ("zone-no-stop",)
    assert report.stationary_duration_seconds == STATIONARY_DURATION_S
    assert report.scene_config_hash == SCH
    assert report.width == 320 and report.height == 240


def test_event_is_persisted_with_a_linked_valid_manifest(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    _run(clip, tmp_path / "runs")

    stored = EventStore(tmp_path / "runs").load("stop-1")
    assert len(stored) == 1
    event, manifest = stored[0].event, stored[0].manifest
    assert event.violation_type is ViolationType.ILLEGAL_STOPPING
    assert event.track_ids == ("iou-1",)  # id survived persistence from the real tracker
    assert event.rule_id == "illegal_stopping"
    assert event.rule_version == "0.1.0-provisional"
    assert event.end_at is None
    assert event.scene_config_hash == SCH

    # Dwell reached the (test-scene) threshold; motion_threshold recorded not applied.
    measurements = {m.name: m for m in event.measurements}
    assert measurements["dwell_seconds"].value >= STATIONARY_DURATION_S
    thresholds = {t.name: t for t in event.thresholds}
    assert thresholds["stationary_duration"].value == STATIONARY_DURATION_S
    assert thresholds["motion_threshold"].value == 0.5

    # Evidence manifest linkage + rule trace (evidence generation).
    assert manifest.event_id == event.event_id
    assert manifest.trigger_frame is not None
    assert event.event_id in manifest.trigger_frame.locator  # relative trigger locator
    assert manifest.trigger_frame.sha256 is None  # nothing rendered/hashed (honest stub)
    assert manifest.scene_config_hash == event.scene_config_hash
    assert manifest.rule_trace[0].label == "rule:illegal_stopping"


def test_dwell_uses_media_pts_not_wall_clock(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    _run(clip, tmp_path / "runs")
    event = EventStore(tmp_path / "runs").load("stop-1")[0].event
    dwell = (event.trigger_at - event.start_at).total_seconds()
    assert dwell == pytest.approx(STATIONARY_DURATION_S, abs=0.15)  # from decoded PTS
    assert event.start_at.year == 1970  # media epoch anchor, not the 2026 wall-clock
    assert event.start_at.tzinfo is not None
    assert event.created_at == event.trigger_at  # deterministic data timestamp


# --- determinism / replay -----------------------------------------------------
def test_replay_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    first = _run(clip, tmp_path / "runs-a")
    second = _run(clip, tmp_path / "runs-b")

    a = EventStore(tmp_path / "runs-a").load("stop-1")[0].event
    b = EventStore(tmp_path / "runs-b").load("stop-1")[0].event
    assert a.event_id == b.event_id  # content-derived id is replay-stable
    assert (first.event_count, first.scene_config_hash) == (
        second.event_count,
        second.scene_config_hash,
    )

    # Byte-identical persisted files (write-once idempotent re-persist).
    file_a = (tmp_path / "runs-a" / "stop-1" / "events" / f"{a.event_id}.json").read_bytes()
    file_b = (tmp_path / "runs-b" / "stop-1" / "events" / f"{b.event_id}.json").read_bytes()
    assert file_a == file_b
    man_a = (tmp_path / "runs-a" / "stop-1" / "manifests" / f"{a.event_id}.json").read_bytes()
    man_b = (tmp_path / "runs-b" / "stop-1" / "manifests" / f"{b.event_id}.json").read_bytes()
    assert man_a == man_b
    # Re-running into the same root is an idempotent no-op (no conflict raised).
    _run(clip, tmp_path / "runs-a")


def test_multiple_replays_on_one_clip_agree(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    ids = {
        _run(clip, tmp_path / f"runs-{i}", run_id=f"r{i}").event_count for i in range(3)
    }
    assert ids == {1}  # every replay confirms exactly one event


# --- no-event / empty runs ----------------------------------------------------
def test_drive_through_clip_confirms_nothing(tmp_path: Path) -> None:
    # A vehicle driving through the zone (never stationary) is not an illegal stop.
    from trafficpulse.persistence.errors import RunNotFoundError

    clip = write_drive_through_clip(tmp_path / "drive.mp4")
    report = run_illegal_stopping_slice(
        clip=clip,
        scene=SCENE,
        detector=scripted_drive_through_detector(),
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=tmp_path / "runs",
        run_id="drive-1",
    )
    assert report.frames_processed >= 1
    assert report.event_count == 0
    assert report.manifest_count == 0
    # A zero-event run persists nothing (no run dir), distinct from a decode error.
    with pytest.raises(RunNotFoundError):
        EventStore(tmp_path / "runs").load("drive-1")


def test_empty_clip_raises_typed_ingestion_error(tmp_path: Path) -> None:
    # A clip with no decodable frames is an ingestion error, not a zero-event run.
    from trafficpulse.ingestion import VideoIngestionError

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")  # not a decodable video
    with pytest.raises(VideoIngestionError):
        _run(empty, tmp_path / "runs")


# --- truthful provenance (P2-U1 shape end to end) ----------------------------
_DET_REF = ModelRef(name="rtdetr-r50vd", version="provisional")
_TRK_REF = ModelRef(name="iou-tracker", version="0.1.0-provisional")


def test_models_propagated_through_the_slice(tmp_path: Path) -> None:
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    detector_config = stopping_detector_config().model_copy(update={"source_model": _DET_REF})
    run_illegal_stopping_slice(
        clip=clip,
        scene=SCENE,
        detector=scripted_stopping_detector(),
        tracker=IouTracker(tracker_config=TrackerConfig(tracker=_TRK_REF)),
        detector_config=detector_config,
        output_dir=tmp_path / "runs",
        run_id="prov-1",
    )
    stored = EventStore(tmp_path / "runs").load("prov-1")[0]
    assert stored.event.models == normalize_model_refs([_DET_REF, _TRK_REF])
    assert stored.manifest.models == stored.event.models
    assert all(ref.weights_hash is None for ref in stored.event.models)


def test_provenance_does_not_change_the_decision(tmp_path: Path) -> None:
    # The event id (the decision) is byte-identical with and without provenance.
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    with_refs = run_illegal_stopping_slice(
        clip=clip,
        scene=SCENE,
        detector=scripted_stopping_detector(),
        tracker=IouTracker(tracker_config=TrackerConfig(tracker=_TRK_REF)),
        detector_config=stopping_detector_config().model_copy(
            update={"source_model": _DET_REF}
        ),
        output_dir=tmp_path / "runs-p",
        run_id="p",
    )
    without = _run(clip, tmp_path / "runs-n", run_id="p")
    a = EventStore(tmp_path / "runs-p").load("p")[0].event
    b = EventStore(tmp_path / "runs-n").load("p")[0].event
    assert a.event_id == b.event_id
    assert with_refs.event_count == without.event_count == 1


# --- CLI wiring ---------------------------------------------------------------
def _write_scene_json(tmp_path: Path) -> Path:
    path = tmp_path / "scene.json"
    path.write_text(SCENE.model_dump_json(), encoding="utf-8")
    return path


def test_cli_success_path_with_injected_detector(tmp_path, capsys, monkeypatch) -> None:
    # Drive the real CLI (arg parsing -> scene load -> run -> persist -> JSON out),
    # substituting the scripted detector for the real RT-DETR build so the default
    # suite needs no torch/checkpoint. Everything else is the production code path.
    import json

    import trafficpulse.pipeline.illegal_stopping_runner as runner

    monkeypatch.setattr(runner, "_build_rtdetr_detector", lambda **_: scripted_stopping_detector())
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    code = main(
        [
            "--clip", str(clip),
            "--scene", str(_write_scene_json(tmp_path)),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "cli-1",
            "--checkpoint", "cp-test-checkpoint",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["event_count"] == 1
    assert report["manifest_count"] == 1
    assert report["run_id"] == "cli-1"
    assert report["no_stopping_zone_ids"] == ["zone-no-stop"]
    assert report["detector_kind"] == "StubDetector"


def test_cli_stamps_truthful_model_provenance(tmp_path, capsys, monkeypatch) -> None:
    # The composition root constructs honest ModelRefs and wires them so the
    # persisted event + manifest carry truthful detector/tracker provenance:
    # detector name = the --checkpoint passed, tracker = the in-repo IoU associator,
    # weights_hash None everywhere. The injected stub replaces only real inference.
    import trafficpulse.pipeline.illegal_stopping_runner as runner

    monkeypatch.setattr(runner, "_build_rtdetr_detector", lambda **_: scripted_stopping_detector())
    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    code = main(
        [
            "--clip", str(clip),
            "--scene", str(_write_scene_json(tmp_path)),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "prov-cli",
            "--checkpoint", "cp-test-checkpoint",
        ]
    )
    assert code == 0
    stored = EventStore(tmp_path / "runs").load("prov-cli")[0]
    expected = normalize_model_refs(
        (runner._IOU_TRACKER_MODEL_REF, runner._rtdetr_model_ref("cp-test-checkpoint"))
    )
    assert stored.event.models == expected
    assert stored.manifest.models == stored.event.models
    assert all(ref.weights_hash is None for ref in stored.event.models)
    assert {ref.name for ref in stored.event.models} == {"iou-tracker", "cp-test-checkpoint"}


def test_cli_reports_typed_error_for_missing_scene(tmp_path, capsys) -> None:
    code = main(
        [
            "--clip", str(tmp_path / "clip.mp4"),
            "--scene", str(tmp_path / "missing-scene.json"),
            "--output-dir", str(tmp_path / "runs"),
            "--run-id", "x",
            "--checkpoint", "unused",
        ]
    )
    assert code == 2
    assert "SceneConfigurationError" in capsys.readouterr().err


# --- wrong-way non-regression -------------------------------------------------
def test_wrong_way_slice_still_confirms(tmp_path: Path) -> None:
    # Adding the illegal-stopping sibling runner must not disturb the shipped
    # wrong-way slice (it reuses runner.py's stateless CLI helpers unchanged).
    import yaml
    from _slice_fixtures import scripted_down_detector, write_wrong_way_clip

    from trafficpulse.pipeline.runner import run_wrong_way_slice

    scene_path = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"
    scene = SceneConfig.model_validate(yaml.safe_load(scene_path.read_text(encoding="utf-8")))
    clip = write_wrong_way_clip(tmp_path / "ww.mp4")
    report = run_wrong_way_slice(
        clip=clip,
        scene=scene,
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=tmp_path / "runs",
        run_id="ww-1",
        direction_id="dir-north",
    )
    assert report.event_count == 1
    ev = EventStore(tmp_path / "runs").load("ww-1")[0].event
    assert ev.violation_type is ViolationType.WRONG_WAY


# --- opt-in REAL RT-DETR end-to-end (skipped by default) ----------------------
_MODEL = os.environ.get("TRAFFICPULSE_E2E_MODEL")
_DEVICE = os.environ.get("TRAFFICPULSE_E2E_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@pytest.mark.skipif(
    not (_MODEL and _HAVE_DEPS),
    reason=(
        "opt-in real RT-DETR end-to-end illegal-stopping slice: install "
        "trafficpulse[rtdetr] and set TRAFFICPULSE_E2E_MODEL to a locally-available "
        "checkpoint"
    ),
)
def test_real_rtdetr_illegal_stopping_slice_runs_end_to_end(tmp_path: Path) -> None:
    from trafficpulse.detector import RTDetrConfig, RTDetrDetector

    clip = write_illegal_stopping_clip(tmp_path / "clip.mp4")
    detector = RTDetrDetector(
        RTDetrConfig(checkpoint=str(_MODEL), device=_DEVICE, local_files_only=True, threshold=0.5)
    )
    report = run_illegal_stopping_slice(
        clip=clip,
        scene=SCENE,
        detector=detector,
        tracker=IouTracker(),
        detector_config=stopping_detector_config(),
        output_dir=tmp_path / "runs",
        run_id="e2e-rtdetr",
        checkpoint=str(_MODEL),
        device=_DEVICE,
    )
    # Real inference integrated through the whole slice without leaking a backend type.
    assert report.detector_kind == "RTDetrDetector"
    assert report.frames_processed >= 1
    assert isinstance(report.event_count, int) and report.event_count >= 0
    assert report.manifest_count == report.event_count

"""P1-U10 -> P1-U11 integration: run the wrong-way slice, then persist + reload.

Drives the real ``WrongWayPipeline`` (P1-U10) with a scripted ``StubDetector`` +
``StubTracker`` over synthetic frames -- the same deterministic in-memory slice the
pipeline suite uses -- takes its ``ConfirmedEvent`` output, persists it through the
P1-U11 ``EventStore``, reloads it, and verifies the linked ``EvidenceManifest``.
Then it repeats the slice to confirm second-run / replay identity. This exercises
the actual U10 event, not a hand-built one, so the two units are wired for real.

Self-contained (no cross-directory helper import): it builds the pipeline inputs
from the public APIs and the committed example scene, anchoring timestamps at the
pipeline's fixed media-time epoch.
"""

from pathlib import Path

import numpy as np
import yaml

from trafficpulse.contracts import ObjectClass, SceneConfig, ViolationType
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector
from trafficpulse.ingestion.video import FrameRecord
from trafficpulse.persistence import EventStore, build_evidence_manifest
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.tracking import ScriptedAssignment, StubTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE: SceneConfig = SceneConfig.model_validate(
    yaml.safe_load((REPO_ROOT / "configs" / "scenes" / "example-scene.yaml").read_text("utf-8"))
)
CAMERA = SCENE.scene.camera_id
NORTH_DIRECTION_ID = "dir-north"  # legal north = (0,-1); moving down is wrong-way
DETECTOR_CONFIG = DetectorConfig(label_map={"car": ObjectClass.CAR})
FRAME_INTERVAL_S = 1.0 / 30.0
FRAME_COUNT = 45  # > the scene's 1.0 s wrong_way min_persistence
_PIXEL = np.zeros((1, 1, 3), dtype=np.uint8)


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(
            source_id="vsrc-test",
            camera_id=CAMERA,
            frame_id=f"vfrm-{i}",
            frame_index=i,
            timestamp_seconds=i * FRAME_INTERVAL_S,
            width=1,
            height=1,
            image=_PIXEL,
        )
        for i in range(FRAME_COUNT)
    ]


def _moving_down_detector() -> StubDetector:
    # A car moving DOWN (+y) each frame -> contradicts legal north -> wrong-way.
    per_frame = {
        i: (RawDetection(label="car", score=0.9, box=(50.0, 50.0 + 5.0 * i, 70.0, 70.0 + 5.0 * i)),)
        for i in range(FRAME_COUNT)
    }
    return StubDetector(per_frame=per_frame)


def _pipeline() -> WrongWayPipeline:
    script = {i: (ScriptedAssignment(track_id="t1"),) for i in range(FRAME_COUNT)}
    return WrongWayPipeline(
        detector=_moving_down_detector(),
        tracker=StubTracker(script),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )


def test_u10_event_is_a_real_wrong_way_event() -> None:
    events = _pipeline().process(_frames())
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.WRONG_WAY
    assert events[0].track_ids == ("t1",)


def test_persist_and_reload_u10_event_with_linked_manifest(tmp_path: Path) -> None:
    events = _pipeline().process(_frames())
    store = EventStore(tmp_path)
    stored = store.persist("slice-run-1", events)
    assert len(stored) == 1

    reloaded = store.load("slice-run-1")
    assert len(reloaded) == 1
    event = reloaded[0].event
    manifest = reloaded[0].manifest

    # The reloaded event equals the pipeline's event (semantic round-trip).
    assert event == events[0]
    # The manifest is the minimal stub for that event, correctly linked.
    assert manifest == build_evidence_manifest(events[0])
    assert manifest.event_id == event.event_id
    assert manifest.evidence_package_id == f"evp-{event.event_id}"
    assert manifest.trigger_frame is not None
    assert event.event_id in manifest.trigger_frame.locator
    # Provenance from the real run survived.
    assert event.scene_config_hash is not None
    assert manifest.scene_config_hash == event.scene_config_hash


def test_second_run_of_the_slice_is_identity_stable(tmp_path: Path) -> None:
    # Two fresh runs of the deterministic slice mint the same content-derived event
    # id, so persisting into the same run is idempotent (no duplicate, no error).
    first = _pipeline().process(_frames())
    second = _pipeline().process(_frames())
    assert tuple(e.event_id for e in first) == tuple(e.event_id for e in second)

    store = EventStore(tmp_path)
    store.persist("slice-run-1", first)
    store.persist("slice-run-1", second)  # idempotent replay
    reloaded = store.load("slice-run-1")
    assert len(reloaded) == 1
    assert reloaded[0].event == first[0]


def test_persist_writes_only_under_the_given_root(tmp_path: Path) -> None:
    events = _pipeline().process(_frames())
    EventStore(tmp_path).persist("slice-run-1", events)
    # Everything landed under tmp_path/slice-run-1 (a gitignored-style runtime root).
    written = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"slice-run-1"}

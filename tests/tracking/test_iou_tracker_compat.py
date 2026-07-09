"""Real end-to-end smoke: IouTracker -> heading derivation -> wrong-way (P1-U9).

Unlike the P1-U7 RT-DETR smoke test (which needs weights and is skipped by
default), the greedy-IoU backend requires **no** model artifact, dependency,
network, or GPU, so its real smoke runs unconditionally in the default suite. It
drives synthetic, in-memory ``Detection`` sequences through the *real*
``IouTracker`` and proves the emitted ``TrackState``s flow into the existing P1-U4
heading derivation and P1-U4 wrong-way reasoner with **no** conversion glue,
yielding a real wrong-way ``ConfirmedEvent`` -- and that the whole path is
deterministic on replay.

The scene is the repository's example scene: legal direction *north* = ``(0, -1)``
(image space, +y down), so an object moving **down** contradicts it. Boxes advance
in small steps so consecutive detections overlap enough for IoU association to
hold one identity across the whole track.
"""

from datetime import timedelta
from pathlib import Path

import yaml
from _builders import BASE, FRAME_INTERVAL_S

from trafficpulse.contracts import (
    BoundingBox,
    ConfirmedEvent,
    Detection,
    ObjectClass,
    SceneConfig,
    ViolationType,
    scene_config_hash,
)
from trafficpulse.observations.heading import (
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.wrong_way import WrongWayReasoner, wrong_way_parameters
from trafficpulse.tracking import IouTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
SCENE = SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))
PARAMS = wrong_way_parameters(SCENE)
SCH = scene_config_hash(SCENE)
NORTH = next(d for d in SCENE.legal_directions if d.direction_id == "dir-north")
LANE = NORTH.zone_ids[0]

# 45 frames at 30 fps spans ~1.47 s, exceeding the scene's 1.0 s min_persistence
# (mirrors the P1-U4 wrong-way confirmation fixture).
_FRAME_COUNT = 45
_STEP_PX = 5.0  # small enough that consecutive boxes overlap (IoU ~0.6 > 0.3)


def _moving_down(frame_index: int, *, camera_id: str = "cam1", x: float = 50.0) -> Detection:
    """A car whose bbox centre advances +``_STEP_PX`` in y each frame (wrong-way)."""

    top = 50.0 + frame_index * _STEP_PX
    return Detection(
        detection_id=f"det-{camera_id}-{frame_index}",
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp=BASE + timedelta(seconds=frame_index * FRAME_INTERVAL_S),
        object_class=ObjectClass.CAR,
        confidence=0.9,
        bbox=BoundingBox(x1=x, y1=top, x2=x + 20.0, y2=top + 20.0),
    )


def _track_via_iou(frame_count: int = _FRAME_COUNT) -> list:
    """Run one moving object through the real IouTracker; collect its TrackStates."""

    tracker = IouTracker()
    states = []
    for i in range(frame_count):
        (state,) = tracker.update([_moving_down(i)])
        states.append(state)
    return states


def _reasoner() -> WrongWayReasoner:
    return WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)


# --- identity ----------------------------------------------------------------
def test_iou_tracker_holds_one_identity_across_the_whole_track() -> None:
    states = _track_via_iou()
    assert len(states) == _FRAME_COUNT
    assert {s.track_id for s in states} == {"iou-1"}  # single continuous identity


# --- heading-derivation compatibility (no glue) ------------------------------
def test_iou_states_feed_heading_derivation_without_glue() -> None:
    track = _track_via_iou()
    observations = derive_heading_observations(
        track, legal_direction=NORTH.vector, lane_id=LANE, deviation_max_degrees=120.0
    )
    assert observations
    # Straight-down motion against a legal "north/up" is a 180-degree contradiction.
    assert all(abs(o.deviation_degrees - 180.0) < 1e-9 for o in observations)
    assert all(o.is_contradiction for o in observations)
    assert all(o.track_id == "iou-1" for o in observations)


# --- wrong-way confirmation (real end-to-end) --------------------------------
def test_iou_track_produces_a_wrong_way_confirmed_event() -> None:
    track = _track_via_iou()
    derivation = derive_heading_observations_with_taint(
        track, legal_direction=NORTH.vector, lane_id=LANE, deviation_max_degrees=120.0
    )
    reasoner = _reasoner()
    events = reasoner.run_derivation(derivation)
    assert len(events) >= 1
    event = events[0]
    assert isinstance(event, ConfirmedEvent)
    assert event.violation_type is ViolationType.WRONG_WAY
    assert event.camera_id == "cam1"
    assert "iou-1" in event.track_ids
    assert derivation.taint_restart_ids == frozenset()  # backend emits no taint


def test_end_to_end_is_deterministic_on_replay() -> None:
    def once() -> tuple[str, ...]:
        derivation = derive_heading_observations_with_taint(
            _track_via_iou(), legal_direction=NORTH.vector, lane_id=LANE,
            deviation_max_degrees=120.0,
        )
        return tuple(e.event_id for e in _reasoner().run_derivation(derivation))

    assert once() == once()  # identical event ids across independent runs


# --- multiple simultaneous objects through the real backend ------------------
def test_two_objects_track_independently_end_to_end() -> None:
    tracker = IouTracker()
    per_track: dict[str, list] = {}
    for i in range(_FRAME_COUNT):
        # Two well-separated cars, both moving down (both wrong-way).
        states = tracker.update([_moving_down(i, x=50.0), _moving_down(i, x=300.0)])
        assert len(states) == 2
        for s in states:
            per_track.setdefault(s.track_id, []).append(s)
    assert set(per_track) == {"iou-1", "iou-2"}  # two stable, independent identities

    total_events = 0
    for track in per_track.values():
        derivation = derive_heading_observations_with_taint(
            track, legal_direction=NORTH.vector, lane_id=LANE, deviation_max_degrees=120.0
        )
        total_events += len(_reasoner().run_derivation(derivation))
    assert total_events == 2  # each object independently confirms wrong-way

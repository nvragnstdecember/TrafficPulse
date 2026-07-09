"""Cross-package seam compatibility: StubTracker -> existing heading derivation (P1-U8).

Proves that ``TrackState`` values produced through the tracker seam flow into the
existing P1-U4 heading derivation with **no** conversion glue or contract
mutation, and that a scripted taint reaches the derivation's taint-restart
machinery. This is seam verification only -- it does not build the P1-U10
pipeline.
"""

from datetime import timedelta

import pytest
from _builders import BASE, FRAME_INTERVAL_S

from trafficpulse.contracts import BoundingBox, Detection, ObjectClass
from trafficpulse.contracts.scene import DirectionVector
from trafficpulse.observations.heading import (
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
from trafficpulse.tracking import ScriptedAssignment, StubTracker

# Legal direction "up" (decreasing y); a track moving *down* (+y) is wrong-way.
UP = DirectionVector(dx=0.0, dy=-1.0)


def _moving_down_detection(frame_index: int) -> Detection:
    """A detection whose bbox center advances +10px in y each frame."""

    top = 100.0 + frame_index * 10.0
    return Detection(
        detection_id=f"det-{frame_index}",
        camera_id="cam1",
        frame_index=frame_index,
        timestamp=BASE + timedelta(seconds=frame_index * FRAME_INTERVAL_S),
        object_class=ObjectClass.CAR,
        confidence=0.9,
        bbox=BoundingBox(x1=50.0, y1=top, x2=70.0, y2=top + 20.0),
    )


def _track_via_stub(frame_count: int, *, tainted_frame: int | None = None) -> list:
    """Run a single moving track through the stub and collect its TrackStates."""

    script = {
        i: [ScriptedAssignment("T1", tainted=(i == tainted_frame))] for i in range(frame_count)
    }
    tracker = StubTracker(script)
    states = []
    for i in range(frame_count):
        (state,) = tracker.update([_moving_down_detection(i)])
        states.append(state)
    return states


def test_stub_track_feeds_heading_derivation_without_glue() -> None:
    track = _track_via_stub(5)
    observations = derive_heading_observations(
        track, legal_direction=UP, lane_id="lane-1", deviation_max_degrees=120.0
    )
    # A track moving straight down against a legal "up" direction is a 180-degree
    # contradiction on every usable step -- no conversion between the seams.
    assert observations
    assert all(o.deviation_degrees == pytest.approx(180.0) for o in observations)
    assert all(o.is_contradiction for o in observations)
    assert all(o.track_id == "T1" for o in observations)


def test_scripted_taint_reaches_derivation_taint_restart() -> None:
    # Taint frame 2; derivation must skip the tainted steps and flag the first
    # clean observation that resumes after them as a taint restart.
    track = _track_via_stub(5, tainted_frame=2)
    derivation = derive_heading_observations_with_taint(
        track, legal_direction=UP, lane_id="lane-1", deviation_max_degrees=120.0
    )
    assert derivation.taint_restart_ids  # taint propagated through the seam intact
    restart_ids = derivation.taint_restart_ids
    assert any(o.observation_id in restart_ids for o in derivation.observations)


def test_clean_track_has_no_taint_restart() -> None:
    derivation = derive_heading_observations_with_taint(
        _track_via_stub(5), legal_direction=UP, lane_id="lane-1", deviation_max_degrees=120.0
    )
    assert derivation.taint_restart_ids == frozenset()

"""TrackAdapter conversion tests (P1-U8).

The single centralized ``TrackState`` construction point: field carry-through
from the source detection, identity/lifecycle/taint stamping, velocity wrapping,
provenance, order preservation, and malformed-assignment rejection.
"""

import math

import pytest
from _builders import make_detection

from trafficpulse.contracts import ModelRef, TrackState, Velocity
from trafficpulse.contracts.enums import TrackStatus
from trafficpulse.tracking import TrackAdapter, TrackAssignment, TrackerConfig
from trafficpulse.tracking.errors import MalformedAssignmentError


def _assignment(**overrides: object) -> TrackAssignment:
    kwargs: dict[str, object] = {
        "track_id": "T1",
        "detection": make_detection(0),
        "status": TrackStatus.ACTIVE,
    }
    kwargs.update(overrides)
    return TrackAssignment(**kwargs)  # type: ignore[arg-type]


def test_carry_through_from_detection() -> None:
    detection = make_detection(7)
    adapter = TrackAdapter()
    (state,) = adapter.adapt([_assignment(detection=detection)])
    assert state.camera_id == detection.camera_id
    assert state.frame_index == detection.frame_index
    assert state.timestamp == detection.timestamp
    assert state.object_class == detection.object_class
    assert state.bbox == detection.bbox
    # Detection.confidence (required) carries to the optional TrackState.confidence.
    assert state.confidence == detection.confidence


def test_identity_status_taint_stamped() -> None:
    adapter = TrackAdapter()
    (state,) = adapter.adapt(
        [_assignment(track_id="T9", status=TrackStatus.TENTATIVE, tainted=True)]
    )
    assert state.track_id == "T9"
    assert state.status is TrackStatus.TENTATIVE
    assert state.tainted is True


def test_velocity_tuple_wrapped_in_contract() -> None:
    adapter = TrackAdapter()
    (state,) = adapter.adapt([_assignment(velocity=(1.5, -2.5))])
    assert state.velocity == Velocity(vx=1.5, vy=-2.5)


def test_velocity_none_leaves_field_unset() -> None:
    adapter = TrackAdapter()
    (state,) = adapter.adapt([_assignment(velocity=None)])
    assert state.velocity is None


def test_output_order_preserved() -> None:
    adapter = TrackAdapter()
    states = adapter.adapt(
        [
            _assignment(track_id="A", detection=make_detection(0, 0)),
            _assignment(track_id="B", detection=make_detection(0, 1)),
            _assignment(track_id="C", detection=make_detection(0, 2)),
        ]
    )
    assert [s.track_id for s in states] == ["A", "B", "C"]


def test_empty_assignments_yield_empty() -> None:
    assert TrackAdapter().adapt([]) == ()


def test_output_is_exactly_track_state() -> None:
    adapter = TrackAdapter()
    states = adapter.adapt([_assignment()])
    assert all(type(s) is TrackState for s in states)  # no subclass leakage


# --- provenance --------------------------------------------------------------
def test_tracker_provenance_stamped_from_config() -> None:
    ref = ModelRef(name="stub-tracker", version="0.0.0")
    adapter = TrackAdapter(TrackerConfig(tracker=ref))
    (state,) = adapter.adapt([_assignment()])
    assert state.tracker == ref


def test_tracker_provenance_defaults_to_none() -> None:
    (state,) = TrackAdapter().adapt([_assignment()])
    assert state.tracker is None


# --- malformed rejection -----------------------------------------------------
def test_empty_track_id_rejected() -> None:
    with pytest.raises(MalformedAssignmentError):
        TrackAdapter().adapt([_assignment(track_id="")])


def test_non_finite_velocity_rejected() -> None:
    with pytest.raises(MalformedAssignmentError):
        TrackAdapter().adapt([_assignment(velocity=(math.nan, 0.0))])


def test_wrong_length_velocity_rejected() -> None:
    with pytest.raises(MalformedAssignmentError):
        TrackAdapter().adapt([_assignment(velocity=(1.0, 2.0, 3.0))])  # type: ignore[arg-type]

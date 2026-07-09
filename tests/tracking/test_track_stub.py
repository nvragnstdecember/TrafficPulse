"""StubTracker behaviour tests (P1-U8).

Deterministic scripted replay across a frame stream: empty frames, single and
multiple simultaneous tracks, stable identity, appearance/disappearance, output
ordering, replay determinism, reset, and clear failure on stub misuse and out-of-
order / mixed-frame input.
"""

import pytest
from _builders import BASE, make_detection

from trafficpulse.contracts.enums import TrackStatus
from trafficpulse.tracking import ScriptedAssignment, StubTracker
from trafficpulse.tracking.errors import (
    InconsistentDetectionBatchError,
    MalformedAssignmentError,
    NonMonotonicFrameError,
    ScriptedAssignmentError,
)


# --- empty / single ----------------------------------------------------------
def test_empty_frame_yields_empty() -> None:
    assert StubTracker().update([]) == ()


def test_single_detection_yields_one_track_state() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")]})
    (state,) = tracker.update([make_detection(0)])
    assert state.track_id == "T1"
    assert state.status is TrackStatus.ACTIVE
    assert state.frame_index == 0


# --- identity across frames --------------------------------------------------
def test_stable_identity_across_frames() -> None:
    tracker = StubTracker(
        {i: [ScriptedAssignment("T1")] for i in range(4)}
    )
    ids = [tracker.update([make_detection(i)])[0].track_id for i in range(4)]
    assert ids == ["T1", "T1", "T1", "T1"]


def test_multiple_simultaneous_tracks() -> None:
    tracker = StubTracker(
        {0: [ScriptedAssignment("A"), ScriptedAssignment("B")]}
    )
    states = tracker.update([make_detection(0, 0), make_detection(0, 1)])
    assert [s.track_id for s in states] == ["A", "B"]


def test_appearance_and_disappearance() -> None:
    # frame 0: T1 only; frame 1: T1 + T2 appears; frame 2: T2 only (T1 disappears).
    tracker = StubTracker(
        {
            0: [ScriptedAssignment("T1")],
            1: [ScriptedAssignment("T1"), ScriptedAssignment("T2")],
            2: [ScriptedAssignment("T2")],
        }
    )
    assert [s.track_id for s in tracker.update([make_detection(0)])] == ["T1"]
    assert [s.track_id for s in tracker.update([make_detection(1, 0), make_detection(1, 1)])] == [
        "T1",
        "T2",
    ]
    assert [s.track_id for s in tracker.update([make_detection(2)])] == ["T2"]


def test_empty_frame_between_populated_frames_preserves_identity() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")], 2: [ScriptedAssignment("T1")]})
    assert tracker.update([make_detection(0)])[0].track_id == "T1"
    assert tracker.update([]) == ()  # inert empty frame 1
    assert tracker.update([make_detection(2)])[0].track_id == "T1"


# --- propagation -------------------------------------------------------------
def test_frame_and_timestamp_and_box_and_class_propagate() -> None:
    tracker = StubTracker({4: [ScriptedAssignment("T1")]})
    detection = make_detection(4)
    (state,) = tracker.update([detection])
    assert state.frame_index == detection.frame_index
    assert state.timestamp == detection.timestamp
    assert state.bbox == detection.bbox
    assert state.object_class == detection.object_class
    assert state.confidence == detection.confidence


# --- determinism -------------------------------------------------------------
def _run_stream(tracker: StubTracker) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for i in range(3):
        states = tracker.update([make_detection(i)])
        out.append(tuple(s.track_id for s in states))
    return out


def test_output_ordering_is_deterministic() -> None:
    script = {0: [ScriptedAssignment("A"), ScriptedAssignment("B"), ScriptedAssignment("C")]}
    dets = [make_detection(0, 0), make_detection(0, 1), make_detection(0, 2)]
    first = StubTracker(script).update(dets)
    second = StubTracker(script).update(dets)
    assert [s.track_id for s in first] == [s.track_id for s in second] == ["A", "B", "C"]


def test_fresh_instances_replay_equal() -> None:
    script = {i: [ScriptedAssignment("T1")] for i in range(3)}
    assert _run_stream(StubTracker(script)) == _run_stream(StubTracker(script))


def test_reset_replays_same_stream() -> None:
    script = {i: [ScriptedAssignment("T1")] for i in range(3)}
    tracker = StubTracker(script)
    first = _run_stream(tracker)
    tracker.reset()
    assert _run_stream(tracker) == first


def test_full_track_states_equal_across_repeated_runs() -> None:
    script = {i: [ScriptedAssignment("T1")] for i in range(3)}
    a = [StubTracker(script).update([make_detection(i)]) for i in range(3)]
    b = [StubTracker(script).update([make_detection(i)]) for i in range(3)]
    assert a == b  # frozen TrackState equality across repeated scripted runs


# --- taint -------------------------------------------------------------------
def test_scripted_taint_emitted() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1", tainted=True)]})
    (state,) = tracker.update([make_detection(0)])
    assert state.tainted is True


# --- failure modes -----------------------------------------------------------
def test_unscripted_populated_frame_rejected() -> None:
    with pytest.raises(ScriptedAssignmentError):
        StubTracker().update([make_detection(0)])


def test_script_count_mismatch_rejected() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")]})  # one assignment...
    with pytest.raises(ScriptedAssignmentError):
        tracker.update([make_detection(0, 0), make_detection(0, 1)])  # ...two detections


def test_empty_track_id_in_script_rejected() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("")]})
    with pytest.raises(MalformedAssignmentError):
        tracker.update([make_detection(0)])


def test_mixed_frame_batch_rejected() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1"), ScriptedAssignment("T2")]})
    with pytest.raises(InconsistentDetectionBatchError):
        tracker.update([make_detection(0), make_detection(1)])


def test_out_of_order_frames_rejected() -> None:
    tracker = StubTracker({5: [ScriptedAssignment("T1")], 3: [ScriptedAssignment("T1")]})
    tracker.update([make_detection(5)])
    with pytest.raises(NonMonotonicFrameError):
        tracker.update([make_detection(3)])


def test_regressing_timestamp_rejected() -> None:
    tracker = StubTracker({5: [ScriptedAssignment("T1")], 6: [ScriptedAssignment("T1")]})
    tracker.update([make_detection(5)])
    with pytest.raises(NonMonotonicFrameError):
        # frame_index advances (6 > 5) but timestamp regresses.
        tracker.update([make_detection(6, timestamp=BASE)])


def test_reset_clears_progress_for_out_of_order_replay() -> None:
    tracker = StubTracker({5: [ScriptedAssignment("T1")], 3: [ScriptedAssignment("T1")]})
    tracker.update([make_detection(5)])
    tracker.reset()
    # After reset, frame 3 is accepted (would raise NonMonotonicFrameError otherwise).
    assert tracker.update([make_detection(3)])[0].track_id == "T1"

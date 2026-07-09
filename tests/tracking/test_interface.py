"""Tracker interface-conformance tests (P1-U8).

The ``Tracker`` seam: the abstraction cannot be instantiated, ``StubTracker``
satisfies it, and ``update`` returns frozen ``TrackState`` values.
"""

import pytest
from _builders import make_detection

from trafficpulse.contracts import TrackState
from trafficpulse.tracking import ScriptedAssignment, StubTracker, Tracker


def test_tracker_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tracker()  # type: ignore[abstract]


def test_stub_is_a_tracker() -> None:
    assert isinstance(StubTracker(), Tracker)


def test_update_returns_track_states() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")]})
    states = tracker.update([make_detection(0)])
    assert isinstance(states, tuple)
    assert all(isinstance(s, TrackState) for s in states)


def test_stub_implements_reset() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")]})
    tracker.update([make_detection(0)])
    tracker.reset()  # must not raise; returns to pre-stream state
    # After reset the same first frame can be replayed.
    assert tracker.update([make_detection(0)])[0].track_id == "T1"

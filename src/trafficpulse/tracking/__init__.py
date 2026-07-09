"""Tracker-integration foundation for TrafficPulse (Phase 1, unit P1-U8).

The framework-neutral tracker seam that turns per-frame ``Detection`` streams into
ordered, identity-bearing frozen U2 ``TrackState`` sequences -- the tracking
analogue of the P1-U6 detector foundation. It provides: a ``Tracker`` interface
(the dependency-injection boundary), a framework-neutral ``TrackAssignment`` raw
type, a deterministic ``TrackAdapter`` that stamps the frozen ``TrackState``
contract (the single centralized construction point), a pydantic ``TrackerConfig``
(provenance stamp only), a scripted ``StubTracker`` for tests and P1-U10
orchestration, and a ``TrackerError`` taxonomy.

This foundation carries **no** ML / tracker-framework dependency and implements no
association logic (IoU, Kalman, Hungarian, re-ID). A real detection-based tracker
(P1-U9) plugs in without an API change: it emits ``TrackAssignment`` values from
its own native objects inside ``update``, and those native objects never escape.
The first such backend is :class:`IouTracker`, a dependency-free greedy-IoU
associator (the permissive fallback the plan authorises when the audited external
ByteTrack path is blocked). Downstream layers consume
``trafficpulse.contracts.TrackState`` only.
"""

from .adapter import TrackAdapter
from .config import TrackerConfig
from .errors import (
    InconsistentDetectionBatchError,
    MalformedAssignmentError,
    NonMonotonicFrameError,
    ScriptedAssignmentError,
    TrackerError,
)
from .interface import Tracker
from .iou_tracker import IouTracker, IouTrackerConfig
from .raw import TrackAssignment
from .sequencing import FrameKey, FrameProgress, single_frame_key
from .stub import ScriptedAssignment, StubTracker

__all__ = [
    # interface + implementations
    "Tracker",
    "StubTracker",
    "ScriptedAssignment",
    # real backend (P1-U9)
    "IouTracker",
    "IouTrackerConfig",
    # conversion
    "TrackAdapter",
    # configuration
    "TrackerConfig",
    # boundary types
    "TrackAssignment",
    # sequencing helpers (shared with a future real backend)
    "FrameKey",
    "FrameProgress",
    "single_frame_key",
    # errors
    "TrackerError",
    "InconsistentDetectionBatchError",
    "NonMonotonicFrameError",
    "MalformedAssignmentError",
    "ScriptedAssignmentError",
]

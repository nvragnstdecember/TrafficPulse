"""A deterministic stub tracker for tests and wiring (P1-U8).

``StubTracker`` implements the ``Tracker`` interface without any association
logic, motion model, or framework dependency: it *replays* a caller-supplied
script of identity assignments, keyed by ``frame.frame_index`` and paired to each
frame's detections **positionally**. It exists so the tracker seam, the adapter,
and any future tracker-consuming code (P1-U10 orchestration) can be exercised
deterministically before a real tracker exists.

It is deliberately **not** a fake tracker: it computes nothing about identity. The
caller *declares* which detection is which track (and its status / taint / optional
velocity) via the per-frame script, and the stub stamps exactly that. This is what
lets a test construct a track with a known ID switch (``tainted=True``), a known
appearance/disappearance pattern, or multiple simultaneous tracks, with no hidden
IoU / Kalman / Hungarian / re-ID behaviour to reason about.

Determinism
-----------
``update`` is a pure function of the script and the ordered stream of ``update``
calls -- no wall-clock, no randomness, no global state. Two fresh ``StubTracker``s
built from the same script, fed the same frames, produce equal ``TrackState``
tuples. ``reset`` clears only the frame-progress guard, so the same instance can
replay the same stream to the same result.

Scripting
---------
The script maps a ``frame_index`` to an ordered sequence of
:class:`ScriptedAssignment`; the *i*-th assignment applies to the *i*-th detection
of that frame's ``update`` batch (detections arrive in the tracker's deterministic
input order). A populated frame with no script entry, or a count mismatch between
scripted assignments and detections, is stub misuse and raises
:class:`ScriptedAssignmentError`. An empty frame is inert: it returns an empty
result without consulting the script or advancing the frame-progress guard.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts import Detection, TrackState
from ..contracts.enums import TrackStatus
from .adapter import TrackAdapter
from .config import TrackerConfig
from .errors import ScriptedAssignmentError
from .interface import Tracker
from .raw import TrackAssignment
from .sequencing import FrameProgress, single_frame_key


@dataclass(frozen=True)
class ScriptedAssignment:
    """One scripted identity assignment for one detection of one frame.

    Pairs positionally with the frame's detections. ``track_id`` is the identity
    to stamp; ``status`` / ``tainted`` / ``velocity`` are the tracker-owned fields
    the stub cannot infer and the caller therefore declares.
    """

    track_id: str
    status: TrackStatus = TrackStatus.ACTIVE
    tainted: bool = False
    velocity: tuple[float, float] | None = None


class StubTracker(Tracker):
    """A ``Tracker`` that replays scripted per-frame identity assignments."""

    def __init__(
        self,
        script: Mapping[int, Sequence[ScriptedAssignment]] | None = None,
        *,
        config: TrackerConfig | None = None,
    ) -> None:
        self._script: dict[int, tuple[ScriptedAssignment, ...]] = {
            index: tuple(items) for index, items in (script or {}).items()
        }
        self._adapter = TrackAdapter(config)
        self._progress = FrameProgress()

    def update(self, detections: Sequence[Detection]) -> tuple[TrackState, ...]:
        """Return the scripted ``TrackState``s for this frame's detections.

        Raises:
            InconsistentDetectionBatchError: if ``detections`` span >1 frame.
            NonMonotonicFrameError: if the frame does not strictly advance.
            ScriptedAssignmentError: if the frame is unscripted or the script's
                assignment count does not match the detection count.
            MalformedAssignmentError: if a scripted assignment is invalid (empty
                ``track_id`` or malformed velocity).
        """

        key = single_frame_key(detections)
        if key is None:
            return ()  # empty frame: inert, no state change (P1-U8 empty->empty)
        self._progress.advance(key)

        scripted = self._script.get(key.frame_index)
        if scripted is None:
            raise ScriptedAssignmentError(
                f"no scripted assignments for frame_index={key.frame_index} "
                f"with {len(detections)} detection(s)"
            )
        if len(scripted) != len(detections):
            raise ScriptedAssignmentError(
                f"frame_index={key.frame_index}: script has {len(scripted)} "
                f"assignment(s) but frame has {len(detections)} detection(s)"
            )

        assignments = [
            TrackAssignment(
                track_id=entry.track_id,
                detection=detection,
                status=entry.status,
                tainted=entry.tainted,
                velocity=entry.velocity,
            )
            for entry, detection in zip(scripted, detections, strict=True)
        ]
        return self._adapter.adapt(assignments)

    def reset(self) -> None:
        """Clear the frame-progress guard so the same script can replay a stream."""

        self._progress.reset()

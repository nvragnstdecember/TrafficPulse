"""Temporal frame-sequencing helpers for stateful trackers (P1-U8).

Tracking is stateful and temporal: a tracker evolves its state one frame at a
time, and that evolution is only well defined if frames arrive as a single,
ordered stream. This module isolates the two sequencing invariants every tracker
behind the P1-U8 seam must uphold, so they are defined and tested once and reused
unchanged by the deterministic ``StubTracker`` now and by the real P1-U9 backend
later (no interface redesign):

1. **Single-frame batches.** :func:`single_frame_key` confirms all detections in
   one ``update`` batch share one ``(camera_id, frame_index, timestamp)`` identity
   and returns that :class:`FrameKey` (or ``None`` for an empty batch), rejecting a
   cross-frame batch with :class:`InconsistentDetectionBatchError`.
2. **Strictly ascending frames.** :class:`FrameProgress` remembers the last
   processed frame and rejects any frame that does not advance **both**
   ``frame_index`` and ``timestamp`` strictly, with :class:`NonMonotonicFrameError`.

Both checks are pure functions of their inputs (``FrameProgress`` of its call
history) -- no wall-clock, no randomness -- so tracker state stays a deterministic
function of an ordered frame stream.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..contracts import Detection
from .errors import InconsistentDetectionBatchError, NonMonotonicFrameError


@dataclass(frozen=True)
class FrameKey:
    """The shared identity of the one frame an ``update`` batch belongs to."""

    camera_id: str
    frame_index: int
    timestamp: datetime


def single_frame_key(detections: Sequence[Detection]) -> FrameKey | None:
    """Return the shared :class:`FrameKey` of ``detections``, or ``None`` if empty.

    Raises:
        InconsistentDetectionBatchError: if the detections do not all share the
            same ``(camera_id, frame_index, timestamp)`` -- i.e. the batch spans
            more than one frame.
    """

    if not detections:
        return None
    first = detections[0]
    key = FrameKey(first.camera_id, first.frame_index, first.timestamp)
    for detection in detections[1:]:
        if (
            detection.camera_id != key.camera_id
            or detection.frame_index != key.frame_index
            or detection.timestamp != key.timestamp
        ):
            raise InconsistentDetectionBatchError(
                "update batch mixes frames: expected all detections at "
                f"(camera_id={key.camera_id!r}, frame_index={key.frame_index}, "
                f"timestamp={key.timestamp.isoformat()}), got detection "
                f"{detection.detection_id!r} at (camera_id={detection.camera_id!r}, "
                f"frame_index={detection.frame_index}, "
                f"timestamp={detection.timestamp.isoformat()})"
            )
    return key


class FrameProgress:
    """A stateful guard that frames advance strictly in ascending order.

    Holds the last accepted ``(frame_index, timestamp)``. :meth:`advance` accepts
    a new frame only if it is strictly greater in both, and updates the held
    state; :meth:`reset` clears it so the same tracker instance can replay a
    stream from the start deterministically.
    """

    def __init__(self) -> None:
        self._last_frame_index: int | None = None
        self._last_timestamp: datetime | None = None

    def advance(self, key: FrameKey) -> None:
        """Record ``key`` as the current frame, or reject a non-advancing one.

        Raises:
            NonMonotonicFrameError: if ``key`` does not strictly advance both the
                frame index and the timestamp beyond the last accepted frame.
        """

        if self._last_frame_index is not None and key.frame_index <= self._last_frame_index:
            raise NonMonotonicFrameError(
                f"frame_index must strictly increase: got {key.frame_index} "
                f"after {self._last_frame_index}"
            )
        if self._last_timestamp is not None and key.timestamp <= self._last_timestamp:
            raise NonMonotonicFrameError(
                f"timestamp must strictly increase: got {key.timestamp.isoformat()} "
                f"after {self._last_timestamp.isoformat()}"
            )
        self._last_frame_index = key.frame_index
        self._last_timestamp = key.timestamp

    def reset(self) -> None:
        """Forget the last processed frame (for deterministic replay)."""

        self._last_frame_index = None
        self._last_timestamp = None

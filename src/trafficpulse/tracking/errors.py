"""Error taxonomy for the tracker-integration boundary (P1-U8).

Every tracker-integration failure raises a subclass of :class:`TrackerError`,
so callers depend only on this package's error types -- never on a tracker
framework's exceptions and never on a ``pydantic.ValidationError`` leaking out of
adapter conversion. This mirrors ``detector/errors.py`` (P1-U6): configuration
validation is the one deliberate exception -- it is performed by pydantic and
surfaces the idiomatic ``ValidationError`` (as the U5 scene contract and the
detector config do), because config construction is not part of the runtime
tracker->contract seam.

The four subclasses map one-to-one onto concrete, tested failure modes; there is
no speculative "backend unavailable" placeholder (a real backend and its failure
modes are P1-U9).
"""


class TrackerError(Exception):
    """Base class for all tracker-integration errors."""


class InconsistentDetectionBatchError(TrackerError):
    """One ``update`` call mixed detections from more than one frame.

    The single-stream update contract (P1-U8 card) consumes the detections of
    **one** frame: every detection in a batch must share the same
    ``(camera_id, frame_index, timestamp)`` identity. A batch that spans multiple
    frames is ambiguous about which frame the tracker is advancing to and is
    rejected before any track state is produced.
    """


class NonMonotonicFrameError(TrackerError):
    """A frame was fed out of ascending order (invalid update sequence).

    Tracking is stateful and temporal: frames must be fed in strictly ascending
    ``frame_index`` **and** strictly increasing ``timestamp``. Feeding a frame
    that does not advance both is an ambiguous state evolution (a replayed,
    reordered, or duplicated frame) and is rejected. This keeps tracker state a
    deterministic function of an ordered frame stream.
    """


class MalformedAssignmentError(TrackerError):
    """A track assignment could not be stamped into a valid ``TrackState``.

    Raised by the adapter when an assignment is structurally invalid -- an empty
    ``track_id`` (the frozen ``TrackState`` requires a non-empty one), or a
    ``velocity`` that is not a finite 2-tuple. Any originating
    ``pydantic.ValidationError`` is chained as ``__cause__``. Because the
    carried-through fields come from an already-valid frozen ``Detection``, this
    is the adapter's only real validation surface.
    """


class ScriptedAssignmentError(TrackerError):
    """The :class:`~trafficpulse.tracking.stub.StubTracker` script does not match
    the frame it was asked to process.

    Raised when a populated frame has no scripted entry, or when the number of
    scripted assignments for a frame does not equal the number of detections in
    that frame's batch (the stub assigns identities positionally). It signals a
    test/wiring mistake in the script, distinct from a malformed *assignment*.
    """

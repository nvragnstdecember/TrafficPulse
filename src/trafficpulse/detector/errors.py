"""Error taxonomy for the detector-integration boundary (P1-U6).

Every detector-integration failure raises a subclass of :class:`DetectorError`,
so callers depend only on this package's error types -- never on a detector
framework's exceptions and never on a ``pydantic.ValidationError`` leaking out of
adapter conversion. Configuration validation is the one deliberate exception: it
is performed by pydantic and surfaces the idiomatic ``ValidationError`` (as the
U5 scene contract does), because config construction is not part of the runtime
detector->contract seam.
"""


class DetectorError(Exception):
    """Base class for all detector-integration errors."""


class InvalidFrameError(DetectorError):
    """The :class:`~trafficpulse.detector.frame.Frame` identity is unusable.

    Raised by the adapter before any detection is converted when the frame's
    identity fields cannot stamp a valid ``Detection`` -- an empty ``camera_id``,
    a negative ``frame_index``, or a timestamp that is not timezone-aware. This is
    a caller/frame problem, kept distinct from a malformed *detector output*.
    """


class MalformedDetectorOutputError(DetectorError):
    """A detector emitted a structurally invalid ``RawDetection``.

    Raised by the adapter when a raw detection cannot be converted into a valid
    frozen ``Detection`` -- a non-finite or out-of-``[0, 1]`` score, a ``box`` that
    is not a finite 4-tuple, or box coordinates the ``BoundingBox`` contract
    rejects (negative, or ``x2 <= x1`` / ``y2 <= y1``). Any originating
    ``pydantic.ValidationError`` is chained as ``__cause__``. Unlike an unmodeled
    class (silently dropped), a malformed output is always rejected.
    """

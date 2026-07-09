"""The abstract tracker interface -- the dependency-injection seam (P1-U8).

``Tracker`` is the abstraction every tracker implementation satisfies and that
callers depend on instead of a concrete tracker. Injecting a ``Tracker`` (the
deterministic ``StubTracker`` in tests and P1-U10 orchestration, a real
detection-based tracker in P1-U9) is what keeps the permissive-only tracker
choice a bounded, localized change (ADR-001): downstream code depends on this
interface and on the frozen ``TrackState`` contract, never on a tracker
framework.

The update contract (single-stream, stateful, temporal)
------------------------------------------------------
``update`` consumes the ``Detection``s of **one** frame and returns the
``TrackState``s active at that frame, in a stable order. Tracking is stateful, so
frames must be fed as a single ordered stream, one ``update`` per frame in
strictly ascending ``frame_index`` / ``timestamp`` (implementations enforce this
via :mod:`trafficpulse.tracking.sequencing`). Frame identity travels *with* the
detections -- the tracker reads ``camera_id`` / ``frame_index`` / ``timestamp``
from them -- so no separate frame-metadata object (and no ingestion
``FrameRecord`` / PyAV dependency) is needed at this seam.

An empty batch is a valid frame with no detections; it returns an empty result.
The returned values are exclusively frozen ``TrackState`` contracts: no
tracker-native track object, matrix, or identifier escapes this boundary.

``reset`` returns the tracker to its initial state so an identical stream replays
to an identical result from one instance -- the determinism guarantee downstream
reasoning and P1-U10 orchestration rely on.

This foundation intentionally specifies *no* association logic (IoU, Kalman,
Hungarian, re-ID): that is a real backend's concern (P1-U9). The stub assigns
scripted identities; a real backend computes them; both satisfy this same seam.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..contracts import Detection, TrackState


class Tracker(ABC):
    """Abstract, framework-neutral stateful multi-object tracker."""

    @abstractmethod
    def update(self, detections: Sequence[Detection]) -> Sequence[TrackState]:
        """Advance the tracker by one frame and return that frame's ``TrackState``s.

        Implementations must be deterministic for a given construction and stream
        of ``update`` calls, must consume one frame's detections per call in
        ascending order, and must not let tracker-native objects escape: only
        frozen ``TrackState`` values cross this boundary.

        Raises:
            InconsistentDetectionBatchError: if ``detections`` span more than one
                frame.
            NonMonotonicFrameError: if the frame does not strictly follow the
                previously processed frame.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Return the tracker to its initial (pre-stream) state for replay."""
        raise NotImplementedError

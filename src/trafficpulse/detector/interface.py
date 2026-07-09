"""The abstract detector interface -- the dependency-injection seam (P1-U6).

``Detector`` is the abstraction every detector implementation satisfies and that
callers depend on instead of a concrete detector. Injecting a ``Detector`` (the
deterministic ``StubDetector`` in tests, a future RT-DETR detector in production)
is what keeps the permissive-only detector choice a bounded, localized change
(ADR-001): downstream code depends on this interface and on the frozen
``Detection`` contract, never on a detector framework.

``detect`` returns framework-neutral ``RawDetection`` values -- never tensors or
framework objects. Combining those with frame identity and validating them into
frozen ``Detection`` contracts is the adapter's job, not the detector's, which
keeps the detector implementation and the contract-stamping policy independent.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .frame import Frame
from .raw import RawDetection


class Detector(ABC):
    """Abstract, framework-neutral single-frame object detector."""

    @abstractmethod
    def detect(self, frame: Frame) -> Sequence[RawDetection]:
        """Return the detector-native detections for one frame.

        Implementations must be deterministic for a given input and must not let
        framework objects (tensors, model handles) escape: only ``RawDetection``
        values cross this boundary. A real detector runs inference over
        ``frame.image``; the stub ignores pixels and is a pure function of frame
        identity. This is a single-frame operation -- no batching, tracking, or
        temporal state lives here.
        """
        raise NotImplementedError

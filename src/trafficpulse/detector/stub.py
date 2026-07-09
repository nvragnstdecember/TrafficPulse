"""A deterministic stub detector for tests and wiring (P1-U6).

``StubDetector`` implements the ``Detector`` interface without any inference,
model, weights, or framework dependency: it *replays* a caller-supplied script of
``RawDetection`` values, keyed by ``frame.frame_index``. It exists so the adapter,
the dependency-injection seam, and any future detector-consuming code can be
exercised deterministically before a real detector exists. It ignores
``frame.image`` entirely.

Determinism: ``detect`` is a pure function of the script and ``frame.frame_index``
-- no wall-clock, no randomness, no global or per-call mutable state. Given a
frame index, it returns the frames-specific script if one was provided, otherwise
the shared ``default`` script. The returned sequence is an immutable tuple, so a
caller cannot mutate the stub's script through the result.
"""

from collections.abc import Mapping, Sequence

from .frame import Frame
from .interface import Detector
from .raw import RawDetection


class StubDetector(Detector):
    """A ``Detector`` that replays scripted ``RawDetection`` output per frame index."""

    def __init__(
        self,
        default: Sequence[RawDetection] = (),
        *,
        per_frame: Mapping[int, Sequence[RawDetection]] | None = None,
    ) -> None:
        self._default: tuple[RawDetection, ...] = tuple(default)
        self._per_frame: dict[int, tuple[RawDetection, ...]] = {
            index: tuple(items) for index, items in (per_frame or {}).items()
        }

    def detect(self, frame: Frame) -> Sequence[RawDetection]:
        """Return the scripted detections for ``frame.frame_index`` (else ``default``)."""

        return self._per_frame.get(frame.frame_index, self._default)

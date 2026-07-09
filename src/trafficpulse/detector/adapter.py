"""Deterministic conversion of detector output into frozen ``Detection`` contracts.

``DetectionAdapter`` is the boundary ADR-001 mandates: it consumes framework-
neutral ``RawDetection`` values plus frame identity and produces frozen U2
``Detection`` contracts, and it is the *only* place detector output becomes a
contract. Everything downstream (tracking, observations, rules, events, evidence)
consumes ``Detection`` only and never sees a ``RawDetection``, a label string, or
any detector framework type.

Conversion rules (deterministic, order-preserving)
--------------------------------------------------
For each ``RawDetection`` in emission order:

* **Label mapping.** ``label`` maps to a frozen ``ObjectClass`` via
  ``config.label_map``. A label absent from the map is a class this project does
  not model (detectors emit a superset vocabulary) and is **dropped** -- not an
  error.
* **Score validation + gating.** A non-finite or out-of-``[0, 1]`` ``score`` is
  *malformed* and rejected. A valid score below ``config.score_threshold`` is
  **dropped** (standard confidence gating; the default threshold ``0.0`` keeps
  all valid scores).
* **Box validation.** ``box`` must be a finite 4-tuple that the ``BoundingBox``
  contract accepts (non-negative, ``x2 > x1`` and ``y2 > y1``); otherwise it is
  *malformed* and rejected. The explicit finiteness pre-check matters: an
  infinite coordinate can slip past ``BoundingBox``'s ``> 0`` / ``x2 > x1`` guards.
* **Identity.** ``detection_id`` is a deterministic SHA-256 digest over
  ``(camera_id, frame_index, emission ordinal)`` -- stable across identical
  replays, unique within a frame, and independent of which detections are
  filtered out (the ordinal follows the detector's emission order, not the kept
  order). It is a source *label*, not evidence-integrity hashing.
* **Provenance.** ``source_model`` is stamped from ``config.source_model``.

Malformed outputs raise :class:`MalformedDetectorOutputError` (never a leaked
pydantic error); an unusable frame identity raises :class:`InvalidFrameError`.
Determinism: no wall-clock, no randomness -- every id and field is a pure function
of the frame identity, the emission order, and the raw detection.
"""

import hashlib
import math
from collections.abc import Iterable

from pydantic import ValidationError

from ..contracts import BoundingBox, Detection
from .config import DetectorConfig
from .errors import InvalidFrameError, MalformedDetectorOutputError
from .frame import Frame
from .interface import Detector
from .raw import RawDetection

_SEP = "\x1f"  # unit separator; avoids delimiter collisions in the id preimage


class DetectionAdapter:
    """Converts detector-native ``RawDetection`` output into frozen ``Detection``."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def adapt(self, frame: Frame, raw_detections: Iterable[RawDetection]) -> tuple[Detection, ...]:
        """Convert one frame's raw detections into frozen ``Detection`` contracts.

        Raises:
            InvalidFrameError: if the frame identity cannot stamp a ``Detection``.
            MalformedDetectorOutputError: if any raw detection is structurally
                invalid (a malformed output rejects the batch rather than being
                silently dropped, unlike an unmodeled class).
        """

        self._validate_frame(frame)
        results: list[Detection] = []
        for ordinal, raw in enumerate(raw_detections):
            detection = self._adapt_one(frame, ordinal, raw)
            if detection is not None:
                results.append(detection)
        return tuple(results)

    def adapt_from(self, detector: Detector, frame: Frame) -> tuple[Detection, ...]:
        """Run an injected ``Detector`` on ``frame`` and adapt its output.

        The dependency-injection seam in one call: the adapter depends on the
        ``Detector`` *abstraction*, so any implementation (stub now, RT-DETR later)
        drops in unchanged. It processes a single frame only -- it is deliberately
        not a video / tracking / rule pipeline.
        """

        return self.adapt(frame, detector.detect(frame))

    # --- per-detection conversion -------------------------------------------
    def _adapt_one(self, frame: Frame, ordinal: int, raw: RawDetection) -> Detection | None:
        object_class = self._config.label_map.get(raw.label)
        if object_class is None:
            return None  # unmodeled class: dropped, not malformed
        score = self._validated_score(frame, ordinal, raw)
        if score < self._config.score_threshold:
            return None  # confidence gating
        bbox = self._to_bbox(frame, ordinal, raw)
        try:
            return Detection(
                detection_id=self._detection_id(frame, ordinal),
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                object_class=object_class,
                confidence=score,
                bbox=bbox,
                source_model=self._config.source_model,
            )
        except ValidationError as exc:  # pragma: no cover - defensive: fields pre-validated
            raise self._malformed(frame, ordinal, raw, str(exc)) from exc

    def _validated_score(self, frame: Frame, ordinal: int, raw: RawDetection) -> float:
        score = raw.score
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise self._malformed(
                frame, ordinal, raw, f"score {score!r} is not a finite confidence in [0, 1]"
            )
        return score

    def _to_bbox(self, frame: Frame, ordinal: int, raw: RawDetection) -> BoundingBox:
        box = raw.box
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise self._malformed(frame, ordinal, raw, f"box {box!r} is not a finite 4-tuple")
        x1, y1, x2, y2 = box
        try:
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except ValidationError as exc:
            raise self._malformed(frame, ordinal, raw, str(exc)) from exc

    # --- frame validation & identity ----------------------------------------
    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        if not frame.camera_id:
            raise InvalidFrameError("frame.camera_id must be a non-empty string")
        if frame.frame_index < 0:
            raise InvalidFrameError(f"frame.frame_index must be >= 0, got {frame.frame_index}")
        tzinfo = frame.timestamp.tzinfo
        if tzinfo is None or tzinfo.utcoffset(frame.timestamp) is None:
            raise InvalidFrameError("frame.timestamp must be timezone-aware")

    @staticmethod
    def _detection_id(frame: Frame, ordinal: int) -> str:
        preimage = _SEP.join((frame.camera_id, str(frame.frame_index), str(ordinal)))
        return "det-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _malformed(
        frame: Frame, ordinal: int, raw: RawDetection, detail: str
    ) -> MalformedDetectorOutputError:
        return MalformedDetectorOutputError(
            f"malformed detector output at frame_index={frame.frame_index} "
            f"ordinal={ordinal} label={raw.label!r}: {detail}"
        )

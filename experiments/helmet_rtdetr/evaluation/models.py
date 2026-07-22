"""Prediction model + evaluation configuration (H5).

The label space is **reused**, not redefined: :data:`~helmet_rtdetr.rtdetr.data.LABEL_IDS`
(``helmet -> 0``, ``no_helmet -> 1``) is the single contract shared by training
(H4B) and evaluation, so the two can never drift. ``motorcycle`` and
``ignore``-flagged objects are context, never detection targets — exactly the
rule the H4B data layer applies when it builds training targets.

Boxes reuse the H2 :class:`~helmet_rtdetr.unified.BBox` (pixel COCO ``x,y,w,h``),
so ground truth (``UnifiedObject.bbox``) and predictions live in the same
geometry with the same validation.

Validation split (consistent with H1–H4): field-level bounds raise pydantic
``ValidationError``; cross-field semantic rules raise the typed
:class:`~helmet_rtdetr.errors.InvalidEvaluationConfigError` /
:class:`~helmet_rtdetr.errors.InvalidPredictionError`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from ..errors import InvalidEvaluationConfigError, InvalidPredictionError
from ..models import NonEmptyStr, _Model
from ..rtdetr.data import LABEL_IDS
from ..unified import BBox, UnifiedClass

# The evaluable classes, in model-label-id order (helmet=0, no_helmet=1). Derived
# from the shared training label map so evaluation can never disagree with H4B.
EVAL_CLASSES: tuple[UnifiedClass, ...] = tuple(
    sorted(LABEL_IDS, key=lambda label: LABEL_IDS[label])
)

# Model label id -> unified class (the decode direction; inverse of LABEL_IDS).
ID_TO_CLASS: dict[int, UnifiedClass] = {
    label_id: label for label, label_id in LABEL_IDS.items()
}

# The COCO IoU-threshold ladder 0.50:0.05:0.95 (rounded to 2 dp exactly as COCO
# enumerates them); mAP averages over it, AP50/AP75 read single rungs.
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * step, 2) for step in range(10)
)


class Prediction(_Model):
    """One immutable detector prediction on one image.

    ``image_id`` is the manifest-relative image path — the same identity ground
    truth uses (``UnifiedObject.image_path``), so predictions and ground truth
    join without any translation table.
    """

    image_id: NonEmptyStr
    label: UnifiedClass
    score: float = Field(ge=0.0, le=1.0)  # NaN fails the bound check by design
    bbox: BBox

    @model_validator(mode="after")
    def _evaluable_class(self) -> Self:
        if self.label not in LABEL_IDS:
            raise InvalidPredictionError(
                f"prediction class {self.label.value!r} is not a detector class "
                f"(evaluable: {[c.value for c in EVAL_CLASSES]})"
            )
        return self


class EvaluationConfig(_Model):
    """Everything one evaluation needs, declared and validated up front.

    * ``iou_thresholds`` — the AP ladder; mAP averages over it. AP50/AP75 are
      reported only when 0.5 / 0.75 are present (they are in the COCO default).
    * ``matching_iou`` + ``score_threshold`` — the single operating point at
      which precision/recall/F1, TP/FP/FN, and the confusion matrix are
      computed (AP needs no score threshold; it integrates over confidence).
    * ``max_detections`` — per-image cap, highest scores first (COCO uses 100).
    * ``decode_threshold`` / ``batch_size`` / ``device`` — used only by the
      checkpoint-inference path; pure prediction evaluation ignores them.
      ``decode_threshold`` defaults to 0.0 so AP sees the full confidence range.
    """

    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS
    matching_iou: float = Field(default=0.5, gt=0.0, lt=1.0)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    decode_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    batch_size: int = Field(default=8, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def _valid_thresholds(self) -> Self:
        if not self.iou_thresholds:
            raise InvalidEvaluationConfigError("iou_thresholds must not be empty")
        if any(not (0.0 < t < 1.0) for t in self.iou_thresholds):
            raise InvalidEvaluationConfigError(
                f"every IoU threshold must be in (0, 1), got {self.iou_thresholds}"
            )
        if any(
            b <= a for a, b in zip(self.iou_thresholds, self.iou_thresholds[1:], strict=False)
        ):
            raise InvalidEvaluationConfigError(
                f"iou_thresholds must be strictly increasing, got {self.iou_thresholds}"
            )
        return self

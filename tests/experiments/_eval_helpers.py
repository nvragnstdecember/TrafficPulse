"""Shared builders for the H5 evaluation tests.

Reuses the H4B fixture helpers (``unified`` ground-truth builder, image/split
factories) rather than redefining them. Uniquely named (``_eval_helpers``) for
pytest's prepend import mode.
"""

from __future__ import annotations

from _rtdetr_helpers import unified
from helmet_rtdetr.evaluation import EvaluationConfig, Prediction
from helmet_rtdetr.unified import BBox, UnifiedClass

__all__ = ["unified", "pred", "config_5075"]


def pred(
    image: str,
    label: UnifiedClass = UnifiedClass.HELMET,
    *,
    score: float = 0.9,
    box: tuple[float, float, float, float] = (8.0, 8.0, 24.0, 24.0),
) -> Prediction:
    x, y, w, h = box
    return Prediction(image_id=image, label=label, score=score, bbox=BBox(x=x, y=y, w=w, h=h))


def config_5075(**overrides: object) -> EvaluationConfig:
    """A two-rung ladder (0.5, 0.75) — small enough to reason about by hand."""

    return EvaluationConfig(iou_thresholds=(0.5, 0.75), **overrides)  # type: ignore[arg-type]

"""Detection confusion matrix over {helmet, no_helmet} + background (H5).

A pure 2x2 class matrix cannot represent detection errors: a false positive has
no actual class and a missed ground-truth box has no predicted class. The
standard detection formulation adds a virtual **background** bucket:

* matched prediction/ground-truth pair -> ``counts[actual][predicted]``
  (diagonal = class agreement, off-diagonal = inter-class confusion);
* unmatched ground truth -> ``counts[actual][background]`` (a miss);
* unmatched prediction -> ``counts[background][predicted]`` (a spurious box);
* ``counts[background][background]`` is structurally 0 (nothing to count).

Matching here is **class-agnostic** (localisation only, greedy, score-ordered,
at the operating point's IoU + score thresholds) — deliberately different from
the class-aware matching behind TP/FP/FN: a ``no_helmet`` box predicted on a
``helmet`` ground truth should surface as inter-class confusion, not dissolve
into one FP plus one FN. The diagonal may therefore differ slightly from the
class-aware TP counts; both views are reported and documented.

Row/column order is the model-label-id order (helmet, no_helmet) + background
last — deterministic because :data:`EVAL_CLASSES` is.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from pydantic import model_validator

from ..models import _Model
from ..unified import UnifiedObject
from .matching import cap_per_image, match_greedy, prediction_order
from .metrics import evaluable_ground_truth
from .models import EVAL_CLASSES, EvaluationConfig, Prediction

BACKGROUND_LABEL = "background"


class ConfusionMatrix(_Model):
    """Square actual x predicted count matrix; last label is ``background``."""

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    iou_threshold: float
    score_threshold: float

    @model_validator(mode="after")
    def _square_and_sane(self) -> Self:
        size = len(self.labels)
        if len(self.counts) != size or any(len(row) != size for row in self.counts):
            raise ValueError(f"counts must be a {size}x{size} matrix matching labels")
        if any(cell < 0 for row in self.counts for cell in row):
            raise ValueError("confusion counts must be non-negative")
        if self.labels and self.labels[-1] != BACKGROUND_LABEL:
            raise ValueError(f"the last label must be {BACKGROUND_LABEL!r}")
        return self

    def count(self, actual: str, predicted: str) -> int:
        """One cell by label names (raises ``ValueError`` on unknown labels)."""

        return self.counts[self.labels.index(actual)][self.labels.index(predicted)]

    @property
    def total(self) -> int:
        return sum(cell for row in self.counts for cell in row)


def build_confusion_matrix(
    predictions: Iterable[Prediction],
    ground_truth: Iterable[UnifiedObject],
    config: EvaluationConfig,
) -> ConfusionMatrix:
    """Build the matrix at the config's operating point (see module docstring).

    Inputs may arrive in any order; canonical ordering, ground-truth filtering,
    the per-image cap, and the score threshold are applied here — the same
    pipeline order the metrics use, so both views describe the same predictions.
    """

    ordered_gt = evaluable_ground_truth(ground_truth)
    operating = tuple(
        p
        for p in cap_per_image(prediction_order(predictions), config.max_detections)
        if p.score >= config.score_threshold
    )
    result = match_greedy(
        operating, ordered_gt, iou_threshold=config.matching_iou, class_aware=False
    )

    labels = tuple(label.value for label in EVAL_CLASSES) + (BACKGROUND_LABEL,)
    index_of = {label: index for index, label in enumerate(labels)}
    background = index_of[BACKGROUND_LABEL]
    size = len(labels)
    cells = [[0] * size for _ in range(size)]

    for p_index, g_index in result.matches:
        actual = index_of[ordered_gt[g_index].label.value]
        predicted = index_of[operating[p_index].label.value]
        cells[actual][predicted] += 1
    for g_index in result.unmatched_ground_truth:
        cells[index_of[ordered_gt[g_index].label.value]][background] += 1
    for p_index in result.unmatched_predictions:
        cells[background][index_of[operating[p_index].label.value]] += 1

    return ConfusionMatrix(
        labels=labels,
        counts=tuple(tuple(row) for row in cells),
        iou_threshold=config.matching_iou,
        score_threshold=config.score_threshold,
    )

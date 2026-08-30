"""Classification metrics for the CNN-vs-ViT comparison (P4-U5).

Pure Python and framework-free: the caller supplies plain sequences of labels and
scores, so the whole metric core runs in CI without torch, and every number is
auditable by hand. This mirrors H5's pure-core evaluator
(:mod:`helmet_rtdetr.evaluation.evaluator`), but for classification -- H5 itself
computes detection mAP over IoU-matched boxes and is not applicable here.

The ``None``-versus-``0.0`` convention (inherited from H5)
----------------------------------------------------------
* A class **absent from the ground truth** has no meaningful precision, recall or
  F1. Those are ``None`` and the class is excluded from the macro averages -- the
  same choice H5 makes for a class with no ground-truth boxes, and COCO's ``-1``.
* A class **present in the ground truth that the model never gets right** honestly
  scores ``0.0``. That is a real result, not missing data.
* When the model predicts a class zero times, precision is ``0.0`` rather than
  ``None``: recall is necessarily ``0.0`` too, so F1 is ``0.0`` either way, and
  ``0.0`` keeps the class inside the macro average where it belongs.

Primary metric
--------------
**macro-F1**, per architecture-review §12 -- the unweighted mean of per-class F1,
so the 28%-prevalence ``no_helmet`` class counts as much as the majority class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from helmet_rtdetr.models import _Model

from .errors import EmptyEvaluationError, MismatchedPredictionsError
from .labels import HelmetState

#: The label space, in the canonical order used by every confusion matrix.
CLASS_ORDER: tuple[str, ...] = (HelmetState.HELMET.value, HelmetState.NO_HELMET.value)

#: The class the violation rule cares about; PR-AUC is reported for it (§12).
POSITIVE_CLASS = HelmetState.NO_HELMET.value

#: Bin count for the expected-calibration-error estimate (§12: reliability diagram).
DEFAULT_CALIBRATION_BINS = 15


class ClassMetrics(_Model):
    """Per-class precision / recall / F1 at the arg-max operating point."""

    support: int
    predicted: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None


class ReliabilityBin(_Model):
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None


class ClassificationMetrics(_Model):
    """The full §12 metric set for one model on one evaluation slice."""

    samples: int
    accuracy: float
    macro_f1: float | None
    balanced_accuracy: float | None
    per_class: dict[str, ClassMetrics]
    #: ``confusion[true][predicted]`` -- counts, in :data:`CLASS_ORDER`.
    confusion: dict[str, dict[str, int]]
    pr_auc_no_helmet: float | None
    expected_calibration_error: float | None
    reliability: tuple[ReliabilityBin, ...]


def _check(truth: Sequence[str], predicted: Sequence[str]) -> None:
    if len(truth) != len(predicted):
        raise MismatchedPredictionsError(
            f"{len(truth)} labels but {len(predicted)} predictions"
        )
    if not truth:
        raise EmptyEvaluationError("cannot compute metrics over an empty evaluation set")
    unknown = sorted((set(truth) | set(predicted)) - set(CLASS_ORDER))
    if unknown:
        raise MismatchedPredictionsError(f"labels outside the class space: {unknown}")


def confusion_matrix(
    truth: Sequence[str], predicted: Sequence[str]
) -> dict[str, dict[str, int]]:
    """``matrix[true][predicted]`` counts, in :data:`CLASS_ORDER`."""

    _check(truth, predicted)
    matrix = {t: dict.fromkeys(CLASS_ORDER, 0) for t in CLASS_ORDER}
    for actual, guess in zip(truth, predicted, strict=True):
        matrix[actual][guess] += 1
    return matrix


def average_precision(truth: Sequence[str], scores: Sequence[float]) -> float | None:
    """Area under the precision-recall curve for :data:`POSITIVE_CLASS`.

    The step-wise estimate ``sum_n (R_n - R_{n-1}) * P_n`` over samples ranked by
    descending score -- no interpolation, so it neither flatters nor penalises a
    model at sparse operating points. Returns ``None`` when the positive class is
    absent (undefined, not zero).
    """

    if len(truth) != len(scores):
        raise MismatchedPredictionsError(f"{len(truth)} labels but {len(scores)} scores")
    positives = sum(1 for t in truth if t == POSITIVE_CLASS)
    if positives == 0:
        return None

    # Descending score; ties broken by label so the result is order-independent.
    ranked = sorted(zip(scores, truth, strict=True), key=lambda p: (-p[0], p[1]))
    seen_positive = 0
    area = 0.0
    previous_recall = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == POSITIVE_CLASS:
            seen_positive += 1
        recall = seen_positive / positives
        if recall > previous_recall:
            area += (recall - previous_recall) * (seen_positive / index)
            previous_recall = recall
    return area


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[float | None, tuple[ReliabilityBin, ...]]:
    """Equal-width ECE plus the reliability-diagram buckets.

    ``ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|``. An empty bucket
    contributes nothing and reports ``None`` for both of its rates rather than a
    fabricated zero.
    """

    if len(confidences) != len(correct):
        raise MismatchedPredictionsError(
            f"{len(confidences)} confidences but {len(correct)} outcomes"
        )
    if not confidences:
        return None, ()

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, hit in zip(confidences, correct, strict=True):
        # Confidence 1.0 belongs to the last bucket, not a phantom bucket past it.
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, hit))

    total = len(confidences)
    error = 0.0
    reliability: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        lower, upper = index / bins, (index + 1) / bins
        if not bucket:
            reliability.append(
                ReliabilityBin(
                    lower=lower, upper=upper, count=0, mean_confidence=None, accuracy=None
                )
            )
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, hit in bucket if hit) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - mean_confidence)
        reliability.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return error, tuple(reliability)


def compute_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    no_helmet_scores: Sequence[float] | None = None,
    confidences: Sequence[float] | None = None,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> ClassificationMetrics:
    """The full metric set for one model on one slice.

    ``no_helmet_scores`` are the model's probabilities for :data:`POSITIVE_CLASS`
    (used for PR-AUC); ``confidences`` are the arg-max probabilities (used for ECE).
    Both are optional -- the threshold-free metrics simply report ``None`` without
    them, rather than being silently skipped.
    """

    _check(truth, predicted)
    matrix = confusion_matrix(truth, predicted)

    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []
    recalls: list[float] = []
    for label in CLASS_ORDER:
        support = sum(matrix[label].values())
        true_positives = matrix[label][label]
        predicted_count = sum(matrix[other][label] for other in CLASS_ORDER)
        if support == 0:
            per_class[label] = ClassMetrics(
                support=0,
                predicted=predicted_count,
                true_positives=0,
                precision=None,
                recall=None,
                f1=None,
            )
            continue
        recall = true_positives / support
        precision = (true_positives / predicted_count) if predicted_count else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        )
        per_class[label] = ClassMetrics(
            support=support,
            predicted=predicted_count,
            true_positives=true_positives,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        f1_scores.append(f1)
        recalls.append(recall)

    total = len(truth)
    correct = sum(matrix[label][label] for label in CLASS_ORDER)

    ece: float | None = None
    reliability: tuple[ReliabilityBin, ...] = ()
    if confidences is not None:
        hits = [t == p for t, p in zip(truth, predicted, strict=True)]
        ece, reliability = expected_calibration_error(
            confidences, hits, bins=calibration_bins
        )

    return ClassificationMetrics(
        samples=total,
        accuracy=correct / total,
        macro_f1=(sum(f1_scores) / len(f1_scores)) if f1_scores else None,
        balanced_accuracy=(sum(recalls) / len(recalls)) if recalls else None,
        per_class=per_class,
        confusion=matrix,
        pr_auc_no_helmet=(
            average_precision(truth, no_helmet_scores) if no_helmet_scores is not None else None
        ),
        expected_calibration_error=ece,
        reliability=reliability,
    )


def predictions_from_scores(
    no_helmet_scores: Sequence[float],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Arg-max labels and their confidences from binary ``no_helmet`` probabilities.

    A score of exactly 0.5 resolves to ``helmet`` (strict ``>``), so the decision is
    total and reproducible rather than dependent on float noise -- the same
    tie-breaking discipline H5's matcher uses.
    """

    labels: list[str] = []
    confidences: list[float] = []
    for score in no_helmet_scores:
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise MismatchedPredictionsError(f"score {score!r} is not a probability")
        bare = score > 0.5
        labels.append(POSITIVE_CLASS if bare else HelmetState.HELMET.value)
        confidences.append(score if bare else 1.0 - score)
    return tuple(labels), tuple(confidences)

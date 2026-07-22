"""COCO-compatible detection metrics (H5).

Metric definitions, stated explicitly
-------------------------------------
* **AP (per class, per IoU threshold)** — 101-point interpolated average
  precision, exactly COCO's scheme: predictions sorted by descending score,
  greedy-matched (``matching.py``), precision taken through its monotone
  envelope and sampled at recalls ``0.00, 0.01, ..., 1.00``.
* **AP (per class)** — the mean over the configured IoU ladder (COCO default
  0.50:0.05:0.95). **mAP** — the mean over classes. **AP50/AP75** — the ladder
  rungs at IoU 0.50 / 0.75, averaged over classes.
* **Undefined vs zero**: a class with **no ground truth** has undefined AP
  (``None``) and is excluded from the mAP/AP50/AP75 means — COCO's convention
  (it marks such classes ``-1`` and excludes them). A class *with* ground truth
  but no correct predictions scores an honest ``0.0``. When *no* class has
  ground truth, mAP itself is ``None`` — never a fabricated 0.
* **Precision / recall / F1 / TP / FP / FN** — computed at one operating point:
  predictions with ``score >= score_threshold``, class-aware greedy matching at
  ``matching_iou``. Matching runs **on the filtered set** (filtering after
  matching would let sub-threshold predictions claim ground truth). Overall
  precision/recall/F1 are **micro-averaged** (from summed TP/FP/FN); per-class
  values use each class's own counts. Zero denominators yield 0.0.
* **Caps**: the per-image ``max_detections`` cap (COCO: 100) is applied before
  everything; ``num_predictions`` counts the evaluated (post-cap) predictions.
* **Ground-truth filtering**: ``ignore``-flagged objects and non-detector
  classes (``motorcycle``) are excluded from evaluation entirely — the same
  rule the H4B data layer applies to training targets.

Everything here is pure Python float arithmetic over canonically sorted
sequences: byte-determinism follows from input determinism.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence

from ..models import _Model
from ..rtdetr.data import LABEL_IDS
from ..unified import UnifiedClass, UnifiedObject
from .matching import cap_per_image, ground_truth_order, match_greedy, prediction_order
from .models import EVAL_CLASSES, EvaluationConfig, Prediction

_RECALL_SAMPLES = 101  # COCO's 101-point recall grid: 0.00, 0.01, ..., 1.00


def evaluable_ground_truth(objects: Iterable[UnifiedObject]) -> tuple[UnifiedObject, ...]:
    """Filter to detector-class, non-ignored objects, in canonical order."""

    return ground_truth_order(
        obj for obj in objects if not obj.ignore and obj.label in LABEL_IDS
    )


def interpolated_average_precision(
    tp_flags: Sequence[bool], num_ground_truth: int
) -> float | None:
    """101-point interpolated AP from score-ordered true-positive flags.

    ``tp_flags[i]`` says whether the i-th prediction (descending score) matched
    ground truth. Returns ``None`` when the class has no ground truth (AP is
    undefined), 0.0 when it has ground truth but nothing was predicted.
    """

    if num_ground_truth == 0:
        return None
    if not tp_flags:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    tp_cum = 0
    fp_cum = 0
    for flag in tp_flags:
        if flag:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / num_ground_truth)

    # Monotone precision envelope from the right (COCO's interpolation).
    envelope = list(precisions)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])

    total = 0.0
    for step in range(_RECALL_SAMPLES):
        recall_point = step / (_RECALL_SAMPLES - 1)
        index = bisect_left(recalls, recall_point)  # first recall >= the sample
        if index < len(envelope):
            total += envelope[index]
    return total / _RECALL_SAMPLES


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


class ClassMetrics(_Model):
    """One class's full metric set."""

    label: UnifiedClass
    ap: float | None
    ap50: float | None
    ap75: float | None
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    num_ground_truth: int
    num_predictions: int


class EvaluationMetrics(_Model):
    """Overall metrics plus the per-class breakdown (class-id order)."""

    mean_ap: float | None
    ap50: float | None
    ap75: float | None
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    num_ground_truth: int
    num_predictions: int
    per_class: tuple[ClassMetrics, ...]


def compute_metrics(
    predictions: Iterable[Prediction],
    ground_truth: Iterable[UnifiedObject],
    config: EvaluationConfig,
) -> EvaluationMetrics:
    """The full COCO-compatible metric set (see module docstring).

    Inputs may arrive in any order; canonical ordering, ground-truth filtering,
    and the per-image cap are applied here, so the result is a pure function of
    the *content* of the inputs.
    """

    ordered_gt = evaluable_ground_truth(ground_truth)
    ordered_predictions = cap_per_image(
        prediction_order(predictions), config.max_detections
    )

    gt_per_class = {label: 0 for label in EVAL_CLASSES}
    for obj in ordered_gt:
        gt_per_class[obj.label] += 1
    predictions_per_class = {label: 0 for label in EVAL_CLASSES}
    for prediction in ordered_predictions:
        predictions_per_class[prediction.label] += 1

    # --- AP over the IoU ladder (all classes share each matching pass) --------
    ap_by_class: dict[UnifiedClass, list[float | None]] = {c: [] for c in EVAL_CLASSES}
    for threshold in config.iou_thresholds:
        result = match_greedy(
            ordered_predictions, ordered_gt, iou_threshold=threshold, class_aware=True
        )
        matched = {p_index for p_index, _ in result.matches}
        for label in EVAL_CLASSES:
            tp_flags = [
                index in matched
                for index, prediction in enumerate(ordered_predictions)
                if prediction.label is label
            ]
            ap_by_class[label].append(
                interpolated_average_precision(tp_flags, gt_per_class[label])
            )

    def ladder_value(label: UnifiedClass, threshold: float) -> float | None:
        if threshold not in config.iou_thresholds:
            return None
        value = ap_by_class[label][config.iou_thresholds.index(threshold)]
        return value

    # --- the operating point (score >= threshold, IoU = matching_iou) ---------
    operating = tuple(
        p for p in ordered_predictions if p.score >= config.score_threshold
    )
    op_result = match_greedy(
        operating, ordered_gt, iou_threshold=config.matching_iou, class_aware=True
    )
    op_matched = {p_index for p_index, _ in op_result.matches}

    per_class: list[ClassMetrics] = []
    for label in EVAL_CLASSES:
        tp = sum(1 for i in op_matched if operating[i].label is label)
        fp = sum(
            1 for i in op_result.unmatched_predictions if operating[i].label is label
        )
        fn = sum(
            1 for i in op_result.unmatched_ground_truth if ordered_gt[i].label is label
        )
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        defined = [v for v in ap_by_class[label] if v is not None]
        per_class.append(
            ClassMetrics(
                label=label,
                ap=_mean(defined),
                ap50=ladder_value(label, 0.5),
                ap75=ladder_value(label, 0.75),
                precision=precision,
                recall=recall,
                f1=f1,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                num_ground_truth=gt_per_class[label],
                num_predictions=predictions_per_class[label],
            )
        )

    total_tp = sum(c.true_positives for c in per_class)
    total_fp = sum(c.false_positives for c in per_class)
    total_fn = sum(c.false_negatives for c in per_class)
    precision, recall, f1 = _precision_recall_f1(total_tp, total_fp, total_fn)
    return EvaluationMetrics(
        mean_ap=_mean([c.ap for c in per_class if c.ap is not None]),
        ap50=_mean([c.ap50 for c in per_class if c.ap50 is not None]),
        ap75=_mean([c.ap75 for c in per_class if c.ap75 is not None]),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        num_ground_truth=len(ordered_gt),
        num_predictions=len(ordered_predictions),
        per_class=tuple(per_class),
    )

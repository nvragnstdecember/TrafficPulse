"""COCO-compatible metric computation (H5). Pure math, no ML framework."""

from __future__ import annotations

import pytest
from _eval_helpers import config_5075, pred, unified
from helmet_rtdetr.errors import InvalidEvaluationConfigError, InvalidPredictionError
from helmet_rtdetr.evaluation import (
    COCO_IOU_THRESHOLDS,
    EvaluationConfig,
    Prediction,
    compute_metrics,
    interpolated_average_precision,
)
from helmet_rtdetr.unified import BBox, UnifiedClass
from pydantic import ValidationError


# --- interpolated AP -------------------------------------------------------------
def test_ap_is_undefined_without_ground_truth() -> None:
    assert interpolated_average_precision([True, False], 0) is None


def test_ap_is_zero_with_ground_truth_but_no_predictions() -> None:
    assert interpolated_average_precision([], 3) == 0.0


def test_ap_perfect_ranking_is_one() -> None:
    assert interpolated_average_precision([True, True], 2) == pytest.approx(1.0)


def test_ap_hand_computed_interpolation() -> None:
    # Flags [TP, FP, TP] over 2 ground truths:
    # precisions (1, 1/2, 2/3), recalls (0.5, 0.5, 1.0); envelope (1, 2/3, 2/3).
    # Samples 0.00-0.50 (51 of them) read 1.0; 0.51-1.00 (50) read 2/3.
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert interpolated_average_precision([True, False, True], 2) == pytest.approx(expected)


def test_ap_all_false_positives_is_zero() -> None:
    assert interpolated_average_precision([False, False], 2) == 0.0


# --- scenario: perfect predictions ------------------------------------------------
def test_perfect_predictions() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(10, 10, 20, 20)),
        unified("a.png", UnifiedClass.NO_HELMET, box=(50, 50, 20, 20)),
        unified("b.png", UnifiedClass.HELMET, box=(5, 5, 30, 30)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.95, box=(10, 10, 20, 20)),
        pred("a.png", UnifiedClass.NO_HELMET, score=0.9, box=(50, 50, 20, 20)),
        pred("b.png", UnifiedClass.HELMET, score=0.85, box=(5, 5, 30, 30)),
    ]
    metrics = compute_metrics(predictions, ground_truth, EvaluationConfig())
    assert metrics.mean_ap == pytest.approx(1.0)
    assert metrics.ap50 == pytest.approx(1.0)
    assert metrics.ap75 == pytest.approx(1.0)
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (3, 0, 0)


# --- scenario: all false positives -------------------------------------------------
def test_all_false_positives() -> None:
    predictions = [pred("a.png", score=0.9), pred("b.png", UnifiedClass.NO_HELMET, score=0.8)]
    metrics = compute_metrics(predictions, [], EvaluationConfig())
    assert metrics.mean_ap is None  # no ground truth anywhere: AP is undefined
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 2, 0)
    assert metrics.num_ground_truth == 0
    assert metrics.num_predictions == 2


# --- scenario: all false negatives -------------------------------------------------
def test_all_false_negatives() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("a.png", UnifiedClass.NO_HELMET, box=(20, 20, 10, 10)),
    ]
    metrics = compute_metrics([], ground_truth, EvaluationConfig())
    assert metrics.mean_ap == 0.0  # defined (there IS ground truth), honestly zero
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 0, 2)


# --- scenario: mixed -----------------------------------------------------------------
def test_mixed_predictions_per_class_counts() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),  # matched
        unified("a.png", UnifiedClass.NO_HELMET, box=(30, 30, 10, 10)),  # missed
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10)),  # TP
        pred("a.png", UnifiedClass.HELMET, score=0.8, box=(60, 60, 10, 10)),  # FP
    ]
    metrics = compute_metrics(predictions, ground_truth, EvaluationConfig())
    helmet, no_helmet = metrics.per_class
    assert helmet.label is UnifiedClass.HELMET
    assert (helmet.true_positives, helmet.false_positives, helmet.false_negatives) == (1, 1, 0)
    assert helmet.precision == pytest.approx(0.5)
    assert helmet.recall == pytest.approx(1.0)
    assert no_helmet.label is UnifiedClass.NO_HELMET
    assert (no_helmet.true_positives, no_helmet.false_positives, no_helmet.false_negatives) == (
        0,
        0,
        1,
    )
    # Micro-averaged overall: TP=1, FP=1, FN=1.
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


# --- scenario: empty everything ------------------------------------------------------
def test_empty_predictions_and_empty_ground_truth() -> None:
    metrics = compute_metrics([], [], EvaluationConfig())
    assert metrics.mean_ap is None
    assert metrics.ap50 is None
    assert metrics.ap75 is None
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 0, 0)
    assert metrics.num_predictions == 0
    for class_metrics in metrics.per_class:
        assert class_metrics.ap is None


# --- the IoU ladder --------------------------------------------------------------------
def test_localisation_quality_separates_ap50_from_ap75() -> None:
    # IoU(prediction, ground truth) = 0.6: counts at rungs 0.50-0.60, not above.
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 6))]
    metrics = compute_metrics(predictions, ground_truth, EvaluationConfig())
    helmet = metrics.per_class[0]
    assert helmet.ap50 == pytest.approx(1.0)
    assert helmet.ap75 == pytest.approx(0.0)
    assert helmet.ap == pytest.approx(3 / 10)  # rungs 0.50, 0.55, 0.60 of ten
    assert metrics.mean_ap == pytest.approx(3 / 10)  # no_helmet has no GT: excluded


def test_ladder_without_50_and_75_reports_none() -> None:
    config = EvaluationConfig(iou_thresholds=(0.6, 0.9))
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10))]
    metrics = compute_metrics(predictions, ground_truth, config)
    assert metrics.ap50 is None
    assert metrics.ap75 is None
    assert metrics.mean_ap == pytest.approx(1.0)


# --- operating point --------------------------------------------------------------------
def test_score_threshold_gates_counts_but_not_ap() -> None:
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.4, box=(0, 0, 10, 10))]
    metrics = compute_metrics(predictions, ground_truth, EvaluationConfig(score_threshold=0.5))
    assert metrics.true_positives == 0  # below the operating point
    assert metrics.false_negatives == 1
    assert metrics.ap50 == pytest.approx(1.0)  # AP integrates over confidence
    assert metrics.num_predictions == 1


def test_max_detections_caps_per_image() -> None:
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10)),
        pred("a.png", UnifiedClass.HELMET, score=0.8, box=(30, 30, 10, 10)),
    ]
    metrics = compute_metrics(
        predictions, ground_truth, EvaluationConfig(max_detections=1)
    )
    assert metrics.num_predictions == 1  # the 0.8 prediction was capped away
    assert metrics.false_positives == 0


def test_ignored_and_motorcycle_ground_truth_are_excluded() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10), ignore=True),
        unified("a.png", UnifiedClass.MOTORCYCLE, box=(20, 20, 40, 40)),
    ]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10))]
    metrics = compute_metrics(predictions, ground_truth, EvaluationConfig())
    assert metrics.num_ground_truth == 0
    assert metrics.false_positives == 1  # nothing evaluable to match against


def test_result_is_insertion_order_invariant() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("b.png", UnifiedClass.NO_HELMET, box=(5, 5, 10, 10)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(1, 0, 10, 10)),
        pred("b.png", UnifiedClass.NO_HELMET, score=0.7, box=(5, 5, 10, 10)),
        pred("a.png", UnifiedClass.HELMET, score=0.7, box=(0, 0, 10, 10)),
    ]
    config = config_5075()
    forward = compute_metrics(predictions, ground_truth, config)
    backward = compute_metrics(reversed(predictions), reversed(ground_truth), config)
    assert forward == backward


# --- validation ---------------------------------------------------------------------
def test_default_ladder_is_the_coco_one() -> None:
    assert COCO_IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    assert EvaluationConfig().iou_thresholds == COCO_IOU_THRESHOLDS


def test_empty_threshold_ladder_is_rejected() -> None:
    with pytest.raises(InvalidEvaluationConfigError, match="empty"):
        EvaluationConfig(iou_thresholds=())


def test_unsorted_threshold_ladder_is_rejected() -> None:
    with pytest.raises(InvalidEvaluationConfigError, match="strictly increasing"):
        EvaluationConfig(iou_thresholds=(0.75, 0.5))


def test_out_of_range_threshold_is_rejected() -> None:
    with pytest.raises(InvalidEvaluationConfigError, match="in \\(0, 1\\)"):
        EvaluationConfig(iou_thresholds=(0.5, 1.0))


def test_prediction_score_bounds_are_field_level() -> None:
    with pytest.raises(ValidationError):
        pred("a.png", score=1.5)
    with pytest.raises(ValidationError):
        pred("a.png", score=float("nan"))


def test_motorcycle_prediction_is_rejected() -> None:
    with pytest.raises(InvalidPredictionError, match="not a detector class"):
        Prediction(
            image_id="a.png",
            label=UnifiedClass.MOTORCYCLE,
            score=0.9,
            bbox=BBox(x=0, y=0, w=10, h=10),
        )

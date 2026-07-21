"""Detection confusion matrix (H5). Pure math, no ML framework."""

from __future__ import annotations

import pytest
from _eval_helpers import pred, unified
from helmet_rtdetr.evaluation import (
    BACKGROUND_LABEL,
    ConfusionMatrix,
    EvaluationConfig,
    build_confusion_matrix,
)
from helmet_rtdetr.unified import UnifiedClass
from pydantic import ValidationError


def test_perfect_predictions_fill_the_diagonal() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("a.png", UnifiedClass.NO_HELMET, box=(30, 30, 10, 10)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10)),
        pred("a.png", UnifiedClass.NO_HELMET, score=0.9, box=(30, 30, 10, 10)),
    ]
    matrix = build_confusion_matrix(predictions, ground_truth, EvaluationConfig())
    assert matrix.labels == ("helmet", "no_helmet", BACKGROUND_LABEL)
    assert matrix.count("helmet", "helmet") == 1
    assert matrix.count("no_helmet", "no_helmet") == 1
    assert matrix.total == 2


def test_class_confusion_lands_off_diagonal() -> None:
    # A well-localised box with the WRONG class: inter-class confusion, not FP+FN.
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.NO_HELMET, score=0.9, box=(0, 0, 10, 10))]
    matrix = build_confusion_matrix(predictions, ground_truth, EvaluationConfig())
    assert matrix.count("helmet", "no_helmet") == 1
    assert matrix.count("helmet", "helmet") == 0
    assert matrix.count(BACKGROUND_LABEL, "no_helmet") == 0


def test_missed_ground_truth_counts_against_background_column() -> None:
    ground_truth = [unified("a.png", UnifiedClass.NO_HELMET, box=(0, 0, 10, 10))]
    matrix = build_confusion_matrix([], ground_truth, EvaluationConfig())
    assert matrix.count("no_helmet", BACKGROUND_LABEL) == 1
    assert matrix.total == 1


def test_spurious_prediction_counts_against_background_row() -> None:
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.9)]
    matrix = build_confusion_matrix(predictions, [], EvaluationConfig())
    assert matrix.count(BACKGROUND_LABEL, "helmet") == 1
    assert matrix.count(BACKGROUND_LABEL, BACKGROUND_LABEL) == 0


def test_badly_localised_prediction_is_background_both_ways() -> None:
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.9, box=(50, 50, 10, 10))]
    matrix = build_confusion_matrix(predictions, ground_truth, EvaluationConfig())
    assert matrix.count("helmet", BACKGROUND_LABEL) == 1  # the miss
    assert matrix.count(BACKGROUND_LABEL, "helmet") == 1  # the spurious box


def test_sub_threshold_predictions_are_invisible() -> None:
    ground_truth = [unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))]
    predictions = [pred("a.png", UnifiedClass.HELMET, score=0.3, box=(0, 0, 10, 10))]
    matrix = build_confusion_matrix(
        predictions, ground_truth, EvaluationConfig(score_threshold=0.5)
    )
    assert matrix.count("helmet", "helmet") == 0
    assert matrix.count("helmet", BACKGROUND_LABEL) == 1


def test_empty_inputs_give_an_all_zero_matrix() -> None:
    matrix = build_confusion_matrix([], [], EvaluationConfig())
    assert matrix.total == 0
    assert len(matrix.labels) == 3


def test_matrix_is_insertion_order_invariant() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("b.png", UnifiedClass.NO_HELMET, box=(0, 0, 10, 10)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.NO_HELMET, score=0.9, box=(0, 0, 10, 10)),
        pred("b.png", UnifiedClass.NO_HELMET, score=0.8, box=(0, 0, 10, 10)),
    ]
    config = EvaluationConfig()
    forward = build_confusion_matrix(predictions, ground_truth, config)
    backward = build_confusion_matrix(reversed(predictions), reversed(ground_truth), config)
    assert forward == backward


def test_count_rejects_unknown_labels() -> None:
    matrix = build_confusion_matrix([], [], EvaluationConfig())
    with pytest.raises(ValueError):
        matrix.count("helmet", "bicycle")


def test_non_square_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="matrix"):
        ConfusionMatrix(
            labels=("helmet", "no_helmet", BACKGROUND_LABEL),
            counts=((0, 0), (0, 0)),
            iou_threshold=0.5,
            score_threshold=0.5,
        )


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ConfusionMatrix(
            labels=("helmet", "no_helmet", BACKGROUND_LABEL),
            counts=((0, 0, 0), (0, -1, 0), (0, 0, 0)),
            iou_threshold=0.5,
            score_threshold=0.5,
        )


def test_background_must_be_the_last_label() -> None:
    with pytest.raises(ValidationError, match="background"):
        ConfusionMatrix(
            labels=(BACKGROUND_LABEL, "helmet", "no_helmet"),
            counts=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            iou_threshold=0.5,
            score_threshold=0.5,
        )

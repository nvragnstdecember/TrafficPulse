"""Classification metrics for the CNN-vs-ViT comparison (P4-U5).

Every expectation here is computed by hand from a confusion matrix small enough to
verify by reading it -- the ``_eval_helpers.config_5075()`` philosophy H5 used for
its two-rung IoU ladder. A metric core checked only against itself is not checked.
"""

from __future__ import annotations

import pytest
from helmet_cnn_vit.errors import EmptyEvaluationError, MismatchedPredictionsError
from helmet_cnn_vit.metrics import (
    CLASS_ORDER,
    average_precision,
    compute_metrics,
    confusion_matrix,
    expected_calibration_error,
    predictions_from_scores,
)

HELMET, NO_HELMET = CLASS_ORDER


# --- confusion matrix ----------------------------------------------------------------


def test_the_confusion_matrix_is_true_by_predicted() -> None:
    truth = [HELMET, HELMET, NO_HELMET, NO_HELMET]
    predicted = [HELMET, NO_HELMET, NO_HELMET, HELMET]
    matrix = confusion_matrix(truth, predicted)
    assert matrix[HELMET] == {HELMET: 1, NO_HELMET: 1}
    assert matrix[NO_HELMET] == {HELMET: 1, NO_HELMET: 1}


def test_a_label_outside_the_class_space_is_refused() -> None:
    with pytest.raises(MismatchedPredictionsError, match="outside the class space"):
        confusion_matrix([HELMET, "turban"], [HELMET, HELMET])


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(MismatchedPredictionsError):
        confusion_matrix([HELMET], [HELMET, HELMET])


def test_an_empty_evaluation_is_refused() -> None:
    with pytest.raises(EmptyEvaluationError):
        compute_metrics([], [])


# --- hand-checked metric arithmetic -----------------------------------------------------


def test_metrics_on_a_hand_counted_confusion_matrix() -> None:
    """Six helmet crops, four no_helmet.

    Predicted: helmet 5/6 right (1 leaks to no_helmet); no_helmet 2/4 right.

    helmet    : TP=5 predicted=5+2=7  P=5/7   R=5/6   F1=2*(5/7)(5/6)/((5/7)+(5/6))
    no_helmet : TP=2 predicted=1+2=3  P=2/3   R=2/4
    """

    truth = [HELMET] * 6 + [NO_HELMET] * 4
    predicted = [HELMET] * 5 + [NO_HELMET] + [NO_HELMET] * 2 + [HELMET] * 2

    result = compute_metrics(truth, predicted)
    helmet = result.per_class[HELMET]
    bare = result.per_class[NO_HELMET]

    assert (helmet.support, helmet.true_positives, helmet.predicted) == (6, 5, 7)
    assert helmet.precision == pytest.approx(5 / 7)
    assert helmet.recall == pytest.approx(5 / 6)
    assert helmet.f1 == pytest.approx(2 * (5 / 7) * (5 / 6) / ((5 / 7) + (5 / 6)))

    assert (bare.support, bare.true_positives, bare.predicted) == (4, 2, 3)
    assert bare.precision == pytest.approx(2 / 3)
    assert bare.recall == pytest.approx(0.5)
    assert bare.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))

    assert result.accuracy == pytest.approx(7 / 10)
    assert result.macro_f1 == pytest.approx((helmet.f1 + bare.f1) / 2)
    assert result.balanced_accuracy == pytest.approx((5 / 6 + 0.5) / 2)


def test_macro_f1_is_not_accuracy_under_imbalance() -> None:
    """The reason §12 makes macro-F1 primary: a majority-only model looks good on
    accuracy and bad on macro-F1."""

    truth = [HELMET] * 90 + [NO_HELMET] * 10
    always_helmet = [HELMET] * 100
    result = compute_metrics(truth, always_helmet)

    assert result.accuracy == pytest.approx(0.90)
    assert result.per_class[NO_HELMET].recall == 0.0
    assert result.per_class[NO_HELMET].f1 == 0.0
    assert result.macro_f1 == pytest.approx(0.4736842, abs=1e-6)
    assert result.balanced_accuracy == pytest.approx(0.5)


def test_a_perfect_classifier_scores_one() -> None:
    truth = [HELMET, NO_HELMET, HELMET, NO_HELMET]
    result = compute_metrics(truth, list(truth))
    assert result.accuracy == 1.0
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)


# --- the None-versus-0.0 convention ---------------------------------------------------------


def test_a_class_absent_from_the_truth_reports_none_and_leaves_the_macro_average() -> None:
    """H5's convention: undefined is not zero."""

    truth = [HELMET, HELMET, HELMET]
    predicted = [HELMET, HELMET, NO_HELMET]
    result = compute_metrics(truth, predicted)

    bare = result.per_class[NO_HELMET]
    assert bare.support == 0
    assert bare.precision is None and bare.recall is None and bare.f1 is None
    assert bare.predicted == 1  # the spurious prediction is still counted
    # macro-F1 averages only the class that has ground truth.
    assert result.macro_f1 == pytest.approx(result.per_class[HELMET].f1)


def test_a_class_present_but_never_right_honestly_scores_zero() -> None:
    truth = [HELMET, NO_HELMET]
    predicted = [HELMET, HELMET]
    bare = compute_metrics(truth, predicted).per_class[NO_HELMET]
    assert bare.support == 1
    assert bare.recall == 0.0
    assert bare.precision == 0.0  # never predicted -> no credit, not undefined
    assert bare.f1 == 0.0


# --- PR-AUC ---------------------------------------------------------------------------------


def test_average_precision_is_one_for_a_perfect_ranking() -> None:
    truth = [NO_HELMET, NO_HELMET, HELMET, HELMET]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert average_precision(truth, scores) == pytest.approx(1.0)


def test_average_precision_on_a_hand_computed_ranking() -> None:
    """Ranked: NO(1/1), H, NO(2/3). AP = 1*(1/1) + 0 + (1/2)*(2/3) ... stepwise:

    at rank 1 recall 1/2, precision 1/1 -> contributes (1/2 - 0) * 1 = 0.5
    at rank 2 no positive, recall unchanged -> contributes 0
    at rank 3 recall 2/2, precision 2/3 -> contributes (1 - 1/2) * 2/3 = 1/3
    """

    truth = [NO_HELMET, HELMET, NO_HELMET]
    scores = [0.9, 0.8, 0.7]
    assert average_precision(truth, scores) == pytest.approx(0.5 + 1 / 3)


def test_average_precision_is_none_when_the_positive_class_is_absent() -> None:
    assert average_precision([HELMET, HELMET], [0.1, 0.2]) is None


# --- calibration -------------------------------------------------------------------------------


def test_a_perfectly_calibrated_set_has_zero_calibration_error() -> None:
    """Confidence 1.0 and always correct: no gap between confidence and accuracy."""

    error, _ = expected_calibration_error([1.0] * 10, [True] * 10, bins=10)
    assert error == pytest.approx(0.0)


def test_a_confidently_wrong_model_has_maximal_calibration_error() -> None:
    error, _ = expected_calibration_error([1.0] * 10, [False] * 10, bins=10)
    assert error == pytest.approx(1.0)


def test_confidence_of_exactly_one_lands_in_the_last_bin() -> None:
    _, bins = expected_calibration_error([1.0], [True], bins=10)
    assert bins[-1].count == 1
    assert sum(b.count for b in bins) == 1


def test_empty_reliability_bins_report_none_not_zero() -> None:
    _, bins = expected_calibration_error([0.95], [True], bins=10)
    empty = [b for b in bins if b.count == 0]
    assert empty
    assert all(b.accuracy is None and b.mean_confidence is None for b in empty)


def test_calibration_error_is_the_weighted_gap() -> None:
    """Half at confidence 0.9 all correct, half at 0.9 all wrong:
    one bin, mean confidence 0.9, accuracy 0.5, ECE = |0.5 - 0.9| = 0.4."""

    error, _ = expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5, bins=10)
    assert error == pytest.approx(0.4)


# --- score -> label ------------------------------------------------------------------------------


def test_scores_become_labels_with_a_total_tie_break() -> None:
    labels, confidences = predictions_from_scores([0.9, 0.1, 0.5])
    assert labels == (NO_HELMET, HELMET, HELMET)  # 0.5 resolves to helmet (strict >)
    assert confidences == pytest.approx((0.9, 0.9, 0.5))


def test_a_score_outside_zero_to_one_is_refused() -> None:
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(MismatchedPredictionsError):
            predictions_from_scores([bad])


def test_the_full_metric_set_threads_scores_through() -> None:
    truth = [NO_HELMET, HELMET, NO_HELMET, HELMET]
    scores = [0.8, 0.2, 0.6, 0.3]
    labels, confidences = predictions_from_scores(scores)
    result = compute_metrics(
        truth, labels, no_helmet_scores=scores, confidences=confidences
    )
    assert result.samples == 4
    assert result.pr_auc_no_helmet == pytest.approx(1.0)
    assert result.expected_calibration_error is not None
    assert sum(b.count for b in result.reliability) == 4


def test_threshold_free_metrics_are_none_when_scores_are_not_supplied() -> None:
    result = compute_metrics([HELMET, NO_HELMET], [HELMET, NO_HELMET])
    assert result.pr_auc_no_helmet is None
    assert result.expected_calibration_error is None
    assert result.reliability == ()

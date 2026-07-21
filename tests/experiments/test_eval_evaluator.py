"""HelmetEvaluator's pure evaluation path (H5). No ML framework."""

from __future__ import annotations

import pytest
from _eval_helpers import pred, unified
from helmet_rtdetr.errors import InvalidPredictionError
from helmet_rtdetr.evaluation import EvaluationConfig, HelmetEvaluator
from helmet_rtdetr.unified import UnifiedClass


def test_default_config_is_the_coco_one() -> None:
    evaluator = HelmetEvaluator()
    assert evaluator.config == EvaluationConfig()


def test_explicit_universe_counts_negative_images() -> None:
    ground_truth = [unified("a.png", UnifiedClass.HELMET)]
    report = HelmetEvaluator().evaluate(
        [], ground_truth, image_ids=["a.png", "negative.png"]
    )
    assert report.dataset.num_images == 2
    assert report.dataset.num_ground_truth == 1
    assert report.dataset.ground_truth_per_class == {"helmet": 1, "no_helmet": 0}


def test_explicit_universe_rejects_predictions_on_unknown_images() -> None:
    with pytest.raises(InvalidPredictionError, match="unknown image"):
        HelmetEvaluator().evaluate(
            [pred("stray.png")], [unified("a.png", UnifiedClass.HELMET)], image_ids=["a.png"]
        )


def test_false_positive_on_a_declared_negative_image_counts() -> None:
    report = HelmetEvaluator().evaluate(
        [pred("negative.png", score=0.9)],
        [unified("a.png", UnifiedClass.HELMET)],
        image_ids=["a.png", "negative.png"],
    )
    assert report.metrics.false_positives == 1
    assert report.metrics.false_negatives == 1  # a.png's box was never predicted


def test_inferred_universe_accepts_any_prediction_images() -> None:
    report = HelmetEvaluator().evaluate([pred("anywhere.png")], [])
    assert report.dataset.num_images == 1
    assert report.metrics.false_positives == 1


def test_report_carries_the_evaluators_config() -> None:
    config = EvaluationConfig(score_threshold=0.25, max_detections=10)
    report = HelmetEvaluator(config).evaluate([], [unified("a.png", UnifiedClass.HELMET)])
    assert report.config == config
    assert report.checkpoint is None
    assert report.dataset.split is None


def test_per_class_rows_are_in_model_label_id_order() -> None:
    report = HelmetEvaluator().evaluate([], [unified("a.png", UnifiedClass.NO_HELMET)])
    assert [c.label for c in report.metrics.per_class] == [
        UnifiedClass.HELMET,
        UnifiedClass.NO_HELMET,
    ]


# --- decoded-detection conversion (pure; the checkpoint path's door) ----------------
def test_decoded_prediction_clamps_to_image_bounds() -> None:
    from helmet_rtdetr.evaluation.evaluator import _decoded_prediction

    prediction = _decoded_prediction(
        "a.png", score=0.8, label_id=0, xyxy=(-5.0, -2.0, 70.0, 30.0), width=64, height=64
    )
    assert prediction is not None
    assert (prediction.bbox.x, prediction.bbox.y) == (0.0, 0.0)
    assert (prediction.bbox.x2, prediction.bbox.y2) == (64.0, 30.0)
    assert prediction.label is UnifiedClass.HELMET


def test_decoded_prediction_drops_degenerate_boxes() -> None:
    from helmet_rtdetr.evaluation.evaluator import _decoded_prediction

    # Entirely outside the image: zero area after clamping.
    assert (
        _decoded_prediction(
            "a.png", score=0.8, label_id=1, xyxy=(70.0, 70.0, 80.0, 80.0), width=64, height=64
        )
        is None
    )


def test_decoded_prediction_rejects_foreign_label_ids() -> None:
    from helmet_rtdetr.evaluation.evaluator import _decoded_prediction

    with pytest.raises(InvalidPredictionError, match="label id 7"):
        _decoded_prediction(
            "a.png", score=0.8, label_id=7, xyxy=(0.0, 0.0, 10.0, 10.0), width=64, height=64
        )

"""IoU + deterministic greedy matching (H5). Pure math, no ML framework."""

from __future__ import annotations

import pytest
from _eval_helpers import pred, unified
from helmet_rtdetr.evaluation import (
    cap_per_image,
    ground_truth_order,
    iou,
    match_greedy,
    prediction_order,
)
from helmet_rtdetr.unified import BBox, UnifiedClass


def box(x: float, y: float, w: float, h: float) -> BBox:
    return BBox(x=x, y=y, w=w, h=h)


# --- IoU -----------------------------------------------------------------------
def test_iou_identical_boxes_is_one() -> None:
    assert iou(box(10, 10, 20, 20), box(10, 10, 20, 20)) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou(box(0, 0, 10, 10), box(50, 50, 10, 10)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0


def test_iou_half_overlap_known_value() -> None:
    # inter 10x20=200; union 400+400-200=600
    assert iou(box(10, 10, 20, 20), box(20, 10, 20, 20)) == pytest.approx(1 / 3)


def test_iou_containment_known_value() -> None:
    assert iou(box(0, 0, 10, 10), box(0, 0, 20, 20)) == pytest.approx(0.25)


def test_iou_is_symmetric() -> None:
    a, b = box(0, 0, 15, 10), box(5, 5, 20, 20)
    assert iou(a, b) == iou(b, a)


# --- canonical orders -----------------------------------------------------------
def test_prediction_order_is_score_descending_with_content_tiebreak() -> None:
    low = pred("a.png", score=0.3)
    high = pred("a.png", score=0.9)
    tie_b = pred("b.png", score=0.5)
    tie_a = pred("a.png", score=0.5)
    ordered = prediction_order([low, tie_b, high, tie_a])
    assert ordered == (high, tie_a, tie_b, low)


def test_prediction_order_is_insertion_invariant() -> None:
    predictions = [pred("a.png", score=s) for s in (0.1, 0.9, 0.5)]
    assert prediction_order(predictions) == prediction_order(reversed(predictions))


def test_ground_truth_order_is_insertion_invariant() -> None:
    objects = [
        unified("b.png", UnifiedClass.HELMET),
        unified("a.png", UnifiedClass.NO_HELMET),
        unified("a.png", UnifiedClass.HELMET),
    ]
    ordered = ground_truth_order(objects)
    assert ordered == ground_truth_order(reversed(objects))
    assert [o.image_path for o in ordered] == ["a.png", "a.png", "b.png"]
    assert ordered[0].label is UnifiedClass.HELMET  # class id 0 before 1 within an image


# --- per-image cap ---------------------------------------------------------------
def test_cap_per_image_keeps_highest_scored() -> None:
    predictions = prediction_order(
        [
            pred("a.png", score=0.9, box=(0, 0, 10, 10)),
            pred("a.png", score=0.5, box=(20, 20, 10, 10)),
            pred("b.png", score=0.4, box=(0, 0, 10, 10)),
        ]
    )
    capped = cap_per_image(predictions, 1)
    assert [p.image_id for p in capped] == ["a.png", "b.png"]
    assert capped[0].score == 0.9


# --- greedy matching --------------------------------------------------------------
def test_higher_score_claims_the_ground_truth() -> None:
    gt = ground_truth_order([unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))])
    predictions = prediction_order(
        [
            pred("a.png", score=0.9, box=(0, 0, 10, 10)),
            pred("a.png", score=0.8, box=(0, 0, 10, 10)),
        ]
    )
    result = match_greedy(predictions, gt, iou_threshold=0.5)
    assert result.matches == ((0, 0),)  # the 0.9 prediction, in canonical order
    assert result.unmatched_predictions == (1,)
    assert result.unmatched_ground_truth == ()


def test_iou_below_threshold_never_matches() -> None:
    gt = ground_truth_order([unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))])
    predictions = prediction_order([pred("a.png", score=0.9, box=(8, 8, 10, 10))])
    result = match_greedy(predictions, gt, iou_threshold=0.5)
    assert result.matches == ()
    assert result.unmatched_predictions == (0,)
    assert result.unmatched_ground_truth == (0,)


def test_iou_exactly_at_threshold_matches() -> None:
    # IoU((0,0,10,10), (0,0,10,5)) = 50/100 = 0.5 exactly
    gt = ground_truth_order([unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))])
    predictions = prediction_order([pred("a.png", score=0.9, box=(0, 0, 10, 5))])
    assert match_greedy(predictions, gt, iou_threshold=0.5).matches == ((0, 0),)


def test_prediction_takes_highest_iou_candidate() -> None:
    gt = ground_truth_order(
        [
            unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
            unified("a.png", UnifiedClass.HELMET, box=(2, 0, 10, 10)),
        ]
    )
    predictions = prediction_order([pred("a.png", score=0.9, box=(2, 0, 10, 10))])
    result = match_greedy(predictions, gt, iou_threshold=0.3)
    matched_gt = gt[result.matches[0][1]]
    assert matched_gt.bbox.x == 2  # the perfect-overlap box, not the offset one


def test_iou_tie_resolves_to_lowest_canonical_index() -> None:
    # Both ground-truth boxes overlap the prediction with IoU 1/3.
    gt = ground_truth_order(
        [
            unified("a.png", UnifiedClass.HELMET, box=(5, 0, 10, 10)),
            unified("a.png", UnifiedClass.HELMET, box=(15, 0, 10, 10)),
        ]
    )
    predictions = prediction_order([pred("a.png", score=0.9, box=(10, 0, 10, 10))])
    result = match_greedy(predictions, gt, iou_threshold=0.2)
    assert result.matches == ((0, 0),)  # index 0 in canonical ground-truth order


def test_class_aware_matching_respects_classes() -> None:
    gt = ground_truth_order([unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))])
    predictions = prediction_order(
        [pred("a.png", UnifiedClass.NO_HELMET, score=0.9, box=(0, 0, 10, 10))]
    )
    aware = match_greedy(predictions, gt, iou_threshold=0.5, class_aware=True)
    agnostic = match_greedy(predictions, gt, iou_threshold=0.5, class_aware=False)
    assert aware.matches == ()
    assert agnostic.matches == ((0, 0),)


def test_matching_never_crosses_images() -> None:
    gt = ground_truth_order([unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10))])
    predictions = prediction_order([pred("b.png", score=0.9, box=(0, 0, 10, 10))])
    assert match_greedy(predictions, gt, iou_threshold=0.5).matches == ()


def test_empty_inputs_produce_empty_result() -> None:
    result = match_greedy((), (), iou_threshold=0.5)
    assert result.matches == ()
    assert result.unmatched_predictions == ()
    assert result.unmatched_ground_truth == ()

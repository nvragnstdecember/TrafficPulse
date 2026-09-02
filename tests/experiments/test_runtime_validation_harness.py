"""P4-U6-V harness logic: the parts that decide what a number means.

These are the pure functions whose bugs would silently corrupt the comparison -- the
GT-matching geometry, the two operating points, the abstention accounting, and the
threshold-selection rule. They are tested on fixtures rather than on the real corpus so the
suite stays fast and needs neither the dataset nor a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.analyse import (  # noqa: E402
    ABSTAIN,
    COVERAGE_FLOOR,
    THRESHOLD_GRID,
    evaluate,
    forced_binary,
    native,
    select_threshold,
)
from helmet_runtime_validation.derive import GT_MATCH_IOU, iou  # noqa: E402

from trafficpulse.contracts import BoundingBox  # noqa: E402


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


# --- ground-truth matching geometry ------------------------------------------
def test_identical_boxes_have_iou_one() -> None:
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero() -> None:
    assert iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_touching_boxes_have_iou_zero_not_a_division_error() -> None:
    assert iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0


def test_half_overlap_is_one_third() -> None:
    # Two 10x10 boxes overlapping in a 5x10 strip: inter 50, union 150.
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_the_match_threshold_is_the_pre_committed_value() -> None:
    """PROTOCOL §3 fixes 0.50 a priori; drift here would move every recovery number."""

    assert GT_MATCH_IOU == 0.50


# --- operating point A: forced binary ----------------------------------------
def test_forced_binary_ignores_non_binary_mass() -> None:
    """Zero-shot's turban/uncertain mass must not decide a binary comparison."""

    label, score = forced_binary(
        {"helmet": 0.30, "no_helmet": 0.10, "turban": 0.55, "uncertain": 0.05}
    )
    assert label == "helmet"
    assert score == pytest.approx(0.75)  # 0.30 / (0.30 + 0.10)


def test_forced_binary_renormalises_to_one() -> None:
    _, score_a = forced_binary({"helmet": 0.2, "no_helmet": 0.6})
    assert score_a == pytest.approx(0.75)


def test_forced_binary_never_abstains() -> None:
    for probabilities in ({"helmet": 0.5, "no_helmet": 0.5}, {"turban": 0.9, "helmet": 0.1}):
        assert forced_binary(probabilities)[0] in {"helmet", "no_helmet"}


# --- operating point B: native -----------------------------------------------
def test_zeroshot_turban_abstains_and_is_never_no_helmet() -> None:
    """The whole safety point: a turban must not become a violation by relabelling."""

    label, _ = native("zeroshot", {"turban": 0.7, "helmet": 0.2, "no_helmet": 0.1}, None)
    assert label == ABSTAIN


def test_zeroshot_uncertain_abstains() -> None:
    assert native("zeroshot", {"uncertain": 0.6, "helmet": 0.4}, None)[0] == ABSTAIN


def test_zeroshot_binary_prediction_stands() -> None:
    assert native("zeroshot", {"no_helmet": 0.8, "helmet": 0.2}, None)[0] == "no_helmet"


def test_a_binary_backend_abstains_below_the_threshold() -> None:
    assert native("resnet", {"helmet": 0.55, "no_helmet": 0.45}, 0.6)[0] == ABSTAIN


def test_a_binary_backend_commits_at_or_above_the_threshold() -> None:
    assert native("resnet", {"helmet": 0.60, "no_helmet": 0.40}, 0.6)[0] == "helmet"


def test_no_threshold_means_always_commit() -> None:
    assert native("deit", {"helmet": 0.51, "no_helmet": 0.49}, None)[0] == "helmet"


# --- abstention accounting ---------------------------------------------------
def test_coverage_counts_abstentions_in_the_denominator() -> None:
    """Denominator honesty: an abstention is not a removed crop."""

    result = evaluate(
        ["helmet", "no_helmet", "helmet", "no_helmet"],
        ["helmet", "no_helmet", ABSTAIN, ABSTAIN],
        [0.9, 0.9, 0.4, 0.4],
    )
    assert result["total"] == 4
    assert result["covered"] == 2
    assert result["abstained"] == 2
    assert result["coverage"] == pytest.approx(0.5)


def test_metrics_are_computed_only_over_covered_crops() -> None:
    result = evaluate(
        ["helmet", "no_helmet"], ["helmet", ABSTAIN], [0.9, 0.3]
    )
    assert result["covered"] == 1
    assert result["metrics"] is not None


def test_total_abstention_reports_no_metrics_rather_than_a_perfect_score() -> None:
    """Abstaining on everything must not look like flawless accuracy."""

    result = evaluate(["helmet", "no_helmet"], [ABSTAIN, ABSTAIN], [0.1, 0.1])
    assert result["coverage"] == 0.0
    assert result["metrics"] is None


# --- threshold selection rule ------------------------------------------------
def test_the_grid_is_the_pre_committed_one() -> None:
    assert THRESHOLD_GRID[0] is None
    assert THRESHOLD_GRID[1] == 0.50
    assert THRESHOLD_GRID[-1] == 0.95
    assert COVERAGE_FLOOR == 0.90


def test_selection_refuses_a_threshold_that_breaches_the_coverage_floor() -> None:
    """A threshold that abstains on most crops can never be selected, however accurate."""

    crop_ids = [f"c{i}" for i in range(10)]
    truth = ["helmet"] * 5 + ["no_helmet"] * 5
    # Every crop is a weak, correct call: a high threshold would abstain on all of them.
    probabilities = {
        crop_id: (
            {"helmet": 0.55, "no_helmet": 0.45}
            if t == "helmet"
            else {"helmet": 0.45, "no_helmet": 0.55}
        )
        for crop_id, t in zip(crop_ids, truth, strict=True)
    }
    selection = select_threshold(truth, probabilities, crop_ids, "resnet")
    assert selection["selected"] in (None, 0.50)
    breached = [g for g in selection["grid"] if g["threshold"] == 0.95]
    assert breached[0]["coverage"] < COVERAGE_FLOOR


def test_selection_records_the_whole_grid_not_only_the_winner() -> None:
    """The trace is the audit trail for 'we did not tune on test'."""

    crop_ids = ["a", "b"]
    truth = ["helmet", "no_helmet"]
    probabilities = {
        "a": {"helmet": 0.99, "no_helmet": 0.01},
        "b": {"helmet": 0.01, "no_helmet": 0.99},
    }
    selection = select_threshold(truth, probabilities, crop_ids, "resnet")
    assert len(selection["grid"]) == len(THRESHOLD_GRID)
    assert "coverage >= 0.90" in selection["rule"]


# --- exact McNemar: log-space form vs the frozen one -------------------------
def test_log_space_mcnemar_matches_the_frozen_implementation() -> None:
    """Same test, different arithmetic. It must agree wherever the frozen one can run."""

    import random

    from helmet_cnn_vit.stats import mcnemar
    from helmet_runtime_validation.analyse import exact_mcnemar

    random.seed(0)
    for n in (1, 10, 50, 200, 400):
        a = [random.random() < 0.7 for _ in range(n)]
        b = [random.random() < 0.6 for _ in range(n)]
        assert exact_mcnemar(a, b)["p_value"] == pytest.approx(mcnemar(a, b).p_value)


def test_log_space_mcnemar_survives_a_discordant_count_that_overflows_the_frozen_one() -> None:
    """The reason it exists: >1023 discordant crops overflow ``2.0**n`` in float."""

    from helmet_cnn_vit.stats import mcnemar
    from helmet_runtime_validation.analyse import exact_mcnemar

    a = [True] * 1200 + [False] * 400
    b = [False] * 1200 + [True] * 400
    with pytest.raises(OverflowError):
        mcnemar(a, b)
    result = exact_mcnemar(a, b)
    assert result["discordant"] == 1600
    assert 0.0 <= result["p_value"] <= 1.0


def test_identical_models_give_a_p_value_of_one() -> None:
    from helmet_runtime_validation.analyse import exact_mcnemar

    outcomes = [True, False, True, True]
    assert exact_mcnemar(outcomes, outcomes)["p_value"] == 1.0


# --- P4-U7 recovery taxonomy -------------------------------------------------
# The four buckets must stay distinct: collapsing them would hide whether the 46%
# loss is an operating-point, a capability, a matching, or an association problem.
def _frame(eligible_box, motorcycles, persons, *, gt_all=None):
    return {
        "video_id": "v",
        "frame_index": 1,
        "gt_all": gt_all if gt_all is not None else [{"track_id": "t", "box": eligible_box}],
        "eligible": [
            {
                "crop_id": "c1",
                "split": "val",
                "track_id": "t",
                "label": "helmet",
                "site_id": "s",
                "box": eligible_box,
            }
        ],
        "motorcycles": motorcycles,
        "persons": persons,
    }


def test_bucket_a_when_nothing_overlaps_the_ground_truth() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    record = _frame([0, 0, 100, 100], [{"box": [500, 500, 600, 600], "score": 0.9}], [])
    assert classify_frame(record, 0.5)[0]["bucket"] == "A_no_detection"


def test_bucket_b_when_a_detection_overlaps_but_misses_the_iou_floor() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    # Overlaps substantially but well under IoU 0.50.
    record = _frame([0, 0, 100, 100], [{"box": [70, 70, 200, 200], "score": 0.9}], [])
    entry = classify_frame(record, 0.5)[0]
    assert entry["bucket"] == "B_detected_unmatched"
    assert 0.0 < entry["best_iou"] < 0.50


def test_bucket_c_when_matched_but_no_rider_associates() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    record = _frame([0, 0, 100, 100], [{"box": [0, 0, 100, 100], "score": 0.9}], [])
    assert classify_frame(record, 0.5)[0]["bucket"] == "C_association_failed"


def test_bucket_c_when_two_riders_associate_to_a_single_rider_ground_truth() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    record = _frame(
        [0, 0, 100, 100],
        [{"box": [0, 0, 100, 100], "score": 0.9}],
        [{"box": [0, 0, 50, 50], "score": 0.9}, {"box": [50, 50, 100, 100], "score": 0.9}],
    )
    entry = classify_frame(record, 0.5)[0]
    assert entry["bucket"] == "C_association_failed"
    assert entry["n_riders"] == 2


def test_bucket_d_when_matched_with_exactly_one_rider() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    record = _frame(
        [0, 0, 100, 100],
        [{"box": [0, 0, 100, 100], "score": 0.9}],
        [{"box": [10, 0, 90, 60], "score": 0.9}],
    )
    assert classify_frame(record, 0.5)[0]["bucket"] == "D_recovered"


def test_a_sub_threshold_detection_is_recorded_even_when_it_does_not_count() -> None:
    """This is what separates 'operating point' from 'cannot see it'."""

    from helmet_runtime_validation.recovery import classify_frame

    record = _frame([0, 0, 100, 100], [{"box": [0, 0, 100, 100], "score": 0.2}], [])
    entry = classify_frame(record, 0.5)[0]
    assert entry["bucket"] == "A_no_detection"  # invisible at this threshold
    assert entry["best_iou_any_score"] == pytest.approx(1.0)  # but it existed
    assert entry["score_of_best_any"] == pytest.approx(0.2)


def test_lowering_the_threshold_can_move_a_crop_from_a_to_d() -> None:
    from helmet_runtime_validation.recovery import classify_frame

    record = _frame(
        [0, 0, 100, 100],
        [{"box": [0, 0, 100, 100], "score": 0.2}],
        [{"box": [10, 0, 90, 60], "score": 0.2}],
    )
    assert classify_frame(record, 0.5)[0]["bucket"] == "A_no_detection"
    assert classify_frame(record, 0.15)[0]["bucket"] == "D_recovered"


def test_detection_precision_counts_unannotated_detections_as_false_positives() -> None:
    from helmet_runtime_validation.recovery import detection_prf

    record = _frame(
        [0, 0, 100, 100],
        [{"box": [0, 0, 100, 100], "score": 0.9}, {"box": [500, 500, 600, 600], "score": 0.9}],
        [],
    )
    tp, fp, fn = detection_prf(record, 0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_detection_recall_uses_the_full_frame_annotation_not_the_eligible_subset() -> None:
    """A multi-rider motorcycle is a real motorcycle: missing it is a detector error."""

    from helmet_runtime_validation.recovery import detection_prf

    record = _frame(
        [0, 0, 100, 100],
        [{"box": [0, 0, 100, 100], "score": 0.9}],
        [],
        gt_all=[
            {"track_id": "t", "box": [0, 0, 100, 100]},
            {"track_id": "other", "box": [300, 300, 400, 400]},
        ],
    )
    tp, fp, fn = detection_prf(record, 0.5)
    assert (tp, fp, fn) == (1, 0, 1)


def test_the_sweep_grid_is_the_pre_committed_one() -> None:
    from helmet_runtime_validation.recovery import BASELINE_THRESHOLD, THRESHOLD_GRID

    assert THRESHOLD_GRID == (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
    assert BASELINE_THRESHOLD == 0.50


def test_selection_breaks_ties_to_the_more_conservative_threshold() -> None:
    from helmet_runtime_validation.recovery import select

    rows = [
        {"threshold": 0.50, "detection_f1": 0.80},
        {"threshold": 0.40, "detection_f1": 0.80},
        {"threshold": 0.30, "detection_f1": 0.79},
    ]
    assert select(rows)["selected_threshold"] == 0.50


def test_the_association_overlap_matches_the_production_policy() -> None:
    """The sweep restates the policy as geometry; it must not drift from the real one."""

    from helmet_runtime_validation import recovery

    from trafficpulse.association.riders import DEFAULT_MIN_OVERLAP, RiderAssociationConfig

    assert RiderAssociationConfig().min_overlap == recovery.DEFAULT_MIN_OVERLAP
    assert recovery.DEFAULT_MIN_OVERLAP == DEFAULT_MIN_OVERLAP

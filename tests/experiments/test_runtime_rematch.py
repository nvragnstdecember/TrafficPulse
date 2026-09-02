"""P4-U8 matching-convention logic: the rules that decide what a recovery number means.

These cover the pure functions whose bugs would silently change the corrected recovery
rate -- the union proxy, the bucket taxonomy, the head-crop gate replication, and the
stratification helpers. They run on fixtures, so the suite needs neither the HELMET corpus
nor a model.

The head-gate test is the important one: ``head_crop_outcome`` reproduces
``extract_head_region`` without pixels, and a divergence there would move crops between
"recovered" and "gated" invisibly. It is therefore asserted against the production function
on real arrays rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.rematch import (  # noqa: E402
    CONVENTIONS,
    GT_MATCH_IOU,
    REASON_MANY_RIDERS,
    REASON_NO_MATCH,
    REASON_NO_RIDER,
    REASON_RECOVERED,
    REASON_TOO_SHORT,
    SUSPICIOUS_MOTORCYCLE_IOU,
    build_proxies,
    head_crop_outcome,
    match_frame,
    riders_by_motorcycle,
    summarise,
    track_states,
    union_box,
)

from trafficpulse.contracts.primitives import BoundingBox  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    extract_head_region,
)

FRAME_W, FRAME_H = 1920, 1080


def frame(
    *,
    motorcycles: list[tuple[list[float], float]],
    persons: list[tuple[list[float], float]],
    eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    """A dump record in the shape ``detection_dump.jsonl`` writes."""

    return {
        "video_id": "clip",
        "frame_index": 1,
        "frame_w": FRAME_W,
        "frame_h": FRAME_H,
        "gt_all": [],
        "eligible": eligible,
        "motorcycles": [{"box": box, "score": score} for box, score in motorcycles],
        "persons": [{"box": box, "score": score} for box, score in persons],
    }


def gt(box: list[float], *, crop_id: str = "crop-1", label: str = "helmet") -> dict[str, Any]:
    return {
        "crop_id": crop_id,
        "split": "val",
        "track_id": "t1",
        "label": label,
        "site_id": "site",
        "box": box,
    }


# --- the union proxy ---------------------------------------------------------
def test_union_of_one_box_is_that_box() -> None:
    assert union_box([(1.0, 2.0, 3.0, 4.0)]) == (1.0, 2.0, 3.0, 4.0)


def test_union_spans_every_input_box() -> None:
    assert union_box([(10.0, 10.0, 20.0, 20.0), (5.0, 30.0, 15.0, 40.0)]) == (
        5.0,
        10.0,
        20.0,
        40.0,
    )


def test_union_of_nothing_raises_rather_than_inventing_a_box() -> None:
    with pytest.raises(ValueError, match="at least one box"):
        union_box([])


def test_rider_inclusive_proxy_grows_upward_to_cover_the_rider() -> None:
    """The whole correction: the vehicle box excludes the rider's head, the proxy does not."""

    states = track_states("clip", 1, [(100.0, 200.0, 200.0, 300.0)], [(120.0, 150.0, 180.0, 260.0)])
    (proxy,) = build_proxies(states, convention="rider_inclusive")
    assert proxy.motorcycle_box == (100.0, 200.0, 200.0, 300.0)
    assert proxy.proxy_box == (100.0, 150.0, 200.0, 300.0)
    assert proxy.rider_count == 1


def test_motorcycle_only_proxy_is_the_vehicle_box_even_when_a_rider_associates() -> None:
    states = track_states("clip", 1, [(100.0, 200.0, 200.0, 300.0)], [(120.0, 150.0, 180.0, 260.0)])
    (proxy,) = build_proxies(states, convention="motorcycle_only")
    assert proxy.proxy_box == proxy.motorcycle_box
    assert proxy.rider_count == 1, "the rider list is convention-independent"


def test_an_unridden_motorcycle_has_the_vehicle_box_under_both_conventions() -> None:
    states = track_states("clip", 1, [(100.0, 200.0, 200.0, 300.0)], [])
    for convention in CONVENTIONS:
        (proxy,) = build_proxies(states, convention=convention)
        assert proxy.proxy_box == proxy.motorcycle_box
        assert proxy.rider_count == 0


def test_an_unknown_convention_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown convention"):
        build_proxies(track_states("clip", 1, [], []), convention="something_else")  # type: ignore[arg-type]


# --- association is the production policy, not a copy of it ------------------
def test_association_uses_the_production_rule_one_motorcycle_per_rider() -> None:
    """A person overlapping two motorcycles is assigned to its best one only."""

    states = track_states(
        "clip",
        1,
        [(0.0, 0.0, 100.0, 100.0), (50.0, 0.0, 250.0, 100.0)],
        [(60.0, 10.0, 90.0, 40.0)],
    )
    riders = riders_by_motorcycle(states)
    assert sum(len(v) for v in riders.values()) == 1, "the rider is linked once, not twice"


def test_a_motorcycle_may_carry_several_riders() -> None:
    states = track_states(
        "clip",
        1,
        [(0.0, 0.0, 200.0, 200.0)],
        [(10.0, 10.0, 60.0, 60.0), (100.0, 100.0, 150.0, 150.0)],
    )
    riders = riders_by_motorcycle(states)
    assert len(riders["mot-00000"]) == 2


# --- head-crop gates: replicated without pixels, asserted against production ---
@pytest.mark.parametrize(
    "rider_box",
    [
        (100.0, 100.0, 200.0, 400.0),  # comfortably inside the frame
        (100.0, 100.0, 200.0, 130.0),  # 30px tall -> 9px head region, below the floor
        (0.0, 0.0, 40.0, 200.0),  # flush against the top-left frame edge
        (1900.0, 1000.0, 2100.0, 1200.0),  # partly off the bottom-right corner
        (1950.0, 1100.0, 2100.0, 1300.0),  # entirely past the frame edge
    ],
)
def test_the_pixel_free_head_gate_agrees_with_the_production_extractor(
    rider_box: tuple[float, float, float, float],
) -> None:
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    region = extract_head_region(
        BoundingBox(x1=rider_box[0], y1=rider_box[1], x2=rider_box[2], y2=rider_box[3]),
        image,
        head_fraction=DEFAULT_HEAD_FRACTION,
    )
    production_reason = None
    if region.image is None:
        production_reason = "head_region_off_frame"
    elif region.height_px < DEFAULT_MIN_CROP_HEIGHT_PX:
        production_reason = "head_crop_below_min_height_gate"

    reason, height = head_crop_outcome(rider_box, FRAME_W, FRAME_H)
    assert reason == production_reason
    assert height == pytest.approx(region.height_px)


# --- the bucket taxonomy -----------------------------------------------------
def test_bucket_a_when_nothing_overlaps_the_annotation() -> None:
    record = frame(
        motorcycles=[([800.0, 800.0, 900.0, 900.0], 0.9)],
        persons=[],
        eligible=[gt([100.0, 100.0, 200.0, 200.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.bucket == "A_no_detection"
    assert outcome.reason == REASON_NO_MATCH


def test_bucket_b_when_something_overlaps_but_misses_the_floor() -> None:
    record = frame(
        motorcycles=[([180.0, 180.0, 260.0, 260.0], 0.9)],
        persons=[],
        eligible=[gt([100.0, 100.0, 200.0, 200.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.bucket == "B_detected_unmatched"
    assert 0.0 < outcome.best_iou < GT_MATCH_IOU


def test_bucket_c_when_the_match_carries_no_rider() -> None:
    record = frame(
        motorcycles=[([100.0, 100.0, 200.0, 200.0], 0.9)],
        persons=[],
        eligible=[gt([100.0, 100.0, 200.0, 200.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.bucket == "C_association_failed"
    assert outcome.reason == REASON_NO_RIDER


def test_bucket_c_when_two_riders_associate_to_a_single_rider_annotation() -> None:
    record = frame(
        motorcycles=[([100.0, 100.0, 300.0, 300.0], 0.9)],
        persons=[([110.0, 60.0, 180.0, 280.0], 0.9), ([200.0, 60.0, 280.0, 280.0], 0.9)],
        eligible=[gt([100.0, 60.0, 300.0, 300.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.bucket == "C_association_failed"
    assert outcome.reason == REASON_MANY_RIDERS


def test_bucket_d_when_one_rider_associates_and_the_crop_clears_the_gates() -> None:
    record = frame(
        motorcycles=[([100.0, 200.0, 200.0, 300.0], 0.9)],
        persons=[([110.0, 100.0, 190.0, 260.0], 0.9)],
        eligible=[gt([100.0, 100.0, 200.0, 300.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.reason == REASON_RECOVERED
    assert outcome.bucket == "D_recovered"
    assert outcome.head_height_px == pytest.approx(48.0)


def test_a_tiny_rider_is_gated_not_recovered() -> None:
    record = frame(
        motorcycles=[([100.0, 118.0, 130.0, 150.0], 0.9)],
        persons=[([104.0, 100.0, 126.0, 130.0], 0.9)],
        eligible=[gt([100.0, 100.0, 130.0, 150.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.reason == REASON_TOO_SHORT
    assert outcome.bucket == "D_gated_too_short"


def test_the_correction_can_move_a_crop_from_b_to_d() -> None:
    """The single fact P4-U8 exists to measure, in one fixture."""

    record = frame(
        motorcycles=[([100.0, 260.0, 200.0, 400.0], 0.9)],
        persons=[([110.0, 100.0, 190.0, 320.0], 0.9)],
        eligible=[gt([100.0, 100.0, 200.0, 400.0])],
    )
    (before,) = match_frame(record, convention="motorcycle_only", split="val").outcomes
    (after,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert before.bucket == "B_detected_unmatched"
    assert after.bucket == "D_recovered"
    assert after.best_iou > before.best_iou


# --- discipline: detections below the operating point never participate -------
def test_sub_threshold_detections_are_invisible_to_the_matcher() -> None:
    record = frame(
        motorcycles=[([100.0, 100.0, 200.0, 200.0], 0.49)],
        persons=[([110.0, 100.0, 190.0, 160.0], 0.9)],
        eligible=[gt([100.0, 100.0, 200.0, 200.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.bucket == "A_no_detection", "threshold 0.50 is unchanged by this unit"


def test_a_proxy_is_used_by_at_most_one_annotation() -> None:
    box = [100.0, 100.0, 200.0, 200.0]
    record = frame(
        motorcycles=[(box, 0.9)],
        persons=[([110.0, 90.0, 190.0, 160.0], 0.9)],
        eligible=[gt(box, crop_id="crop-a"), gt(box, crop_id="crop-b")],
    )
    result = match_frame(record, convention="rider_inclusive", split="val")
    reasons = sorted(o.reason for o in result.outcomes)
    assert reasons == [REASON_NO_MATCH, REASON_RECOVERED]
    assert result.contested_proxies == 1, "order-dependence is counted, not hidden"


def test_only_the_requested_split_is_matched() -> None:
    record = frame(
        motorcycles=[([100.0, 200.0, 200.0, 300.0], 0.9)],
        persons=[([110.0, 100.0, 190.0, 260.0], 0.9)],
        eligible=[gt([100.0, 100.0, 200.0, 300.0])],
    )
    record["eligible"][0]["split"] = "test"
    assert match_frame(record, convention="rider_inclusive", split="val").outcomes == []
    assert len(match_frame(record, convention="rider_inclusive", split="test").outcomes) == 1


# --- the suspicious-match guard ---------------------------------------------
def test_a_match_carried_entirely_by_the_rider_union_is_flagged() -> None:
    """A vehicle box barely touching the annotation, rescued by a large rider box."""

    record = frame(
        motorcycles=[([100.0, 380.0, 200.0, 400.0], 0.9)],
        persons=[([100.0, 100.0, 200.0, 390.0], 0.9)],
        eligible=[gt([100.0, 100.0, 200.0, 400.0])],
    )
    (outcome,) = match_frame(record, convention="rider_inclusive", split="val").outcomes
    assert outcome.reason == REASON_RECOVERED
    assert outcome.motorcycle_only_iou < SUSPICIOUS_MOTORCYCLE_IOU
    assert outcome.suspicious_match is True


# --- reporting ---------------------------------------------------------------
def test_an_empty_stratum_reports_none_not_a_fabricated_zero() -> None:
    summary = summarise([], contested=0)
    assert summary["recovery_rate"] is None
    assert summary["head_height_px_recovered"] is None


def test_the_summary_separates_detection_failure_from_association_failure() -> None:
    records = [
        frame(  # A: nothing overlapping
            motorcycles=[([800.0, 800.0, 900.0, 900.0], 0.9)],
            persons=[],
            eligible=[gt([100.0, 100.0, 200.0, 200.0], crop_id="crop-a")],
        ),
        frame(  # C: matched, no rider
            motorcycles=[([100.0, 100.0, 200.0, 200.0], 0.9)],
            persons=[],
            eligible=[gt([100.0, 100.0, 200.0, 200.0], crop_id="crop-c")],
        ),
    ]
    outcomes = [
        o
        for record in records
        for o in match_frame(record, convention="rider_inclusive", split="val").outcomes
    ]
    summary = summarise(outcomes, contested=0)
    assert summary["detection_failure_rate_of_eligible"] == pytest.approx(0.5)
    assert summary["association_failure_rate_of_eligible"] == pytest.approx(0.5)
    assert summary["association_failure_rate_of_matched"] == pytest.approx(1.0)
    assert summary["recovered"] == 0

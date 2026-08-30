"""Temperature scaling and robustness slicing (P4-U5).

Calibration is load-bearing rather than cosmetic here: the runtime's no-helmet rule
aggregates per-frame confidences over a track, so a systematically over-confident
classifier moves the temporal aggregator's decision point. These tests pin the
direction of the fit (an over-confident model must be softened, not sharpened) and
the discipline that the temperature is fitted on validation alone.
"""

from __future__ import annotations

import math

import pytest
from helmet_cnn_vit.calibrate import apply_temperature, fit_temperature, to_logits
from helmet_cnn_vit.errors import EmptyEvaluationError, MismatchedPredictionsError
from helmet_cnn_vit.metrics import CLASS_ORDER, expected_calibration_error
from helmet_cnn_vit.robustness import (
    CORRUPTIONS,
    SEVERITIES,
    SPEC_HEIGHT_EDGES,
    TRAIN_HEIGHT_TERTILES,
    bucket_by_height,
    bucket_by_key,
    corruption_variants,
    height_bucket,
)

HELMET, NO_HELMET = CLASS_ORDER


# --- logit round trip --------------------------------------------------------------


def test_probabilities_round_trip_through_logits() -> None:
    scores = [0.1, 0.5, 0.9]
    restored = apply_temperature(scores, 1.0)
    assert restored == pytest.approx(scores, abs=1e-6)


def test_saturated_probabilities_do_not_produce_infinite_logits() -> None:
    """fp16 softmax genuinely returns exactly 0.0 and 1.0; an inf would poison the fit."""

    logits = to_logits([0.0, 1.0])
    assert all(math.isfinite(v) for v in logits)


def test_a_non_positive_temperature_is_refused() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(MismatchedPredictionsError):
            apply_temperature([0.5], bad)


# --- the direction of the fit --------------------------------------------------------


def test_an_over_confident_model_is_softened() -> None:
    """Confident-but-often-wrong predictions must be pushed toward 0.5 (T > 1)."""

    # 100 crops at 0.95 confidence for no_helmet, but only 60% actually are.
    truth = [NO_HELMET] * 60 + [HELMET] * 40
    scores = [0.95] * 100
    fit = fit_temperature(truth, scores)
    assert fit.temperature > 1.0
    assert fit.improved

    softened = apply_temperature(scores, fit.temperature)
    assert all(s < 0.95 for s in softened)


def test_an_under_confident_model_is_sharpened() -> None:
    """Hedged-but-correct predictions must be pushed away from 0.5 (T < 1)."""

    truth = [NO_HELMET] * 98 + [HELMET] * 2
    scores = [0.55] * 100
    fit = fit_temperature(truth, scores)
    assert fit.temperature < 1.0
    assert apply_temperature(scores, fit.temperature)[0] > 0.55


def test_calibration_reduces_the_expected_calibration_error() -> None:
    """The end-to-end point of the exercise, measured the way §12 asks."""

    truth = [NO_HELMET] * 60 + [HELMET] * 40
    scores = [0.95] * 100

    before, _ = expected_calibration_error(scores, [t == NO_HELMET for t in truth])
    fit = fit_temperature(truth, scores)
    softened = apply_temperature(scores, fit.temperature)
    after, _ = expected_calibration_error(softened, [t == NO_HELMET for t in truth])

    assert before is not None and after is not None
    assert after < before


def test_the_fit_is_deterministic() -> None:
    truth = [NO_HELMET] * 30 + [HELMET] * 70
    scores = [0.8] * 30 + [0.4] * 70
    assert fit_temperature(truth, scores) == fit_temperature(truth, scores)


def test_calibration_refuses_misaligned_or_empty_input() -> None:
    with pytest.raises(MismatchedPredictionsError):
        fit_temperature([HELMET], [0.1, 0.2])
    with pytest.raises(EmptyEvaluationError):
        fit_temperature([], [])


# --- robustness slicing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "bucket"),
    [(10.0, "<32"), (32.0, "32-64"), (64.0, "32-64"), (65.0, ">64"), (500.0, ">64")],
)
def test_the_spec_height_buckets_use_inclusive_edges(height: float, bucket: str) -> None:
    assert height_bucket(height, SPEC_HEIGHT_EDGES) == bucket


def test_the_train_derived_tertiles_split_this_corpus_usefully() -> None:
    """§12's absolute buckets are degenerate here; the tertiles are the usable slice."""

    assert TRAIN_HEIGHT_TERTILES == (170.0, 287.0)
    heights = [100.0, 200.0, 400.0]
    groups = bucket_by_height(heights, edges=TRAIN_HEIGHT_TERTILES)
    assert sorted(groups) == ["170-287", "<170", ">287"]
    assert all(len(v) == 1 for v in groups.values())


def test_height_bucketing_preserves_index_order() -> None:
    groups = bucket_by_height([500.0, 10.0, 600.0], edges=SPEC_HEIGHT_EDGES)
    assert groups[">64"] == [0, 2]
    assert groups["<32"] == [1]


def test_site_bucketing_groups_by_key_in_sorted_order() -> None:
    groups = bucket_by_key(["Mandalay_1", "Bago_rural", "Mandalay_1"])
    assert list(groups) == ["Bago_rural", "Mandalay_1"]
    assert groups["Mandalay_1"] == [0, 2]


def test_the_corruption_grid_is_four_kinds_at_three_severities() -> None:
    variants = corruption_variants()
    assert len(variants) == len(CORRUPTIONS) * len(SEVERITIES) == 12
    assert set(CORRUPTIONS) == {
        "brightness",
        "gaussian_blur",
        "jpeg_compression",
        "motion_blur",
    }
    # Deterministic order, so a report's rows are stable across runs.
    assert variants == tuple(sorted(variants))

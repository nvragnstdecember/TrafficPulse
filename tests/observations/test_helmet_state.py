"""Head-region geometry, quality gating, and helmet-observation stamping (P4-U4).

Pure derivation: no classifier, no pixels beyond synthetic arrays, no ML.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from trafficpulse.classifier import RawHelmetPrediction
from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import (
    HelmetState,
    ObjectClass,
    ProducerKind,
    RiderSlot,
    TrackStatus,
)
from trafficpulse.observations.helmet import (
    HeadCropConfig,
    HeadRegion,
    HelmetObservationConfig,
    build_observation,
    extract_head_region,
    gate_crop,
    head_region_box,
    laplacian_variance,
    rider_slot,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)


def rider(box: tuple[float, float, float, float] = (10, 10, 50, 110)) -> TrackState:
    x1, y1, x2, y2 = box
    return TrackState(
        track_id="p1",
        camera_id="cam-1",
        timestamp=BASE,
        frame_index=0,
        object_class=ObjectClass.PERSON,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        status=TrackStatus.ACTIVE,
    )


def sharp_image(height: int = 200, width: int = 200) -> np.ndarray:
    """High-frequency checkerboard: large Laplacian variance."""

    ys, xs = np.mgrid[0:height, 0:width]
    pattern = (((ys // 2) + (xs // 2)) % 2 * 255).astype(np.uint8)
    return np.stack([pattern] * 3, axis=-1)


def flat_image(height: int = 200, width: int = 200) -> np.ndarray:
    """Uniform image: zero Laplacian variance (maximally 'blurred')."""

    return np.full((height, width, 3), 128, dtype=np.uint8)


# --- head-region geometry ----------------------------------------------------
def test_head_region_is_the_top_fraction_at_full_width() -> None:
    box = head_region_box(BoundingBox(x1=10, y1=10, x2=50, y2=110), head_fraction=0.30)

    assert box is not None
    assert (box.x1, box.x2) == (10, 50)  # full width: containment over tightness
    assert box.y1 == 10
    assert box.y2 == pytest.approx(40.0)  # 10 + 0.30 * 100


def test_head_fraction_is_configurable() -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=40, y2=100)

    assert head_region_box(bbox, head_fraction=0.5) is not None
    assert head_region_box(bbox, head_fraction=0.5).y2 == pytest.approx(50.0)  # type: ignore[union-attr]
    assert head_region_box(bbox, head_fraction=1.0).y2 == pytest.approx(100.0)  # type: ignore[union-attr]


def test_region_is_clipped_to_the_frame() -> None:
    """A detector box may extend past the frame edge; the slice must not wrap."""

    bbox = BoundingBox(x1=180, y1=0, x2=260, y2=100)  # extends past a 200px-wide frame
    region = extract_head_region(bbox, sharp_image(200, 200), head_fraction=0.5)

    assert region.image is not None
    assert region.image.shape[1] == 20  # 180..200, not 80


def test_region_entirely_off_frame_has_no_crop() -> None:
    bbox = BoundingBox(x1=500, y1=500, x2=600, y2=600)
    region = extract_head_region(bbox, sharp_image(200, 200), head_fraction=0.5)

    assert region.image is None
    assert region.box is not None  # geometry still known


def test_missing_image_yields_no_crop_but_keeps_geometry() -> None:
    region = extract_head_region(BoundingBox(x1=0, y1=0, x2=40, y2=100), None, head_fraction=0.3)

    assert region.image is None
    assert region.height_px == pytest.approx(30.0)


def test_crop_is_a_view_and_does_not_copy_or_mutate_the_frame() -> None:
    image = sharp_image()
    before = image.copy()
    region = extract_head_region(BoundingBox(x1=0, y1=0, x2=40, y2=100), image, head_fraction=0.3)

    assert region.image is not None
    assert np.array_equal(image, before)
    assert region.image.base is image  # a view, not a copy


def test_height_px_is_reported_in_frame_pixel_space() -> None:
    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=40, y2=100), sharp_image(), head_fraction=0.30
    )
    assert region.height_px == pytest.approx(30.0)


# --- blur measure ------------------------------------------------------------
def test_flat_image_has_zero_laplacian_variance() -> None:
    assert laplacian_variance(flat_image(50, 50)) == pytest.approx(0.0)


def test_sharp_image_has_high_laplacian_variance() -> None:
    assert laplacian_variance(sharp_image(50, 50)) > 1000.0


def test_tiny_crop_returns_zero_variance_without_raising() -> None:
    assert laplacian_variance(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0


# --- quality gate ------------------------------------------------------------
def test_usable_crop_passes_the_gate() -> None:
    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=40, y2=100), sharp_image(), head_fraction=0.30
    )
    assert gate_crop(region, config=HeadCropConfig()) is None


def test_missing_pixels_are_gated() -> None:
    region = HeadRegion(box=BoundingBox(x1=0, y1=0, x2=10, y2=10), height_px=10.0, image=None)
    reason = gate_crop(region, config=HeadCropConfig())

    assert reason is not None and "no pixels" in reason


def test_degenerate_region_is_gated() -> None:
    region = HeadRegion(box=None, height_px=0.0, image=None)
    reason = gate_crop(region, config=HeadCropConfig())

    assert reason is not None and "degenerate" in reason


def test_crop_below_the_height_floor_is_gated() -> None:
    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=40, y2=20), sharp_image(), head_fraction=0.30
    )  # 6px head region

    reason = gate_crop(region, config=HeadCropConfig(min_crop_height_px=12.0))
    assert reason is not None and "below" in reason


def test_blur_gate_is_disabled_by_default() -> None:
    """Unset by design: no held-out data exists to calibrate a blur threshold."""

    assert HeadCropConfig().min_blur_variance is None

    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=100, y2=200), flat_image(), head_fraction=0.30
    )
    assert gate_crop(region, config=HeadCropConfig()) is None  # flat, yet not gated


def test_blur_gate_fires_when_explicitly_enabled() -> None:
    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=100, y2=200), flat_image(), head_fraction=0.30
    )

    reason = gate_crop(region, config=HeadCropConfig(min_blur_variance=10.0))
    assert reason is not None and "blurred" in reason


def test_blur_gate_passes_a_sharp_crop() -> None:
    region = extract_head_region(
        BoundingBox(x1=0, y1=0, x2=100, y2=200), sharp_image(), head_fraction=0.30
    )
    assert gate_crop(region, config=HeadCropConfig(min_blur_variance=10.0)) is None


# --- rider slots -------------------------------------------------------------
def test_a_lone_rider_is_the_driver() -> None:
    assert rider_slot(1) is RiderSlot.DRIVER


def test_multiple_riders_are_unknown_not_guessed() -> None:
    """Without the bike's travel direction, driver-vs-pillion is undeterminable."""

    assert rider_slot(2) is RiderSlot.UNKNOWN
    assert rider_slot(3) is RiderSlot.UNKNOWN


def test_zero_riders_is_unknown() -> None:
    assert rider_slot(0) is RiderSlot.UNKNOWN


# --- observation stamping ----------------------------------------------------
def _region(height: float = 30.0) -> HeadRegion:
    return HeadRegion(
        box=BoundingBox(x1=0, y1=0, x2=40, y2=max(height, 0.1)),
        height_px=height,
        image=np.zeros((int(height) or 1, 40, 3), dtype=np.uint8),
    )


def _observe(
    prediction: RawHelmetPrediction | None,
    *,
    gate_reason: str | None = None,
    config: HelmetObservationConfig | None = None,
    rider_count: int = 1,
):
    return build_observation(
        rider(),
        region=_region(),
        prediction=prediction,
        gate_reason=gate_reason,
        rider_count=rider_count,
        config=config or HelmetObservationConfig(),
    )


def test_observation_carries_the_rider_not_the_motorcycle() -> None:
    """track_id names the RIDER; the bike link lives in the Association contract."""

    assert _observe(RawHelmetPrediction("no_helmet", 0.8)).track_id == "p1"


def test_native_label_is_mapped_to_the_frozen_ontology() -> None:
    assert _observe(RawHelmetPrediction("no_helmet", 0.8)).helmet_state is HelmetState.NO_HELMET
    assert _observe(RawHelmetPrediction("helmet", 0.8)).helmet_state is HelmetState.HELMET
    assert _observe(RawHelmetPrediction("turban", 0.8)).helmet_state is HelmetState.TURBAN


def test_real_classifier_score_travels_as_confidence() -> None:
    assert _observe(RawHelmetPrediction("no_helmet", 0.77)).confidence == pytest.approx(0.77)


def test_crop_height_travels_with_the_observation() -> None:
    assert _observe(RawHelmetPrediction("helmet", 0.9)).crop_height_px == pytest.approx(30.0)


def test_provenance_is_stamped() -> None:
    producer = _observe(RawHelmetPrediction("helmet", 0.9)).producer

    assert producer.kind is ProducerKind.MODEL
    assert producer.name and producer.version


def test_gated_crop_abstains_with_no_fabricated_confidence() -> None:
    """The classifier never ran, so confidence is None -- never a fabricated 0.0."""

    observation = _observe(None, gate_reason="head crop 3.0px is below the 12.0px floor")

    assert observation.helmet_state is HelmetState.UNCERTAIN
    assert observation.confidence is None


def test_gated_crop_still_reports_its_measured_height() -> None:
    observation = _observe(None, gate_reason="too small")
    assert observation.crop_height_px == pytest.approx(30.0)


def test_low_score_abstains_but_keeps_the_real_score() -> None:
    """Uncertainty must not discard the measurement that produced it."""

    observation = _observe(
        RawHelmetPrediction("no_helmet", 0.4),
        config=HelmetObservationConfig(min_confidence=0.6),
    )

    assert observation.helmet_state is HelmetState.UNCERTAIN
    assert observation.confidence == pytest.approx(0.4)


def test_score_at_the_floor_is_not_abstained() -> None:
    observation = _observe(
        RawHelmetPrediction("no_helmet", 0.6),
        config=HelmetObservationConfig(min_confidence=0.6),
    )
    assert observation.helmet_state is HelmetState.NO_HELMET


def test_unmapped_native_label_abstains_rather_than_guessing() -> None:
    """The P4-U1 'motorbike' lesson: an unknown vocabulary must never be guessed."""

    observation = _observe(RawHelmetPrediction("bareheaded", 0.99))

    assert observation.helmet_state is HelmetState.UNCERTAIN
    assert observation.confidence == pytest.approx(0.99)


def test_uncertain_is_never_upgraded_to_certainty() -> None:
    observation = _observe(RawHelmetPrediction("uncertain", 0.99))
    assert observation.helmet_state is HelmetState.UNCERTAIN


def test_out_of_range_score_is_clamped_into_the_contract_bound() -> None:
    observation = _observe(RawHelmetPrediction("helmet", 1.0000001))
    assert observation.confidence is not None and observation.confidence <= 1.0


def test_slot_reflects_the_rider_count() -> None:
    assert _observe(RawHelmetPrediction("helmet", 0.9), rider_count=1).rider_slot is (
        RiderSlot.DRIVER
    )
    assert _observe(RawHelmetPrediction("helmet", 0.9), rider_count=2).rider_slot is (
        RiderSlot.UNKNOWN
    )


def test_observation_ids_are_deterministic_and_content_derived() -> None:
    first = _observe(RawHelmetPrediction("helmet", 0.9))
    second = _observe(RawHelmetPrediction("helmet", 0.9))

    assert first.observation_id == second.observation_id


def test_observation_is_a_frozen_contract() -> None:
    observation = _observe(RawHelmetPrediction("helmet", 0.9))
    with pytest.raises(ValidationError):
        observation.helmet_state = HelmetState.NO_HELMET  # type: ignore[misc]


def test_config_is_frozen_and_strict() -> None:
    with pytest.raises(ValidationError):
        HelmetObservationConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        HeadCropConfig(head_fraction=0.0)  # must be > 0
    with pytest.raises(ValidationError):
        HeadCropConfig(head_fraction=1.5)


def test_label_map_is_configurable_per_backend() -> None:
    config = HelmetObservationConfig(label_map={"BARE": HelmetState.NO_HELMET})
    observation = _observe(RawHelmetPrediction("BARE", 0.8), config=config)

    assert observation.helmet_state is HelmetState.NO_HELMET

"""The trained P4-U5 ResNet-50 behind the P4-U2 seam (P4-U6).

Every test here runs through the injected :class:`ResNetInferenceEngine`, so the
suite touches no ML framework -- the same posture as the zero-shot backend's tests.
The one test that loads the *real* checkpoint is opt-in and skipped by default.

The load-bearing invariant is the **class order**. ``NATIVE_LABELS`` must stay
index-aligned with the trained head (``helmet`` -> 0, ``no_helmet`` -> 1, fixed by
``helmet_cnn_vit.datasets.CLASS_INDEX``). Getting it backwards would invert every
prediction while the code still "worked", so it is asserted rather than assumed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from trafficpulse.classifier import (
    Crop,
    MalformedResNetOutputError,
    ResNetHelmetClassifier,
    ResNetHelmetConfig,
    ResNetMissingCropImageError,
)
from trafficpulse.classifier.resnet import (
    ABSTAIN_LABEL,
    IMAGE_SIZE,
    NATIVE_LABELS,
    PAD_RGB,
    square_pad_resize,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeEngine:
    """Returns canned probability rows; records what it was asked to score."""

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._rows = rows
        self.seen: list[NDArray[np.uint8]] = []

    def infer(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[Sequence[float]]:
        self.seen = list(images)
        return self._rows


def crop(track_id: str = "r", frame_index: int = 0, *, with_pixels: bool = True) -> Crop:
    return Crop(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=BASE,
        track_id=track_id,
        image=np.zeros((8, 8, 3), dtype=np.uint8) if with_pixels else None,
    )


def _config(**overrides: object) -> ResNetHelmetConfig:
    return ResNetHelmetConfig(checkpoint=Path("unused.pt"), **overrides)  # type: ignore[arg-type]


def _classifier(
    rows: Sequence[Sequence[float]], **overrides: object
) -> tuple[ResNetHelmetClassifier, _FakeEngine]:
    engine = _FakeEngine(rows)
    return ResNetHelmetClassifier(_config(**overrides), engine=engine), engine


# --- class order -------------------------------------------------------------
def test_native_label_order_matches_the_frozen_training_index() -> None:
    """The head was trained with helmet=0, no_helmet=1. Inverting this inverts everything."""

    assert NATIVE_LABELS == ("helmet", "no_helmet")


def test_index_zero_is_helmet_and_index_one_is_no_helmet() -> None:
    classifier, _ = _classifier([[0.9, 0.1], [0.2, 0.8]])
    predictions = classifier.classify([crop("a"), crop("b")])
    assert [p.label for p in predictions] == ["helmet", "no_helmet"]
    assert predictions[0].score == pytest.approx(0.9)
    assert predictions[1].score == pytest.approx(0.8)


# --- seam contract -----------------------------------------------------------
def test_empty_in_empty_out_never_touches_the_model() -> None:
    classifier, engine = _classifier([[1.0, 0.0]])
    assert classifier.classify([]) == ()
    assert engine.seen == []


def test_predictions_are_returned_in_input_order() -> None:
    classifier, _ = _classifier([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
    labels = [p.label for p in classifier.classify([crop("a"), crop("b"), crop("c")])]
    assert labels == ["helmet", "no_helmet", "helmet"]


def test_a_crop_without_pixels_is_a_typed_error() -> None:
    classifier, _ = _classifier([[1.0, 0.0]])
    with pytest.raises(ResNetMissingCropImageError):
        classifier.classify([crop(with_pixels=False)])


def test_a_row_count_mismatch_is_a_typed_error() -> None:
    classifier, _ = _classifier([[1.0, 0.0]])
    with pytest.raises(MalformedResNetOutputError):
        classifier.classify([crop("a"), crop("b")])


def test_a_wrong_width_row_is_a_typed_error() -> None:
    classifier, _ = _classifier([[0.3, 0.3, 0.4]])
    with pytest.raises(MalformedResNetOutputError):
        classifier.classify([crop()])


def test_a_non_finite_score_is_a_typed_error() -> None:
    classifier, _ = _classifier([[float("nan"), 0.0]])
    with pytest.raises(MalformedResNetOutputError):
        classifier.classify([crop()])


def test_an_exact_tie_resolves_deterministically_to_the_earlier_label() -> None:
    classifier, _ = _classifier([[0.5, 0.5]])
    assert classifier.classify([crop()])[0].label == "helmet"


# --- abstention --------------------------------------------------------------
def test_without_a_threshold_a_weak_call_is_still_binary() -> None:
    classifier, _ = _classifier([[0.51, 0.49]])
    prediction = classifier.classify([crop()])[0]
    assert prediction.label == "helmet"
    assert prediction.score == pytest.approx(0.51)


def test_below_the_floor_the_backend_abstains_with_the_real_score() -> None:
    """The score is reported honestly, not discarded and not rounded up."""

    classifier, _ = _classifier([[0.51, 0.49]], abstain_below=0.6)
    prediction = classifier.classify([crop()])[0]
    assert prediction.label == ABSTAIN_LABEL
    assert prediction.score == pytest.approx(0.51)


def test_at_or_above_the_floor_the_call_stands() -> None:
    classifier, _ = _classifier([[0.6, 0.4]], abstain_below=0.6)
    assert classifier.classify([crop()])[0].label == "helmet"


# --- capability declaration --------------------------------------------------
def test_the_backend_declares_it_cannot_emit_turban() -> None:
    """The whole point of the declaration: a binary model says so out loud."""

    classifier, _ = _classifier([[1.0, 0.0]])
    assert "turban" not in classifier.supported_labels
    assert classifier.supported_labels == frozenset({"helmet", "no_helmet"})


def test_declaring_an_abstain_threshold_adds_uncertain_to_the_vocabulary() -> None:
    classifier, _ = _classifier([[1.0, 0.0]], abstain_below=0.6)
    assert classifier.supported_labels == frozenset({"helmet", "no_helmet", "uncertain"})


def test_the_declared_vocabulary_is_a_promise_the_backend_keeps() -> None:
    """Every label the backend can emit must be inside its declaration."""

    classifier, _ = _classifier([[0.9, 0.1], [0.1, 0.9], [0.51, 0.49]], abstain_below=0.6)
    declared = classifier.supported_labels
    emitted = {p.label for p in classifier.classify([crop("a"), crop("b"), crop("c")])}
    assert emitted <= declared


# --- configuration -----------------------------------------------------------
def test_the_device_string_is_validated() -> None:
    with pytest.raises(ValueError, match="device must be"):
        _config(device="tpu")


def test_a_non_positive_temperature_is_rejected() -> None:
    with pytest.raises(ValueError):
        _config(temperature=0.0)


def test_the_config_is_frozen_and_strict() -> None:
    config = _config()
    with pytest.raises(ValueError):
        config.device = "cuda"  # type: ignore[misc]
    with pytest.raises(ValueError):
        ResNetHelmetConfig(checkpoint=Path("x.pt"), nonsense=1)  # type: ignore[call-arg]


# --- preprocessing fidelity --------------------------------------------------
def test_square_pad_resize_matches_the_frozen_experiment_geometry() -> None:
    """Byte-for-byte agreement with ``helmet_cnn_vit.extract.square_pad_resize``.

    Not a reimplementation test for its own sake: a silent divergence here is a
    train/serve skew that would look like a model quality problem.
    """

    pytest.importorskip("PIL")
    from PIL import Image as PilImage

    rng = np.random.default_rng(0)
    for height, width in ((10, 30), (30, 10), (224, 224), (7, 7)):
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

        ours = square_pad_resize(image)

        # The experiment's arithmetic, inlined so the test does not import the
        # frozen package (which would couple this suite to the research venv).
        pil = PilImage.fromarray(image, mode="RGB")
        side = max(width, height)
        if width != height:
            canvas = PilImage.new("RGB", (side, side), PAD_RGB)
            canvas.paste(pil, ((side - width) // 2, (side - height) // 2))
            pil = canvas
        if pil.size != (IMAGE_SIZE, IMAGE_SIZE):
            pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), PilImage.BILINEAR)
        expected = np.asarray(pil, dtype=np.uint8)

        assert np.array_equal(ours, expected), f"divergence at {height}x{width}"


def test_preprocessing_produces_the_trained_input_geometry() -> None:
    pytest.importorskip("PIL")
    out = square_pad_resize(np.zeros((10, 30, 3), dtype=np.uint8))
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out.dtype == np.uint8


def test_padding_uses_black_not_invented_texture() -> None:
    pytest.importorskip("PIL")
    # A white wide image padded to square must show the pad colour top and bottom.
    image = np.full((10, 30, 3), 255, dtype=np.uint8)
    out = square_pad_resize(image)
    assert tuple(int(v) for v in out[0, out.shape[1] // 2]) == PAD_RGB


def test_a_zero_area_crop_is_a_typed_error() -> None:
    pytest.importorskip("PIL")
    with pytest.raises(MalformedResNetOutputError):
        square_pad_resize(np.zeros((0, 5, 3), dtype=np.uint8))


def test_a_non_rgb_crop_is_a_typed_error() -> None:
    pytest.importorskip("PIL")
    with pytest.raises(MalformedResNetOutputError):
        square_pad_resize(np.zeros((4, 4), dtype=np.uint8))


# --- the real checkpoint (opt-in) --------------------------------------------
CHECKPOINT_ENV = "TRAFFICPULSE_HELMET_RESNET_CHECKPOINT"


@pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=(
        "opt-in real P4-U5 ResNet checkpoint test: set "
        f"{CHECKPOINT_ENV} to a locally-available best.pt"
    ),
)
def test_the_real_checkpoint_loads_into_torchvision_and_scores_a_crop() -> None:
    """Proves the torchvision claim end to end: strict load, then a real forward pass."""

    checkpoint = Path(os.environ[CHECKPOINT_ENV])
    classifier = ResNetHelmetClassifier(ResNetHelmetConfig(checkpoint=checkpoint))
    predictions = classifier.classify([crop("a"), crop("b")])
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.label in classifier.supported_labels
        assert 0.0 <= prediction.score <= 1.0

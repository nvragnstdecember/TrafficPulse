"""The zero-shot helmet backend, exercised through its fake-engine seam (P4-U3).

Every test here runs **without** torch, transformers, weights, a dataset, or a
network: the backend's internal :class:`ZeroShotInferenceEngine` is framework-neutral
by design, so a fake engine exercises the whole backend except the ~15 lines that
actually call the framework (those are covered by the opt-in
``test_zeroshot_smoke.py``). This is the P1-U7 fake-engine pattern.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from trafficpulse.classifier import (
    DEFAULT_HELMET_PROMPTS,
    Crop,
    HelmetClassifier,
    HelmetClassifierError,
    MalformedBackendOutputError,
    MissingCropImageError,
    RawHelmetPrediction,
    ZeroShotBackendError,
    ZeroShotHelmetClassifier,
    ZeroShotHelmetConfig,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)
PROMPTS = {"helmet": "a rider wearing a helmet", "no_helmet": "a bare-headed rider"}


class _FakeEngine:
    """A scripted, framework-free ``ZeroShotInferenceEngine``."""

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._rows = [list(r) for r in rows]
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def infer(
        self, images: Sequence[NDArray[np.uint8]], prompts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        self.calls.append((len(images), tuple(prompts)))
        return self._rows


def crop(track_id: str = "r", frame_index: int = 0, *, with_pixels: bool = True) -> Crop:
    return Crop(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=BASE,
        track_id=track_id,
        image=np.zeros((8, 8, 3), dtype=np.uint8) if with_pixels else None,
    )


def _classifier(
    rows: Sequence[Sequence[float]], prompts: dict[str, str] | None = None
) -> tuple[ZeroShotHelmetClassifier, _FakeEngine]:
    engine = _FakeEngine(rows)
    classifier = ZeroShotHelmetClassifier(
        ZeroShotHelmetConfig(checkpoint="fake/checkpoint", prompts=prompts or PROMPTS),
        engine=engine,
    )
    return classifier, engine


# --- it is a HelmetClassifier, unchanged -------------------------------------
def test_backend_satisfies_the_unchanged_seam() -> None:
    classifier, _ = _classifier([[0.9, 0.1]])
    assert isinstance(classifier, HelmetClassifier)


def test_returns_raw_predictions_only() -> None:
    classifier, _ = _classifier([[0.9, 0.1]])
    predictions = classifier.classify([crop()])

    assert all(isinstance(p, RawHelmetPrediction) for p in predictions)
    assert type(predictions[0].label) is str
    assert type(predictions[0].score) is float


def test_picks_the_best_matching_prompt() -> None:
    classifier, _ = _classifier([[0.2, 0.8]])
    assert classifier.classify([crop()])[0] == RawHelmetPrediction("no_helmet", 0.8)


def test_native_label_is_the_prompt_key() -> None:
    classifier, _ = _classifier([[0.7, 0.3]], prompts={"HAT": "a hat", "BARE": "a bare head"})
    assert classifier.classify([crop()])[0].label == "HAT"


# --- the batch contract (P4-U2) ----------------------------------------------
def test_one_prediction_per_crop_in_input_order() -> None:
    classifier, _ = _classifier([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]])
    predictions = classifier.classify([crop("a"), crop("b"), crop("c")])

    assert [p.label for p in predictions] == ["helmet", "no_helmet", "helmet"]


def test_batches_every_crop_into_one_engine_call() -> None:
    """The whole point of the batch seam: N crops, ONE forward pass."""

    classifier, engine = _classifier([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    classifier.classify([crop("a"), crop("b"), crop("c")])

    assert len(engine.calls) == 1
    assert engine.calls[0][0] == 3


def test_empty_input_never_touches_the_engine() -> None:
    classifier, engine = _classifier([])
    assert tuple(classifier.classify(())) == ()
    assert engine.calls == []


def test_prompts_are_passed_in_declared_order() -> None:
    classifier, engine = _classifier([[0.9, 0.1]])
    classifier.classify([crop()])

    assert engine.calls[0][1] == tuple(PROMPTS.values())


# --- determinism -------------------------------------------------------------
def test_classification_is_deterministic() -> None:
    classifier, _ = _classifier([[0.9, 0.1]])
    assert classifier.classify([crop()]) == classifier.classify([crop()])


def test_exact_tie_resolves_to_the_first_declared_label() -> None:
    """A tie must not depend on iteration accidents."""

    classifier, _ = _classifier([[0.5, 0.5]])
    assert classifier.classify([crop()])[0].label == "helmet"


def test_near_tie_surfaces_as_a_low_score_not_an_uncertain_label() -> None:
    """This backend applies no abstention policy; it reports the relative score.

    A near-tie is the signal a P4-U4 quality gate uses to route to ``uncertain``.
    """

    classifier, _ = _classifier([[0.51, 0.49]])
    prediction = classifier.classify([crop()])[0]

    assert prediction.label == "helmet"
    assert prediction.score == pytest.approx(0.51)


# --- error taxonomy ----------------------------------------------------------
def test_missing_pixels_raise_a_classifier_error() -> None:
    classifier, _ = _classifier([[0.9, 0.1]])
    with pytest.raises(MissingCropImageError) as excinfo:
        classifier.classify([crop(with_pixels=False)])

    assert isinstance(excinfo.value, HelmetClassifierError)


def test_wrong_row_count_is_malformed_output() -> None:
    classifier, _ = _classifier([[0.9, 0.1]])  # one row for two crops
    with pytest.raises(MalformedBackendOutputError):
        classifier.classify([crop("a"), crop("b")])


def test_wrong_score_count_is_malformed_output() -> None:
    classifier, _ = _classifier([[0.9, 0.05, 0.05]])  # three scores for two prompts
    with pytest.raises(MalformedBackendOutputError):
        classifier.classify([crop()])


def test_non_finite_score_is_malformed_output() -> None:
    classifier, _ = _classifier([[float("nan"), 0.1]])
    with pytest.raises(MalformedBackendOutputError):
        classifier.classify([crop()])


def test_backend_errors_are_classifier_errors() -> None:
    """Callers can catch the seam's base error across every backend."""

    assert issubclass(ZeroShotBackendError, HelmetClassifierError)
    assert issubclass(MissingCropImageError, ZeroShotBackendError)


def test_score_is_clamped_into_range() -> None:
    """Float error must not produce a score the P4-U4 Confidence bound rejects."""

    classifier, _ = _classifier([[1.0000000001, 0.0]])
    assert classifier.classify([crop()])[0].score <= 1.0


# --- configuration -----------------------------------------------------------
def test_config_is_frozen_and_strict() -> None:
    config = ZeroShotHelmetConfig(checkpoint="x", prompts=PROMPTS)
    with pytest.raises(ValidationError):
        config.checkpoint = "y"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig(checkpoint="x", weights_path="w.pt")  # type: ignore[call-arg]


def test_config_ships_no_default_checkpoint() -> None:
    """ADR-001: the operator names the artifact; this unit blesses none."""

    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig()  # type: ignore[call-arg]


def test_config_defaults_to_offline() -> None:
    assert ZeroShotHelmetConfig(checkpoint="x").local_files_only is True


def test_single_prompt_is_rejected() -> None:
    """A lone prompt would softmax to 1.0 and assert its label unconditionally."""

    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig(checkpoint="x", prompts={"helmet": "a helmet"})


def test_empty_prompt_text_or_label_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig(checkpoint="x", prompts={"a": "text", "": "other"})
    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig(checkpoint="x", prompts={"a": "text", "b": "  "})


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ZeroShotHelmetConfig(checkpoint="x", device="tpu")


def test_default_prompts_offer_no_uncertain_class() -> None:
    """Uncertainty is the absence of a match, not a visual class to prompt for."""

    assert "uncertain" not in DEFAULT_HELMET_PROMPTS
    assert set(DEFAULT_HELMET_PROMPTS) == {"helmet", "no_helmet", "turban"}


def test_constructing_config_loads_nothing() -> None:
    """Config construction must not import a framework or touch an artifact."""

    ZeroShotHelmetConfig(checkpoint="definitely/not-a-real-model")  # must not raise


# --- the boundary invariant --------------------------------------------------
def test_importing_the_backend_pulls_in_no_ml_framework() -> None:
    ml = [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]
    saved = {name: sys.modules.pop(name) for name in ml}
    try:
        for module in ("trafficpulse.classifier", "trafficpulse.classifier.zeroshot"):
            sys.modules.pop(module, None)
        importlib.import_module("trafficpulse.classifier.zeroshot")
        leaked = [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]
        assert leaked == [], f"the zero-shot backend imported an ML framework: {leaked}"
    finally:
        sys.modules.update(saved)
        importlib.import_module("trafficpulse.classifier")

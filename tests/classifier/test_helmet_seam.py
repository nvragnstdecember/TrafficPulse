"""The classifier seam contract + scripted stub (P4-U2).

Guards the boundary itself: the ABC's abstractness, the batch in/batch out
contract, the stub's script resolution and abstention-safe default, determinism,
and the framework-neutrality invariant ADR-001 requires of this seam.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from trafficpulse.classifier import (
    UNCERTAIN,
    Crop,
    HelmetClassifier,
    HelmetClassifierError,
    RawHelmetPrediction,
    StubHelmetClassifier,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)

HELMET = RawHelmetPrediction(label="helmet", score=0.9)
NO_HELMET = RawHelmetPrediction(label="no_helmet", score=0.8)
TURBAN = RawHelmetPrediction(label="turban", score=0.7)


def crop(frame_index: int, track_id: str, *, with_pixels: bool = False) -> Crop:
    image = np.zeros((4, 4, 3), dtype=np.uint8) if with_pixels else None
    return Crop(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=BASE,
        track_id=track_id,
        image=image,
    )


# --- the interface -----------------------------------------------------------
def test_helmet_classifier_is_abstract() -> None:
    with pytest.raises(TypeError):
        HelmetClassifier()  # type: ignore[abstract]


def test_subclass_must_implement_classify() -> None:
    class Incomplete(HelmetClassifier):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_stub_satisfies_the_interface() -> None:
    assert isinstance(StubHelmetClassifier(), HelmetClassifier)


def test_error_taxonomy_base_is_an_exception() -> None:
    assert issubclass(HelmetClassifierError, Exception)


# --- the batch contract ------------------------------------------------------
def test_one_prediction_per_crop_in_input_order() -> None:
    classifier = StubHelmetClassifier(per_track={"a": HELMET, "b": NO_HELMET, "c": TURBAN})
    crops = [crop(0, "c"), crop(0, "a"), crop(0, "b")]

    predictions = classifier.classify(crops)

    assert len(predictions) == len(crops)
    assert list(predictions) == [TURBAN, HELMET, NO_HELMET]


def test_empty_input_returns_empty_output() -> None:
    assert tuple(StubHelmetClassifier().classify(())) == ()


def test_result_is_an_immutable_tuple() -> None:
    """A caller must not be able to mutate the stub's script through the result."""

    result = StubHelmetClassifier().classify([crop(0, "a")])
    assert isinstance(result, tuple)


# --- script resolution (most specific wins) ----------------------------------
def test_per_crop_beats_per_track_beats_default() -> None:
    classifier = StubHelmetClassifier(
        HELMET,
        per_track={"rider": NO_HELMET},
        per_crop={(7, "rider"): TURBAN},
    )

    assert classifier.classify([crop(7, "rider")])[0] == TURBAN  # per_crop wins
    assert classifier.classify([crop(8, "rider")])[0] == NO_HELMET  # per_track wins
    assert classifier.classify([crop(8, "other")])[0] == HELMET  # default


def test_per_crop_is_keyed_by_frame_and_track() -> None:
    """A temporal script: one rider whose state differs between frames."""

    classifier = StubHelmetClassifier(per_crop={(0, "r"): HELMET, (1, "r"): NO_HELMET})

    assert classifier.classify([crop(0, "r"), crop(1, "r")]) == (HELMET, NO_HELMET)


def test_unscripted_crop_abstains_rather_than_guessing() -> None:
    """An unscripted rider must never fabricate a helmet state (U3 ontology)."""

    assert StubHelmetClassifier().classify([crop(0, "unknown")])[0] == UNCERTAIN
    assert UNCERTAIN.label == "uncertain"


def test_stub_script_is_copied_at_construction() -> None:
    """Mutating the caller's mapping afterwards must not change the stub."""

    script = {"r": HELMET}
    classifier = StubHelmetClassifier(per_track=script)
    script["r"] = NO_HELMET

    assert classifier.classify([crop(0, "r")])[0] == HELMET


# --- determinism + statelessness ---------------------------------------------
def test_classify_is_deterministic_across_repeated_calls() -> None:
    classifier = StubHelmetClassifier(per_track={"r": NO_HELMET})
    crops = [crop(0, "r"), crop(1, "r")]

    assert classifier.classify(crops) == classifier.classify(crops)


def test_classifier_is_stateless_across_calls() -> None:
    """Order of previous calls must not influence a later prediction."""

    classifier = StubHelmetClassifier(per_track={"a": HELMET, "b": NO_HELMET})
    classifier.classify([crop(0, "b"), crop(0, "b"), crop(0, "a")])

    assert classifier.classify([crop(0, "a")])[0] == HELMET


def test_stub_ignores_pixels_entirely() -> None:
    """The stub is a pure function of crop identity; pixels change nothing."""

    classifier = StubHelmetClassifier(per_track={"r": NO_HELMET})

    assert classifier.classify([crop(0, "r", with_pixels=True)]) == classifier.classify(
        [crop(0, "r", with_pixels=False)]
    )


# --- boundary invariants -----------------------------------------------------
def test_importing_the_classifier_package_pulls_in_no_ml_framework() -> None:
    """The P4-U2 foundation carries no ML dependency (mirrors the P1-U6 invariant).

    torch/transformers may be *installed* (the optional rtdetr extra), so assert on
    what this package's own import graph pulls in: re-import it with the ML modules
    evicted and confirm none is re-imported.
    """

    ml_modules = [name for name in sys.modules if name.split(".")[0] in {"torch", "transformers"}]
    saved = {name: sys.modules.pop(name) for name in ml_modules}
    try:
        for module in ("trafficpulse.classifier", "trafficpulse.classifier.stub"):
            sys.modules.pop(module, None)
        importlib.import_module("trafficpulse.classifier")
        leaked = [n for n in sys.modules if n.split(".")[0] in {"torch", "transformers"}]
        assert leaked == [], f"classifier package pulled in ML framework: {leaked}"
    finally:
        sys.modules.update(saved)
        importlib.import_module("trafficpulse.classifier")


def test_only_raw_predictions_cross_the_seam() -> None:
    """No framework-native object escapes: the seam emits RawHelmetPrediction only."""

    predictions: Sequence[RawHelmetPrediction] = StubHelmetClassifier(
        per_track={"r": NO_HELMET}
    ).classify([crop(0, "r", with_pixels=True)])

    assert all(isinstance(p, RawHelmetPrediction) for p in predictions)
    assert all(isinstance(p.label, str) and isinstance(p.score, float) for p in predictions)


def test_crop_compares_by_identity_not_pixels() -> None:
    """Pixel payloads must not affect equality (and must not raise on NumPy ==)."""

    a = crop(0, "r", with_pixels=True)
    b = crop(0, "r", with_pixels=False)

    assert a == b


def test_raw_prediction_label_is_a_plain_string_not_an_enum() -> None:
    """Backend vocabularies are mapped at the adapter (P4-U4), not in the backend."""

    assert type(NO_HELMET.label) is str

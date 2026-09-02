"""Backend capability declaration, and the turban regression it exists to prevent.

The failure being guarded is silent, which is what makes it dangerous. Swap a binary
helmet/no_helmet backend behind the no-helmet rule and nothing raises: the classifier
is correct, the adapter is correct, the reasoner is correct, and every existing test
passes. Only ``rules.no_helmet.exempt_riders`` quietly returns an empty set forever,
so turban-wearing riders are confirmed as violations -- a systematic false-positive
class against a religious group, and a reversal of the H8 real-footage fix.

These tests pin both halves: that the capability check catches it, and that
``exempt_riders`` really does go dead when ``turban`` never arrives (so the guard is
protecting against a real behaviour, not a hypothetical one).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.classifier import (
    Crop,
    HelmetClassifier,
    RawHelmetPrediction,
    StubHelmetClassifier,
)
from trafficpulse.classifier.capabilities import (
    TURBAN_LABEL,
    ClassifierCapabilityError,
    missing_labels,
    require_turban_capability,
)
from trafficpulse.contracts.enums import HelmetState, ProducerKind
from trafficpulse.contracts.observations import HelmetStateObservation, Producer
from trafficpulse.rules.no_helmet import exempt_riders

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="test", version="1", kind=ProducerKind.MODEL)


class _Undeclared(HelmetClassifier):
    """A backend that declares nothing -- the default, and the existing posture."""

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        return tuple(RawHelmetPrediction(label="helmet", score=1.0) for _ in crops)


class _Binary(HelmetClassifier):
    """A backend that declares it can only say helmet/no_helmet."""

    @property
    def supported_labels(self) -> frozenset[str]:
        return frozenset({"helmet", "no_helmet"})

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        return tuple(RawHelmetPrediction(label="no_helmet", score=1.0) for _ in crops)


class _TurbanCapable(HelmetClassifier):
    """A backend that declares the full four-label vocabulary."""

    @property
    def supported_labels(self) -> frozenset[str]:
        return frozenset({"helmet", "no_helmet", "turban", "uncertain"})

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        return tuple(RawHelmetPrediction(label="turban", score=1.0) for _ in crops)


# --- the declaration ---------------------------------------------------------
def test_the_seam_declares_nothing_by_default() -> None:
    """Declaring capability is opt-in, so no existing backend changes behaviour."""

    assert _Undeclared().supported_labels is None
    assert StubHelmetClassifier(()).supported_labels is None


def test_undeclared_is_not_treated_as_incapable() -> None:
    """Unknown is not the same as impossible; only a declaration can prove absence."""

    assert missing_labels(_Undeclared(), frozenset({TURBAN_LABEL})) == frozenset()
    assert require_turban_capability(_Undeclared()) == frozenset()


def test_a_declared_binary_backend_reports_turban_missing() -> None:
    assert missing_labels(_Binary(), frozenset({TURBAN_LABEL})) == frozenset({TURBAN_LABEL})


def test_a_turban_capable_backend_reports_nothing_missing() -> None:
    assert missing_labels(_TurbanCapable(), frozenset({TURBAN_LABEL})) == frozenset()


# --- the check ---------------------------------------------------------------
def test_a_turban_blind_backend_is_refused_by_default() -> None:
    with pytest.raises(ClassifierCapabilityError, match="turban"):
        require_turban_capability(_Binary())


def test_the_refusal_explains_the_consequence_not_just_the_fact() -> None:
    """An operator reading this must learn what would go wrong, not only what is absent."""

    with pytest.raises(ClassifierCapabilityError) as excinfo:
        require_turban_capability(_Binary())
    message = str(excinfo.value)
    assert "exempt_riders" in message
    assert "confirmed as a no-helmet violation" in message
    assert "acknowledge_turban_blind" in message


def test_an_acknowledged_blindness_is_permitted_but_still_reported() -> None:
    """Acknowledging does not make the consequence vanish; it records the choice."""

    missing = require_turban_capability(_Binary(), acknowledged=True)
    assert missing == frozenset({TURBAN_LABEL})


def test_a_capable_backend_passes_and_reports_nothing_missing() -> None:
    assert require_turban_capability(_TurbanCapable()) == frozenset()


def test_the_capability_error_is_not_a_classification_error() -> None:
    """Nothing failed at the seam -- the backend/policy *combination* is unsound."""

    from trafficpulse.classifier.errors import HelmetClassifierError

    assert not issubclass(ClassifierCapabilityError, HelmetClassifierError)


# --- the behaviour being guarded ---------------------------------------------
def _observation(index: int, state: HelmetState) -> HelmetStateObservation:
    return HelmetStateObservation(
        observation_id=f"o{index}",
        camera_id="cam-1",
        track_id="rider-1",
        timestamp=BASE + timedelta(seconds=index),
        helmet_state=state,
        confidence=0.9,
        producer=PRODUCER,
    )


def test_exempt_riders_really_does_go_dead_without_turban_observations() -> None:
    """The guard protects a real behaviour: no turban in, no exemption out, ever."""

    binary_only = [_observation(i, HelmetState.NO_HELMET) for i in range(5)]
    assert exempt_riders(binary_only) == frozenset()

    with_turban = [*binary_only, *(_observation(5 + i, HelmetState.TURBAN) for i in range(5))]
    assert exempt_riders(with_turban) == frozenset({"rider-1"})

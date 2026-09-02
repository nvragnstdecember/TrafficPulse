"""An analysis-only engine: it classifies, and it cannot confirm.

The rule/analysis split exists because the two claims came as a package -- the only way
to obtain helmet state was to configure a violation rule, which forced a turban-blind
backend to choose between enforcing unsafely and seeing nothing at all. These tests
pin the property that makes the split worth having: a run can classify every rider and
still be structurally incapable of minting an event.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from _engine_helpers import SCENE, frame_records, stub_engine
from _helmet_fixtures import scripted_helmet_classifier
from _triple_fixtures import scripted_rider_count_detector

from trafficpulse.classifier import Crop, HelmetClassifier, RawHelmetPrediction
from trafficpulse.contracts import ObjectClass
from trafficpulse.detector import DetectorConfig
from trafficpulse.engine import (
    EngineConfig,
    HelmetAnalysisConfig,
    IterableFrameSource,
    NoHelmetRuleConfig,
    build_analyses,
    build_rules,
)
from trafficpulse.engine.errors import EngineConfigurationError
from trafficpulse.observations.helmet_stability import HelmetStabilizationConfig
from trafficpulse.pipeline.helmet_analysis import HelmetAnalysisObserver

#: The native labels the real RT-DETR checkpoint emits for this traffic, mapped the
#: way ``serve.py`` maps them. A wrong key here would silently disable rider
#: association altogether, so it mirrors the production map rather than inventing one.
RIDER_DETECTOR_CONFIG = DetectorConfig(
    label_map={"motorbike": ObjectClass.MOTORCYCLE, "person": ObjectClass.PERSON}
)


class _TurbanBlindClassifier(HelmetClassifier):
    """Declares a binary vocabulary, like the trained P4-U5 ResNet-50."""

    @property
    def supported_labels(self) -> frozenset[str]:
        return frozenset({"helmet", "no_helmet"})

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        return tuple(RawHelmetPrediction(label="no_helmet", score=0.95) for _ in crops)


# --- the registry -------------------------------------------------------------------
def test_an_analysis_contributes_an_observer_and_no_strategy() -> None:
    """The structural guarantee: there is nowhere for an event to come from."""

    built = build_analyses((HelmetAnalysisConfig(),), classifier=scripted_helmet_classifier())

    assert len(built) == 1
    assert isinstance(built[0].observer, HelmetAnalysisObserver)
    assert not hasattr(built[0], "strategy")


def test_an_analysis_without_a_classifier_fails_fast() -> None:
    """The same fail-fast the rule gets -- never a silent no-op."""

    with pytest.raises(EngineConfigurationError, match="helmet_analysis"):
        build_analyses((HelmetAnalysisConfig(),), classifier=None)


def test_the_analysis_carries_its_stabilization_policy_to_the_observer() -> None:
    policy = HelmetStabilizationConfig(window=9, min_samples=2)

    built = build_analyses(
        (HelmetAnalysisConfig(stabilization=policy),),
        classifier=scripted_helmet_classifier(),
    )

    assert isinstance(built[0].observer, HelmetAnalysisObserver)
    assert built[0].observer.stabilization == policy


# --- the guard is not routed around ---------------------------------------------------
def test_an_analysis_runs_on_a_backend_whose_rule_is_refused() -> None:
    """The whole point: classify honestly while the turban-dependent rule stays refused.

    Same backend, same engine, two declarations -- the rule is refused, the analysis
    builds. An analysis asserts no violation and grants no exemption, so there is
    nothing for a missing label to silently disable.
    """

    binary = _TurbanBlindClassifier()

    with pytest.raises(EngineConfigurationError, match="turban"):
        build_rules((NoHelmetRuleConfig(),), scene=SCENE, classifier=binary)

    assert len(build_analyses((HelmetAnalysisConfig(),), classifier=binary)) == 1


def test_declaring_an_analysis_does_not_relax_the_rule_guard() -> None:
    """Configuring both must not make the refused rule buildable."""

    with pytest.raises(EngineConfigurationError, match="turban"):
        build_rules(
            (NoHelmetRuleConfig(),), scene=SCENE, classifier=_TurbanBlindClassifier()
        )


# --- engine composition ----------------------------------------------------------------
def test_an_engine_may_be_configured_with_analysis_and_no_rules() -> None:
    """Observing without enforcing is a legitimate deployment, not a misconfiguration."""

    config = EngineConfig(rules=(), analysis=(HelmetAnalysisConfig(),))

    assert config.analysis == (HelmetAnalysisConfig(),)


def test_an_engine_configured_to_do_nothing_at_all_is_still_refused() -> None:
    with pytest.raises(EngineConfigurationError, match="at least one rule or analysis"):
        EngineConfig(rules=())


def _rider_engine(riders: int = 1):
    """An analysis-only engine over a scripted motorcycle-with-``riders`` stream.

    Reuses the triple-riding fixture's detector script, which emits the same native
    ``motorbike``/``person`` labels the real RT-DETR checkpoint does -- so the label
    map, the association and the head-crop geometry exercised here are the shipped
    ones rather than a parallel arrangement built for this test.
    """

    return stub_engine(
        detector=scripted_rider_count_detector(riders=riders),
        rules=(),
        analysis=(HelmetAnalysisConfig(),),
        detector_config=RIDER_DETECTOR_CONFIG,
        classifier=scripted_helmet_classifier(),
    )


def _analysis_observer(engine: object) -> HelmetAnalysisObserver:
    return next(
        candidate
        for candidate in engine.frame_observers()  # type: ignore[attr-defined]
        if isinstance(candidate, HelmetAnalysisObserver)
    )


def test_an_analysis_only_run_classifies_riders_and_confirms_nothing() -> None:
    """End to end over the real front half: observations yes, events no."""

    engine, _sink = _rider_engine()

    result = engine.run(IterableFrameSource(frame_records(24), source_id="vsrc-test"))

    assert result.events == ()
    # The classifier really ran: this is perception, not a disabled stage.
    assert _analysis_observer(engine).report().riders_observed > 0


def test_an_analysis_only_run_over_multi_rider_traffic_still_confirms_nothing() -> None:
    """The case the demo will actually meet, and the one most tempting to guess at."""

    engine, _sink = _rider_engine(riders=3)

    result = engine.run(IterableFrameSource(frame_records(24), source_id="vsrc-test"))
    report = _analysis_observer(engine).report()

    assert result.events == ()
    assert report.multi_rider_riders == report.riders_observed > 0
    assert report.eligible_riders == 0


def test_the_analysis_observer_is_surfaced_for_the_overlay_driver() -> None:
    """The composition root retrieves it the same way it retrieves a rule's observer."""

    engine, _sink = _rider_engine()

    assert any(
        isinstance(observer, HelmetAnalysisObserver)
        for observer in engine.frame_observers()
    )

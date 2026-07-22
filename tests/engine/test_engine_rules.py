"""Rule registry + multi-rule composition (H6)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from _engine_helpers import DETECTOR_CONFIG, NORTH_DIRECTION_ID, SCENE, frame_records
from _helmet_fixtures import scripted_helmet_classifier
from _pipeline_helpers import moving_down_detector

from trafficpulse.contracts import TrackState
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector.frame import Frame
from trafficpulse.engine import (
    CompositeFrameObserver,
    EngineConfigurationError,
    IllegalStoppingRuleConfig,
    MultiRuleFinalize,
    NoHelmetRuleConfig,
    UnsupportedRuleError,
    WrongWayRuleConfig,
    build_rules,
    require_shipped,
)
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.pipeline.errors import SceneConfigurationError
from trafficpulse.tracking import IouTracker


# --- registry --------------------------------------------------------------------
def test_build_rules_realises_all_three_shipped_rules() -> None:
    from _stopping_fixtures import illegal_stopping_test_scene

    built = build_rules(
        (
            WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),
            IllegalStoppingRuleConfig(),
            NoHelmetRuleConfig(),
        ),
        scene=illegal_stopping_test_scene(),
        classifier=scripted_helmet_classifier(),
    )
    assert [rule.violation for rule in built] == [
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.NO_HELMET,
    ]
    assert built[0].observer is None
    assert built[1].observer is None
    assert built[2].observer is not None  # the pixel side-channel


def test_no_helmet_without_classifier_fails_fast() -> None:
    with pytest.raises(EngineConfigurationError, match="HelmetClassifier"):
        build_rules((NoHelmetRuleConfig(),), scene=SCENE, classifier=None)


def test_scene_resolution_failures_propagate_unchanged() -> None:
    with pytest.raises(SceneConfigurationError):
        # The example scene declares two legal directions; wrong-way needs an id.
        build_rules((WrongWayRuleConfig(),), scene=SCENE)


def test_require_shipped_names_the_gap() -> None:
    for violation in (
        ViolationType.RED_LIGHT_JUMPING,
        ViolationType.SPEEDING,
    ):
        with pytest.raises(UnsupportedRuleError, match=violation.value):
            require_shipped(violation)
    for violation in (
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.NO_HELMET,
        ViolationType.TRIPLE_RIDING,  # shipped in v1.1 U3
    ):
        require_shipped(violation)  # shipped: no error


# --- composite observer -----------------------------------------------------------
class _RecordingObserver:
    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        self._log.append(f"{self._name}:observe:{frame.frame_index}")

    def reset(self) -> None:
        self._log.append(f"{self._name}:reset")


def test_composite_observer_fans_out_in_order() -> None:
    log: list[str] = []
    composite = CompositeFrameObserver(
        [_RecordingObserver(log, "a"), _RecordingObserver(log, "b")]
    )
    from datetime import UTC, datetime

    frame = Frame(camera_id="cam", frame_index=3, timestamp=datetime(1970, 1, 1, tzinfo=UTC))
    composite.observe(frame, [])
    composite.reset()
    assert log == ["a:observe:3", "b:observe:3", "a:reset", "b:reset"]


# --- multi-rule finalize equivalence -----------------------------------------------
def test_single_rule_composition_equals_the_standalone_pipeline() -> None:
    """The load-bearing composition claim: MultiRuleFinalize adds no behaviour."""

    records = frame_records(45)
    standalone = WrongWayPipeline(
        detector=moving_down_detector(45),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    ).process(records)
    assert standalone  # the scenario genuinely confirms at least one event

    from trafficpulse.pipeline.base import CompositionPipeline

    composed = CompositionPipeline(
        detector=moving_down_detector(45),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=MultiRuleFinalize(
            build_rules((WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),), scene=SCENE)
        ),
    ).process(records)
    assert composed == standalone


def test_multi_rule_events_are_the_union_of_per_rule_events() -> None:
    from trafficpulse.pipeline.base import CompositionPipeline

    records = frame_records(45)
    strategy = MultiRuleFinalize(
        build_rules(
            (
                WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),
                WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),
            ),
            scene=SCENE,
        )
    )
    events = CompositionPipeline(
        detector=moving_down_detector(45),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=strategy,
    ).process(records)
    single = WrongWayPipeline(
        detector=moving_down_detector(45),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    ).process(records)
    # Two identical rules produce the same event twice -- proving per-rule fan-out
    # feeds every rule the full track history independently.
    assert len(events) == 2 * len(single)

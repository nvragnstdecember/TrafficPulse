"""Engine configuration validation (H6)."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from trafficpulse.contracts import ObjectClass
from trafficpulse.contracts.enums import SignalState
from trafficpulse.engine import (
    EngineConfig,
    EngineConfigurationError,
    IllegalStoppingRuleConfig,
    InferenceConfig,
    NoHelmetRuleConfig,
    RedLightRuleConfig,
    RuleConfig,
    SchedulerConfig,
    SignalPhaseSpec,
    WrongWayRuleConfig,
)

_LABELS = {"car": ObjectClass.CAR}


# --- scheduler -----------------------------------------------------------------
def test_scheduler_defaults() -> None:
    config = SchedulerConfig()
    assert (config.frame_stride, config.target_fps, config.queue_capacity) == (1, None, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_stride": 0},
        {"target_fps": 0.0},
        {"target_fps": -5.0},
        {"queue_capacity": 0},
    ],
)
def test_scheduler_field_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(**kwargs)  # type: ignore[arg-type]


def test_scheduler_is_frozen_and_strict() -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(unknown_field=1)  # type: ignore[call-arg]


# --- inference -------------------------------------------------------------------
def test_inference_accepts_auto_and_explicit_devices() -> None:
    for device in ("auto", "cpu", "cuda", "cuda:1"):
        config = InferenceConfig(checkpoint="ckpt", label_map=_LABELS, device=device)
        assert config.device == device


def test_inference_rejects_bad_device() -> None:
    with pytest.raises(EngineConfigurationError, match="device"):
        InferenceConfig(checkpoint="ckpt", label_map=_LABELS, device="tpu")


def test_inference_rejects_empty_label_map() -> None:
    with pytest.raises(EngineConfigurationError, match="label_map"):
        InferenceConfig(checkpoint="ckpt", label_map={})


def test_inference_rejects_empty_checkpoint() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(checkpoint="", label_map=_LABELS)


# --- rules -----------------------------------------------------------------------
def test_rule_union_parses_by_kind() -> None:
    adapter: TypeAdapter[RuleConfig] = TypeAdapter(RuleConfig)
    assert isinstance(adapter.validate_python({"kind": "wrong_way"}), WrongWayRuleConfig)
    assert isinstance(
        adapter.validate_python({"kind": "illegal_stopping"}), IllegalStoppingRuleConfig
    )
    assert isinstance(adapter.validate_python({"kind": "no_helmet"}), NoHelmetRuleConfig)


def test_rule_union_accepts_red_light_and_still_rejects_an_unshipped_kind() -> None:
    # H13 moved red_light_jumping from "has a contract, has no reasoner" into the
    # closed union. Speeding is now the only violation left on the other side.
    adapter: TypeAdapter[RuleConfig] = TypeAdapter(RuleConfig)

    red_light = adapter.validate_python(
        {"kind": "red_light_jumping", "schedule": [{"at_seconds": 0.0, "state": "red"}]}
    )
    assert isinstance(red_light, RedLightRuleConfig)
    assert red_light.schedule[0].state is SignalState.RED

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "speeding"})


def test_a_red_light_schedule_must_be_declared_in_time_order() -> None:
    # Out-of-order phases make the step function ambiguous, and the resolver would
    # silently pick one reading. Refuse the config instead -- as a pydantic
    # ValidationError, because this config is parsed from a client request body and
    # that is what the API turns into a clean 422.
    with pytest.raises(ValidationError):
        RedLightRuleConfig(
            schedule=(
                SignalPhaseSpec(at_seconds=10.0, state=SignalState.RED),
                SignalPhaseSpec(at_seconds=2.0, state=SignalState.GREEN),
            )
        )


def test_a_red_light_rule_defaults_to_no_approach_selection() -> None:
    # A single-approach scene needs no ids; they exist for the ambiguous case only.
    config = RedLightRuleConfig(schedule=(SignalPhaseSpec(at_seconds=0.0, state=SignalState.RED),))
    assert config.stop_line_id is None
    assert config.zone_id is None


def test_illegal_stopping_rule_defaults_match_derivation_layer() -> None:
    from trafficpulse.observations.stationary import (
        STATIONARY_EPSILON_PX,
        STATIONARY_WINDOW,
    )

    config = IllegalStoppingRuleConfig()
    assert config.stationary_window == STATIONARY_WINDOW
    assert config.stationary_epsilon_px == STATIONARY_EPSILON_PX


# --- engine ----------------------------------------------------------------------
def test_engine_config_requires_at_least_one_rule() -> None:
    with pytest.raises(EngineConfigurationError, match="at least one rule"):
        EngineConfig(rules=())


def test_engine_config_round_trips_json() -> None:
    config = EngineConfig(
        rules=(WrongWayRuleConfig(direction_id="dir-north"), NoHelmetRuleConfig()),
        scheduler=SchedulerConfig(frame_stride=2, target_fps=5.0),
        batch_size=4,
        inference=InferenceConfig(checkpoint="ckpt", label_map=_LABELS),
    )
    assert EngineConfig.model_validate_json(config.model_dump_json()) == config


def test_engine_config_batch_size_bound() -> None:
    with pytest.raises(ValidationError):
        EngineConfig(rules=(WrongWayRuleConfig(),), batch_size=0)

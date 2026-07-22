"""Training configuration validation + serialization (H4A)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _training_helpers import make_config
from helmet_rtdetr.errors import InvalidTrainingConfigError
from helmet_rtdetr.training import (
    AdamWConfig,
    CheckpointPolicy,
    CosineSchedulerConfig,
    ExperimentConfig,
    LoggingConfig,
    OneCycleSchedulerConfig,
    ResumeConfig,
    SgdConfig,
    StepSchedulerConfig,
)
from pydantic import ValidationError


# --- optimizer configs -------------------------------------------------------
def test_adamw_valid_defaults() -> None:
    config = AdamWConfig(lr=1e-4)
    assert config.kind == "adamw"
    assert config.weight_decay == 1e-4


def test_adamw_rejects_non_positive_lr() -> None:
    with pytest.raises(ValidationError):
        AdamWConfig(lr=0.0)


def test_adamw_rejects_out_of_range_betas() -> None:
    with pytest.raises(InvalidTrainingConfigError, match="betas"):
        AdamWConfig(lr=1e-4, betas=(0.9, 1.0))


def test_sgd_valid() -> None:
    config = SgdConfig(lr=0.01, momentum=0.9, nesterov=True)
    assert config.kind == "sgd"


def test_sgd_rejects_momentum_of_one() -> None:
    with pytest.raises(ValidationError):
        SgdConfig(lr=0.01, momentum=1.0)


def test_sgd_nesterov_requires_momentum() -> None:
    with pytest.raises(InvalidTrainingConfigError, match="momentum"):
        SgdConfig(lr=0.01, nesterov=True)


# --- scheduler configs -------------------------------------------------------
def test_cosine_defaults() -> None:
    assert CosineSchedulerConfig().warmup_steps == 0


def test_cosine_rejects_zero_min_lr_fraction() -> None:
    with pytest.raises(ValidationError):
        CosineSchedulerConfig(min_lr_fraction=0.0)


def test_step_requires_positive_step_size() -> None:
    with pytest.raises(ValidationError):
        StepSchedulerConfig(step_size=0)


def test_step_rejects_gamma_above_one() -> None:
    with pytest.raises(ValidationError):
        StepSchedulerConfig(step_size=2, gamma=1.5)


def test_one_cycle_pct_start_bounds() -> None:
    with pytest.raises(ValidationError):
        OneCycleSchedulerConfig(pct_start=1.0)
    with pytest.raises(ValidationError):
        OneCycleSchedulerConfig(pct_start=0.0)


# --- checkpoint policy -------------------------------------------------------
def test_policy_defaults_are_conservative() -> None:
    policy = CheckpointPolicy()
    assert policy.save_best is False
    assert policy.keep_last == 1
    assert policy.every_n_epochs is None


def test_policy_save_best_requires_metric() -> None:
    with pytest.raises(InvalidTrainingConfigError, match="best_metric"):
        CheckpointPolicy(save_best=True)


def test_policy_rejects_invalid_metric_name() -> None:
    with pytest.raises(InvalidTrainingConfigError, match="metric name"):
        CheckpointPolicy(save_best=True, best_metric="Val Loss!")


def test_policy_rejects_zero_keep_last() -> None:
    with pytest.raises(ValidationError):
        CheckpointPolicy(keep_last=0)


def test_policy_rejects_zero_every_n_epochs() -> None:
    with pytest.raises(ValidationError):
        CheckpointPolicy(every_n_epochs=0)


# --- experiment config -------------------------------------------------------
def test_full_config_round_trips_through_json(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    reloaded = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert reloaded == config


def test_discriminated_union_parses_the_right_optimizer(tmp_path: Path) -> None:
    raw = make_config(tmp_path).model_dump()
    raw["optimizer"] = {"kind": "sgd", "lr": 0.01}
    config = ExperimentConfig.model_validate(raw)
    assert isinstance(config.optimizer, SgdConfig)


def test_unknown_optimizer_kind_is_rejected(tmp_path: Path) -> None:
    raw = make_config(tmp_path).model_dump()
    raw["optimizer"] = {"kind": "adamax", "lr": 0.01}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_name_must_be_a_slug(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_config(tmp_path, name="Not A Slug")


def test_epochs_and_batch_size_must_be_positive(tmp_path: Path) -> None:
    base = make_config(tmp_path).model_dump()
    for field, value in (("epochs", 0), ("batch_size", 0), ("num_workers", -1), ("seed", -1)):
        raw = dict(base)
        raw[field] = value
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(raw)


def test_config_is_frozen_and_strict(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ValidationError):
        config.epochs = 99  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**config.model_dump(), "unknown": 1})


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    assert make_config(tmp_path).model_dump_json() == make_config(tmp_path).model_dump_json()


# --- fingerprint (the resume identity) ---------------------------------------
def test_fingerprint_ignores_the_resume_block(tmp_path: Path) -> None:
    a = make_config(tmp_path, resume=False)
    b = make_config(tmp_path, resume=True)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_any_other_field(tmp_path: Path) -> None:
    a = make_config(tmp_path, seed=7)
    b = make_config(tmp_path, seed=8)
    assert a.fingerprint() != b.fingerprint()


def test_logging_config_bounds() -> None:
    with pytest.raises(ValidationError):
        LoggingConfig(log_every_n_steps=0)
    with pytest.raises(ValidationError):
        LoggingConfig(backend="tensorboard")  # type: ignore[arg-type]


def test_resume_config_defaults() -> None:
    resume = ResumeConfig()
    assert resume.enabled is False
    assert resume.from_checkpoint == "latest"

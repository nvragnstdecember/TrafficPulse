"""Validated training-experiment configuration (H4A).

Strongly typed, frozen, fail-loud configuration for every future RT-DETR (or
other-model) training run. Nothing here constructs an optimizer, scheduler, or
model — these are *declarations* the H4B loop will realise. Keeping construction
out is what lets this module stay ML-free and the trainer model-agnostic.

Validation split (consistent with H1–H3): field-level bounds raise pydantic
``ValidationError``; cross-field semantic rules raise the typed
:class:`~helmet_rtdetr.errors.InvalidTrainingConfigError`.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from ..errors import InvalidTrainingConfigError
from ..models import NonEmptyStr, Slug, _Model
from .metrics import METRIC_NAME_RE


# --- optimizers (configuration only; construction is H4B) ---------------------
class AdamWConfig(_Model):
    """AdamW declaration (the RT-DETR-family default)."""

    kind: Literal["adamw"] = "adamw"
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)

    @model_validator(mode="after")
    def _betas_in_range(self) -> Self:
        if not all(0.0 <= b < 1.0 for b in self.betas):
            raise InvalidTrainingConfigError(f"adamw betas must be in [0, 1), got {self.betas}")
        return self


class SgdConfig(_Model):
    """SGD declaration (a future alternative; nothing constructs it yet)."""

    kind: Literal["sgd"] = "sgd"
    lr: float = Field(gt=0.0)
    momentum: float = Field(default=0.0, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    nesterov: bool = False

    @model_validator(mode="after")
    def _nesterov_needs_momentum(self) -> Self:
        if self.nesterov and self.momentum <= 0.0:
            raise InvalidTrainingConfigError("nesterov SGD requires momentum > 0")
        return self


OptimizerConfig: TypeAlias = Annotated[AdamWConfig | SgdConfig, Field(discriminator="kind")]


# --- schedulers (configuration only) ------------------------------------------
class CosineSchedulerConfig(_Model):
    """Warmup + cosine decay (the design-spec default)."""

    kind: Literal["cosine"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    min_lr_fraction: float = Field(default=0.01, gt=0.0, le=1.0)


class StepSchedulerConfig(_Model):
    """Step decay every ``step_size`` epochs by factor ``gamma``."""

    kind: Literal["step"] = "step"
    step_size: int = Field(ge=1)
    gamma: float = Field(default=0.1, gt=0.0, le=1.0)


class OneCycleSchedulerConfig(_Model):
    """One-cycle policy declaration."""

    kind: Literal["one_cycle"] = "one_cycle"
    pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)
    final_div_factor: float = Field(default=1e4, ge=1.0)


SchedulerConfig: TypeAlias = Annotated[
    CosineSchedulerConfig | StepSchedulerConfig | OneCycleSchedulerConfig,
    Field(discriminator="kind"),
]


# --- checkpointing / logging / resume -----------------------------------------
class CheckpointPolicy(_Model):
    """When checkpoints are kept: best, latest window, and periodic saves."""

    save_best: bool = False
    best_metric: str | None = None
    best_mode: Literal["min", "max"] = "max"
    keep_last: int = Field(default=1, ge=1)
    every_n_epochs: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _best_requires_metric(self) -> Self:
        if self.save_best:
            if self.best_metric is None:
                raise InvalidTrainingConfigError("save_best requires best_metric to be set")
            if not METRIC_NAME_RE.match(self.best_metric):
                raise InvalidTrainingConfigError(
                    f"best_metric {self.best_metric!r} is not a valid metric name"
                )
        return self


class LoggingConfig(_Model):
    """Structured-logging declaration; the sink backend stays abstract."""

    backend: Literal["jsonl", "memory", "null"] = "jsonl"
    log_every_n_steps: int = Field(default=1, ge=1)


class ResumeConfig(_Model):
    """Whether (and from which checkpoint) an existing run may be resumed."""

    enabled: bool = False
    from_checkpoint: Literal["latest", "best"] = "latest"


# --- the experiment ------------------------------------------------------------
class ExperimentConfig(_Model):
    """One training experiment, fully declared and validated up front."""

    name: Slug
    output_root: NonEmptyStr
    seed: int = Field(default=0, ge=0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    num_workers: int = Field(default=0, ge=0)
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointPolicy = CheckpointPolicy()
    logging: LoggingConfig = LoggingConfig()
    resume: ResumeConfig = ResumeConfig()

    def fingerprint(self) -> str:
        """Identity of everything except the ``resume`` block.

        Used by resume validation: a run may be continued with a different resume
        setting, but never with a silently different experiment definition.
        """

        neutral = self.model_copy(update={"resume": ResumeConfig()})
        return hashlib.sha256(neutral.model_dump_json().encode("utf-8")).hexdigest()

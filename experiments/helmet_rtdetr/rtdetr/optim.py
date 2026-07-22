"""Optimizer + scheduler construction from the H4A declarations (H4B).

The H4A configs *declare*; this module *realises* them as torch objects. No
hyperparameter is hardcoded — every value flows from the validated config.

Schedulers return their stepping granularity alongside the object, because the
three families genuinely differ: cosine and one-cycle are per-**step** schedules
(one-cycle is even *defined* over total steps), while classic step-decay is a
per-**epoch** schedule. The loop honours whichever the config implies rather than
forcing one cadence on all three.
"""

from __future__ import annotations

from typing import Any, Literal

from ..training.config import (
    AdamWConfig,
    CosineSchedulerConfig,
    OneCycleSchedulerConfig,
    OptimizerConfig,
    SchedulerConfig,
    SgdConfig,
    StepSchedulerConfig,
)
from .model import require_torch

SchedulerGranularity = Literal["step", "epoch"]


def build_optimizer(config: OptimizerConfig, parameters: Any) -> Any:
    """Construct the torch optimizer the H4A config declares."""

    torch = require_torch()
    if isinstance(config, AdamWConfig):
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
    assert isinstance(config, SgdConfig)  # the discriminated union's only other arm
    return torch.optim.SGD(
        parameters,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )


def build_scheduler(
    config: SchedulerConfig, optimizer: Any, *, total_steps: int
) -> tuple[Any, SchedulerGranularity]:
    """Construct the LR scheduler + its stepping granularity.

    ``total_steps`` is the whole run's optimizer-step budget (steps-per-epoch x
    epochs); cosine and one-cycle are laid out over it.
    """

    torch = require_torch()
    sched = torch.optim.lr_scheduler

    if isinstance(config, CosineSchedulerConfig):
        base_lr = optimizer.param_groups[0]["lr"]
        cosine_steps = max(1, total_steps - config.warmup_steps)
        cosine = sched.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=base_lr * config.min_lr_fraction
        )
        if config.warmup_steps == 0:
            return cosine, "step"
        warmup = sched.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=config.warmup_steps
        )
        return (
            sched.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
            ),
            "step",
        )
    if isinstance(config, StepSchedulerConfig):
        return sched.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma), "epoch"
    assert isinstance(config, OneCycleSchedulerConfig)
    return (
        sched.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=max(1, total_steps),
            pct_start=config.pct_start,
            final_div_factor=config.final_div_factor,
        ),
        "step",
    )

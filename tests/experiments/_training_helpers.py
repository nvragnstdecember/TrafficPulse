"""Shared builders for the H4A training-infrastructure tests.

Uniquely named (``_training_helpers``) so pytest's prepend import mode never
collides across the tests tree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from helmet_rtdetr.training import (
    AdamWConfig,
    Callback,
    CheckpointPolicy,
    CheckpointRecord,
    CosineSchedulerConfig,
    ExperimentConfig,
    LoggingConfig,
    ResumeConfig,
    TrainingState,
)


def make_config(
    output_root: Path,
    *,
    name: str = "exp-a",
    seed: int = 7,
    epochs: int = 3,
    resume: bool = False,
    save_best: bool = True,
    keep_last: int = 2,
    every_n_epochs: int | None = None,
) -> ExperimentConfig:
    """A fully valid ExperimentConfig for tests (memory logging, no file sink)."""

    return ExperimentConfig(
        name=name,
        output_root=str(output_root),
        seed=seed,
        epochs=epochs,
        batch_size=2,
        optimizer=AdamWConfig(lr=1e-4),
        scheduler=CosineSchedulerConfig(warmup_steps=10),
        checkpoint=CheckpointPolicy(
            save_best=save_best,
            best_metric="val/ap" if save_best else None,
            keep_last=keep_last,
            every_n_epochs=every_n_epochs,
        ),
        logging=LoggingConfig(backend="memory"),
        resume=ResumeConfig(enabled=resume),
    )


def make_clock(
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC), step_seconds: float = 1.0
) -> Callable[[], datetime]:
    """A deterministic clock: each call advances by ``step_seconds``."""

    state = {"now": start}

    def clock() -> datetime:
        current = state["now"]
        state["now"] = current + timedelta(seconds=step_seconds)
        return current

    return clock


class Recorder(Callback):
    """Appends '<tag>:<hook>' to a shared list — pins dispatch order across hooks."""

    def __init__(self, log: list[str], tag: str = "cb") -> None:
        self._log = log
        self._tag = tag

    def on_train_start(self, state: TrainingState) -> None:
        self._log.append(f"{self._tag}:train_start")

    def on_epoch_start(self, state: TrainingState) -> None:
        self._log.append(f"{self._tag}:epoch_start")

    def on_batch_start(self, state: TrainingState) -> None:
        self._log.append(f"{self._tag}:batch_start")

    def on_batch_end(self, state: TrainingState) -> None:
        self._log.append(f"{self._tag}:batch_end")

    def on_epoch_end(self, state: TrainingState, metrics: Mapping[str, float]) -> None:
        self._log.append(f"{self._tag}:epoch_end")

    def on_checkpoint(self, state: TrainingState, record: CheckpointRecord) -> None:
        self._log.append(f"{self._tag}:checkpoint")

    def on_train_end(self, state: TrainingState) -> None:
        self._log.append(f"{self._tag}:train_end")

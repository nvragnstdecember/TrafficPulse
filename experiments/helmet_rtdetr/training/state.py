"""Immutable training-state models (H4A).

The state a training run carries between lifecycle events: completed epochs,
global step, best-metric bookkeeping, elapsed time, and checkpoint history. All
models are frozen + strict and serialise deterministically, so the state inside a
checkpoint file replays byte-identically.

``elapsed_seconds`` is ``None`` unless the trainer was given a clock — time is
never fabricated from a wall-clock the caller did not inject (the same honesty
rule the runtime pipeline follows).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ..models import NonEmptyStr, _Model


class RunPhase(StrEnum):
    """The trainer lifecycle phase recorded on the state."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"


class CheckpointRole(StrEnum):
    """Why a checkpoint exists. One physical checkpoint may hold several roles."""

    LATEST = "latest"
    BEST = "best"
    PERIODIC = "periodic"


class CheckpointRecord(_Model):
    """Metadata for one saved checkpoint (no tensors — H4A is architecture only).

    ``payload_path`` is reserved for the H4B unit that will actually serialise
    model weights next to this metadata; it stays ``None`` here, honestly meaning
    "no weights were written", never a fabricated path.
    """

    checkpoint_id: NonEmptyStr
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    roles: tuple[CheckpointRole, ...]
    filename: NonEmptyStr
    metric_name: str | None = None
    metric_value: float | None = None
    payload_path: str | None = None
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _at_least_one_role(self) -> Self:
        if not self.roles:
            raise ValueError("a checkpoint must have at least one role")
        return self


class TrainingState(_Model):
    """The immutable, serialisable state of one training run.

    ``epoch`` counts **completed** epochs (0-based next-epoch index), so a resumed
    run continues exactly where the last checkpoint left off.
    """

    phase: RunPhase = RunPhase.INITIALIZED
    epoch: int = Field(default=0, ge=0)
    global_step: int = Field(default=0, ge=0)
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    elapsed_seconds: float | None = None
    checkpoint_history: tuple[CheckpointRecord, ...] = ()


class CheckpointFile(_Model):
    """The on-disk shape of one checkpoint: its record plus the state at save time.

    The saved ``state.checkpoint_history`` deliberately contains only *prior*
    records — the record describing this very checkpoint is the sibling ``record``
    field, not an entry in its own history (which would make the file describe
    itself recursively).
    """

    record: CheckpointRecord
    state: TrainingState

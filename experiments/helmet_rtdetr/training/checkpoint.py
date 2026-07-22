"""Checkpoint management: metadata save/load, latest/best, cleanup (H4A).

Manages checkpoint **metadata only** — a :class:`CheckpointFile` (record + state)
as deterministic JSON. Model-weight serialisation is H4B's job; the record's
``payload_path`` slot is where it will point, and cleanup is designed to remove
the payload alongside the metadata when it exists.

Roles and retention
-------------------
Every save is the new **latest**. It additionally becomes **best** when the
policy's metric improved (strictly, per ``best_mode``), and **periodic** when the
completed-epoch count is a multiple of ``every_n_epochs``. Cleanup retains the
union of: the best checkpoint, every periodic checkpoint (they are the durable
archive), and the last ``keep_last`` saves — everything else is deleted. The best
checkpoint is therefore never cleaned up, no matter how old.

An ``index.json`` (deterministic JSON) tracks latest/best/periodic/history so
``latest()``/``best()`` need no directory scanning.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from ..errors import CheckpointError, CheckpointNotFoundError
from ..models import _Model
from .config import CheckpointPolicy
from .state import CheckpointFile, CheckpointRecord, CheckpointRole, TrainingState

_INDEX_FILENAME = "index.json"


class _Index(_Model):
    """The manager's durable pointer state (deterministic JSON)."""

    latest: str | None = None
    best: str | None = None
    best_value: float | None = None
    periodic: tuple[str, ...] = ()
    history: tuple[str, ...] = ()


class CheckpointManager:
    """Saves/loads checkpoint metadata under one directory, per one policy."""

    def __init__(self, directory: Path, policy: CheckpointPolicy) -> None:
        self._dir = directory
        self._policy = policy

    @property
    def directory(self) -> Path:
        return self._dir

    # --- save -----------------------------------------------------------------
    def save(
        self,
        state: TrainingState,
        *,
        metrics: dict[str, float] | None = None,
        created_at: datetime | None = None,
    ) -> CheckpointRecord:
        """Persist one checkpoint's metadata; return its record (with roles).

        ``metrics`` are this epoch's values; required when the policy tracks a
        best metric. Raises :class:`CheckpointError` if the tracked metric is
        absent or non-finite — a best-checkpoint decision is never made on
        missing or poisoned evidence.
        """

        index = self._read_index()
        roles: list[CheckpointRole] = [CheckpointRole.LATEST]

        metric_name: str | None = None
        metric_value: float | None = None
        if self._policy.save_best:
            metric_name = self._policy.best_metric
            assert metric_name is not None  # enforced by CheckpointPolicy validation
            if metrics is None or metric_name not in metrics:
                raise CheckpointError(
                    f"policy tracks best {metric_name!r} but it is absent from the "
                    f"epoch metrics ({sorted(metrics) if metrics else []})"
                )
            metric_value = metrics[metric_name]
            if not math.isfinite(metric_value):
                raise CheckpointError(
                    f"best metric {metric_name!r} is not finite: {metric_value!r}"
                )
            if self._improved(metric_value, index.best_value):
                roles.append(CheckpointRole.BEST)

        every = self._policy.every_n_epochs
        if every is not None and state.epoch > 0 and state.epoch % every == 0:
            roles.append(CheckpointRole.PERIODIC)

        checkpoint_id = f"e{state.epoch:04d}-s{state.global_step:08d}"
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            epoch=state.epoch,
            global_step=state.global_step,
            roles=tuple(roles),
            filename=self._filename(checkpoint_id),
            metric_name=metric_name,
            metric_value=metric_value,
            created_at=created_at,
        )

        self._dir.mkdir(parents=True, exist_ok=True)
        payload = CheckpointFile(record=record, state=state)
        (self._dir / record.filename).write_text(
            payload.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        history = index.history if checkpoint_id in index.history else (
            *index.history,
            checkpoint_id,
        )
        index = index.model_copy(
            update={
                "latest": checkpoint_id,
                "best": checkpoint_id if CheckpointRole.BEST in roles else index.best,
                "best_value": (
                    metric_value if CheckpointRole.BEST in roles else index.best_value
                ),
                "periodic": (
                    (*index.periodic, checkpoint_id)
                    if CheckpointRole.PERIODIC in roles and checkpoint_id not in index.periodic
                    else index.periodic
                ),
                "history": history,
            }
        )
        index = self._cleanup(index)
        self._write_index(index)
        return record

    def _improved(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        return value > best if self._policy.best_mode == "max" else value < best

    # --- load -----------------------------------------------------------------
    def load(self, checkpoint_id: str) -> CheckpointFile:
        """Load one checkpoint's metadata + state by id."""

        path = self._dir / self._filename(checkpoint_id)
        if not path.is_file():
            raise CheckpointNotFoundError(f"no checkpoint {checkpoint_id!r} under {self._dir}")
        try:
            return CheckpointFile.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise CheckpointError(f"checkpoint {path} is corrupt: {exc}") from exc

    def latest(self) -> CheckpointFile:
        """The most recently saved checkpoint."""

        index = self._read_index()
        if index.latest is None:
            raise CheckpointNotFoundError(f"no checkpoints have been saved under {self._dir}")
        return self.load(index.latest)

    def best(self) -> CheckpointFile:
        """The best checkpoint per the policy's metric."""

        index = self._read_index()
        if index.best is None:
            raise CheckpointNotFoundError(
                f"no best checkpoint recorded under {self._dir} "
                "(save_best disabled, or nothing saved yet)"
            )
        return self.load(index.best)

    def checkpoint_ids(self) -> tuple[str, ...]:
        """The retained checkpoint ids, oldest first."""

        return self._read_index().history

    # --- internals --------------------------------------------------------------
    @staticmethod
    def _filename(checkpoint_id: str) -> str:
        return f"ckpt-{checkpoint_id}.json"

    def _cleanup(self, index: _Index) -> _Index:
        keep: set[str] = set(index.history[-self._policy.keep_last :])
        keep.update(index.periodic)
        if index.best is not None:
            keep.add(index.best)
        for checkpoint_id in index.history:
            if checkpoint_id not in keep:
                (self._dir / self._filename(checkpoint_id)).unlink(missing_ok=True)
        return index.model_copy(
            update={"history": tuple(c for c in index.history if c in keep)}
        )

    def _read_index(self) -> _Index:
        path = self._dir / _INDEX_FILENAME
        if not path.is_file():
            return _Index()
        try:
            return _Index.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise CheckpointError(f"checkpoint index {path} is corrupt: {exc}") from exc

    def _write_index(self, index: _Index) -> None:
        (self._dir / _INDEX_FILENAME).write_text(
            index.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

"""Model-agnostic training orchestration (H4A).

The :class:`Trainer` owns everything *around* a training loop — lifecycle,
immutable state, callbacks, checkpointing, structured logging, metrics, and
deterministic setup — while knowing nothing about RT-DETR, torch, datasets, or
batches. It deliberately has **no** training loop: the H4B unit drives these
lifecycle methods from its loop (``begin`` → per epoch ``begin_epoch`` /
``record_batch``... / ``end_epoch`` → ``end``), supplying the model-specific work
between the calls. Any future model plugs in the same way.

Lifecycle guards are strict: every method validates the phase it is called in and
raises :class:`~helmet_rtdetr.errors.TrainerStateError` on misuse, so a broken
loop fails loudly at the first out-of-order call rather than corrupting state.

Determinism and time
--------------------
Seeding is applied at ``begin()`` from the config's single seed (see
``seeding.py``). Wall-clock time enters **only** through the injectable ``clock``;
without one, events carry no timestamp and ``elapsed_seconds`` stays ``None`` —
the same never-fabricate-time rule the runtime pipeline follows.

Resume
------
``begin()`` auto-resumes when the run directory is already initialised and the
config's resume block allows it: the stored config must match the current one
(fingerprint over everything except the resume block), a finished run refuses to
resume, state restores from the latest (or best) checkpoint, and the metrics
store reloads from disk. An initialised run with resume *disabled* raises
:class:`~helmet_rtdetr.errors.DuplicateExperimentError` — a name collision is
never silently overwritten.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..errors import (
    CheckpointNotFoundError,
    DuplicateExperimentError,
    ResumeError,
    TrainerStateError,
)
from .callbacks import Callback, CallbackList
from .checkpoint import CheckpointManager
from .config import ExperimentConfig
from .events import EventData, LogEvent, LogEventKind, LogSink, build_sink
from .metrics import MetricsStore
from .run_layout import RunLayout
from .seeding import apply_seed_plan, derive_seed_plan
from .state import CheckpointRole, RunPhase, TrainingState


class Trainer:
    """Lifecycle orchestrator for one experiment (see module docstring)."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        callbacks: Sequence[Callback] = (),
        sink: LogSink | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._callbacks = CallbackList(callbacks)
        self._layout = RunLayout(Path(config.output_root), config.name)
        self._manager = CheckpointManager(self._layout.checkpoints, config.checkpoint)
        self._metrics = MetricsStore()
        self._sink: LogSink = (
            sink if sink is not None else build_sink(config.logging, self._layout.logs)
        )
        self._clock = clock
        self._sequence = 0
        self._state = TrainingState()
        self._in_epoch = False
        self._current_epoch: int | None = None
        self._resumed = False
        self._t0: datetime | None = None
        self._elapsed_base = 0.0

    # --- read-only surface ------------------------------------------------------
    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def metrics(self) -> MetricsStore:
        return self._metrics

    @property
    def layout(self) -> RunLayout:
        return self._layout

    @property
    def checkpoints(self) -> CheckpointManager:
        return self._manager

    @property
    def resumed(self) -> bool:
        """Whether ``begin()`` continued an existing run rather than starting fresh."""

        return self._resumed

    # --- lifecycle ---------------------------------------------------------------
    def begin(self) -> TrainingState:
        """Initialise (or resume) the run: layout, seeds, config, callbacks."""

        if self._state.phase is not RunPhase.INITIALIZED:
            raise TrainerStateError(
                f"begin() must be the first lifecycle call (phase={self._state.phase.value})"
            )
        initialized = self._layout.is_initialized()
        if initialized and not self._config.resume.enabled:
            raise DuplicateExperimentError(
                f"experiment {self._config.name!r} already exists at {self._layout.run_dir}; "
                "enable resume or choose a new experiment name"
            )
        apply_seed_plan(derive_seed_plan(self._config.seed))
        self._layout.create()
        if initialized:
            self._begin_resumed()
        else:
            self._begin_fresh()
        self._t0 = self._clock() if self._clock is not None else None
        self._elapsed_base = self._state.elapsed_seconds or 0.0
        self._callbacks.on_train_start(self._state)
        return self._state

    def begin_epoch(self) -> TrainingState:
        """Open the next epoch (fails once the configured epoch budget is spent)."""

        self._require(in_epoch=False)
        if self._state.epoch >= self._config.epochs:
            raise TrainerStateError(
                f"all {self._config.epochs} configured epochs are already complete"
            )
        self._current_epoch = self._state.epoch
        self._in_epoch = True
        self._emit(LogEventKind.EPOCH_START, {"epoch": self._current_epoch})
        self._callbacks.on_epoch_start(self._state)
        return self._state

    def record_batch(self) -> TrainingState:
        """Advance the global step by one batch (fires the batch hooks)."""

        self._require(in_epoch=True)
        self._callbacks.on_batch_start(self._state)
        self._state = self._state.model_copy(
            update={"global_step": self._state.global_step + 1}
        )
        self._callbacks.on_batch_end(self._state)
        return self._state

    def end_epoch(self, metrics: Mapping[str, float]) -> TrainingState:
        """Close the epoch: record metrics, checkpoint per policy, fire hooks."""

        self._require(in_epoch=True)
        assert self._current_epoch is not None
        completed = self._current_epoch
        for name in sorted(metrics):
            self._metrics.record(
                name, metrics[name], epoch=completed, step=self._state.global_step
            )

        state = self._state.model_copy(
            update={"epoch": completed + 1, "elapsed_seconds": self._elapsed()}
        )
        record = self._manager.save(
            state,
            metrics=dict(metrics),
            created_at=self._clock() if self._clock is not None else None,
        )
        update: dict[str, Any] = {
            "checkpoint_history": (*state.checkpoint_history, record)
        }
        if CheckpointRole.BEST in record.roles:
            update["best_metric_name"] = record.metric_name
            update["best_metric_value"] = record.metric_value
        self._state = state.model_copy(update=update)

        # Persist metrics every epoch so an interrupted run resumes with history.
        self._metrics.save(self._layout.metrics_path)
        self._in_epoch = False
        self._current_epoch = None
        self._emit(
            LogEventKind.EPOCH_END,
            {"epoch": completed, "global_step": self._state.global_step},
        )
        self._emit(
            LogEventKind.CHECKPOINT,
            {
                "checkpoint_id": record.checkpoint_id,
                "roles": ",".join(role.value for role in record.roles),
            },
        )
        self._callbacks.on_epoch_end(self._state, dict(metrics))
        self._callbacks.on_checkpoint(self._state, record)
        return self._state

    def end(self) -> TrainingState:
        """Finish the run: final state + metrics persisted, finish event, hooks."""

        self._require(in_epoch=False)
        self._state = self._state.model_copy(
            update={"phase": RunPhase.FINISHED, "elapsed_seconds": self._elapsed()}
        )
        self._layout.state_path.write_text(
            self._state.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        self._metrics.save(self._layout.metrics_path)
        self._emit(
            LogEventKind.EXPERIMENT_FINISH,
            {
                "epochs_completed": self._state.epoch,
                "global_step": self._state.global_step,
            },
        )
        self._callbacks.on_train_end(self._state)
        return self._state

    # --- internals ----------------------------------------------------------------
    def _begin_fresh(self) -> None:
        self._layout.config_path.write_text(
            self._config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        self._layout.seed_plan_path.write_text(
            derive_seed_plan(self._config.seed).model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        self._state = TrainingState(phase=RunPhase.RUNNING)
        self._emit(
            LogEventKind.EXPERIMENT_START,
            {
                "name": self._config.name,
                "seed": self._config.seed,
                "epochs": self._config.epochs,
            },
        )

    def _begin_resumed(self) -> None:
        stored = self._load_stored_config()
        if stored.fingerprint() != self._config.fingerprint():
            raise ResumeError(
                "stored config.json does not match the current configuration; only "
                "the resume block may differ between the original run and a resume"
            )
        if self._layout.state_path.is_file():
            try:
                final = TrainingState.model_validate_json(
                    self._layout.state_path.read_text(encoding="utf-8")
                )
            except ValidationError as exc:
                raise ResumeError(f"stored state.json is unreadable: {exc}") from exc
            if final.phase is RunPhase.FINISHED:
                raise ResumeError(
                    f"experiment {self._config.name!r} already finished; refusing to "
                    "resume past end()"
                )
        try:
            checkpoint = (
                self._manager.best()
                if self._config.resume.from_checkpoint == "best"
                else self._manager.latest()
            )
            self._state = checkpoint.state.model_copy(update={"phase": RunPhase.RUNNING})
        except CheckpointNotFoundError:
            # Initialised but never checkpointed: resume from the very beginning.
            self._state = TrainingState(phase=RunPhase.RUNNING)
        self._metrics = MetricsStore.load_or_new(self._layout.metrics_path)
        self._resumed = True
        self._emit(
            LogEventKind.RESUME,
            {"epoch": self._state.epoch, "global_step": self._state.global_step},
        )

    def _load_stored_config(self) -> ExperimentConfig:
        try:
            return ExperimentConfig.model_validate_json(
                self._layout.config_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ResumeError(f"stored config.json is unreadable: {exc}") from exc

    def _require(self, *, in_epoch: bool) -> None:
        if self._state.phase is not RunPhase.RUNNING:
            raise TrainerStateError(
                f"lifecycle call requires a running experiment "
                f"(phase={self._state.phase.value}); call begin() first"
            )
        if self._in_epoch != in_epoch:
            where = "inside" if in_epoch else "outside"
            raise TrainerStateError(f"this call is only valid {where} an epoch")

    def _elapsed(self) -> float | None:
        if self._clock is None or self._t0 is None:
            return self._state.elapsed_seconds
        return self._elapsed_base + (self._clock() - self._t0).total_seconds()

    def _emit(self, kind: LogEventKind, data: EventData) -> None:
        event = LogEvent(
            sequence=self._sequence,
            kind=kind,
            at=self._clock() if self._clock is not None else None,
            data=data,
        )
        self._sink.emit(event)
        self._sequence += 1

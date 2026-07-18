"""The training callback system (H4A).

:class:`Callback` is the extension seam future units hang behaviour on (LR
logging, early-stop counters, progress display) without the trainer knowing any
of it. Every hook is a no-op by default, so a callback overrides only what it
needs. :class:`CallbackList` dispatches to callbacks **in registration order** —
deterministic, and tested.

Callbacks receive immutable :class:`TrainingState` snapshots (plus the epoch
metrics / checkpoint record where natural). They observe; they cannot mutate the
trainer through what they are handed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .state import CheckpointRecord, TrainingState


class Callback:
    """Base callback: override any subset of the lifecycle hooks."""

    def on_train_start(self, state: TrainingState) -> None:
        return

    def on_epoch_start(self, state: TrainingState) -> None:
        return

    def on_batch_start(self, state: TrainingState) -> None:
        return

    def on_batch_end(self, state: TrainingState) -> None:
        return

    def on_epoch_end(self, state: TrainingState, metrics: Mapping[str, float]) -> None:
        return

    def on_checkpoint(self, state: TrainingState, record: CheckpointRecord) -> None:
        return

    def on_train_end(self, state: TrainingState) -> None:
        return


class CallbackList:
    """Dispatches each hook to every callback, in registration order."""

    def __init__(self, callbacks: Sequence[Callback] = ()) -> None:
        self._callbacks: tuple[Callback, ...] = tuple(callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def on_train_start(self, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_train_start(state)

    def on_epoch_start(self, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_epoch_start(state)

    def on_batch_start(self, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_batch_start(state)

    def on_batch_end(self, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_batch_end(state)

    def on_epoch_end(self, state: TrainingState, metrics: Mapping[str, float]) -> None:
        for callback in self._callbacks:
            callback.on_epoch_end(state, metrics)

    def on_checkpoint(self, state: TrainingState, record: CheckpointRecord) -> None:
        for callback in self._callbacks:
            callback.on_checkpoint(state, record)

    def on_train_end(self, state: TrainingState) -> None:
        for callback in self._callbacks:
            callback.on_train_end(state)

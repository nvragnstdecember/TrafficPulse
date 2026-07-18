"""Structured training-event logging with an abstract sink (H4A).

The trainer emits typed :class:`LogEvent` values for the lifecycle moments the
spec names (experiment start/finish, epoch start/end, checkpoint, resume);
*where* they go is a :class:`LogSink` implementation — JSONL file, in-memory (for
tests), or null. No TensorBoard, no logging framework: the backend stays a small
abstract seam a later unit can extend.

Events carry a monotonic ``sequence`` so ordering never depends on timestamps,
and ``at`` is ``None`` unless the trainer was given a clock — deterministic by
default, honest about time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from ..models import _Model
from .config import LoggingConfig

EventData = dict[str, str | int | float | None]


class LogEventKind(StrEnum):
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_FINISH = "experiment_finish"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    CHECKPOINT = "checkpoint"
    RESUME = "resume"


class LogEvent(_Model):
    """One structured lifecycle event."""

    sequence: int = Field(ge=0)
    kind: LogEventKind
    at: datetime | None = None
    data: EventData = Field(default_factory=dict)


class LogSink(ABC):
    """Where structured events go. Implementations must not reorder or drop."""

    @abstractmethod
    def emit(self, event: LogEvent) -> None:
        """Record one event."""


class MemoryLogSink(LogSink):
    """Keeps events in memory — the test double."""

    def __init__(self) -> None:
        self.events: list[LogEvent] = []

    def emit(self, event: LogEvent) -> None:
        self.events.append(event)


class JsonlLogSink(LogSink):
    """Appends one JSON line per event to a file (parents created lazily)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: LogEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")


class NullLogSink(LogSink):
    """Discards everything (explicitly opted into via config)."""

    def emit(self, event: LogEvent) -> None:
        return


def build_sink(config: LoggingConfig, logs_dir: Path) -> LogSink:
    """Construct the sink the config declares (jsonl → ``logs_dir/events.jsonl``)."""

    if config.backend == "jsonl":
        return JsonlLogSink(logs_dir / "events.jsonl")
    if config.backend == "memory":
        return MemoryLogSink()
    return NullLogSink()

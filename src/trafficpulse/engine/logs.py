"""Deterministic structured logging for the engine (H6).

Structured events, never ``print``: every notable engine action emits one
:class:`EngineLogEvent` -- a frozen, JSON-serialisable record with a monotonic
sequence number, a closed event vocabulary, and flat scalar data -- through an
injected :class:`EngineLogSink`.

Determinism (the H4A/H6 time-honesty rule)
------------------------------------------
Sequence numbers are engine-assigned and gap-free, so the event *order* is a
pure function of the run. Wall-clock time enters only through the engine's
injectable ``clock``: without one, ``at`` stays ``None`` -- a timestamp is never
fabricated -- and the emitted stream (and a ``JsonlLogSink`` file) is
byte-identical across replays of the same input.

Sinks
-----
:class:`MemoryLogSink` buffers events for tests/inspection; :class:`JsonlLogSink`
appends one JSON object per line (UTF-8) for operational tailing;
:class:`NullLogSink` (the default) drops everything. All are stdlib+pydantic
only.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

EventData: TypeAlias = dict[str, str | int | float]


class EngineLogEventKind(StrEnum):
    """The closed vocabulary of engine log events."""

    ENGINE_START = "engine_start"
    SOURCE_OPENED = "source_opened"
    FRAME_DROPPED = "frame_dropped"
    BATCH_PROCESSED = "batch_processed"
    FINALIZED = "finalized"
    PERSISTED = "persisted"
    ENGINE_RESET = "engine_reset"
    ENGINE_STOP = "engine_stop"


class EngineLogEvent(BaseModel):
    """One structured, immutable engine log record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    kind: EngineLogEventKind
    at: datetime | None = None  # only from an injected clock; never fabricated
    data: EventData = Field(default_factory=dict)


class EngineLogSink(Protocol):
    """Where structured engine events go (injected; never a global logger)."""

    def emit(self, event: EngineLogEvent) -> None:
        """Record one event. Must not mutate it; must tolerate repeated calls."""
        ...


class NullLogSink:
    """Discards every event (the default sink)."""

    def emit(self, event: EngineLogEvent) -> None:
        return None


class MemoryLogSink:
    """Buffers events in memory, in emission order (tests/inspection)."""

    def __init__(self) -> None:
        self._events: list[EngineLogEvent] = []

    def emit(self, event: EngineLogEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[EngineLogEvent, ...]:
        return tuple(self._events)


class JsonlLogSink:
    """Appends one JSON object per event line to ``path`` (parents created)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: EngineLogEvent) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

"""Structured engine logging: events, sinks, time honesty (H6)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trafficpulse.engine import (
    EngineLogEvent,
    EngineLogEventKind,
    JsonlLogSink,
    MemoryLogSink,
    NullLogSink,
)


def _event(sequence: int, *, at: datetime | None = None) -> EngineLogEvent:
    return EngineLogEvent(
        sequence=sequence,
        kind=EngineLogEventKind.BATCH_PROCESSED,
        at=at,
        data={"frames": sequence},
    )


# --- the event model ---------------------------------------------------------------
def test_events_are_frozen_and_strict() -> None:
    event = _event(0)
    with pytest.raises(ValidationError):
        event.sequence = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        EngineLogEvent(sequence=0, kind=EngineLogEventKind.FINALIZED, extra=1)  # type: ignore[call-arg]


def test_timestamp_defaults_to_honest_none() -> None:
    assert _event(0).at is None
    stamped = _event(1, at=datetime(2026, 7, 21, tzinfo=UTC))
    assert stamped.at is not None


# --- sinks ---------------------------------------------------------------------------
def test_memory_sink_preserves_emission_order() -> None:
    sink = MemoryLogSink()
    for sequence in range(3):
        sink.emit(_event(sequence))
    assert [event.sequence for event in sink.events] == [0, 1, 2]


def test_null_sink_discards() -> None:
    NullLogSink().emit(_event(0))  # nothing observable, nothing raised


def test_jsonl_sink_appends_one_line_per_event(tmp_path: Path) -> None:
    sink = JsonlLogSink(tmp_path / "logs" / "engine.jsonl")
    sink.emit(_event(0))
    sink.emit(_event(1))
    lines = sink.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [EngineLogEvent.model_validate_json(line).sequence for line in lines] == [0, 1]


def test_jsonl_sink_is_byte_deterministic_without_a_clock(tmp_path: Path) -> None:
    first = JsonlLogSink(tmp_path / "a.jsonl")
    second = JsonlLogSink(tmp_path / "b.jsonl")
    for sequence in range(2):
        first.emit(_event(sequence))
        second.emit(_event(sequence))
    assert first.path.read_bytes() == second.path.read_bytes()

"""Metrics storage + structured event sinks (H4A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helmet_rtdetr.errors import (
    InvalidMetricNameError,
    InvalidMetricValueError,
    MetricNotFoundError,
)
from helmet_rtdetr.training import (
    JsonlLogSink,
    LogEvent,
    LogEventKind,
    LoggingConfig,
    MemoryLogSink,
    MetricsStore,
    NullLogSink,
    build_sink,
)


# --- MetricsStore ------------------------------------------------------------
def test_record_and_history() -> None:
    store = MetricsStore()
    store.record("train/loss", 1.5, epoch=0, step=1)
    store.record("train/loss", 1.2, epoch=0, step=2)

    history = store.history("train/loss")
    assert [p.value for p in history] == [1.5, 1.2]
    assert store.latest("train/loss").step == 2


def test_names_are_sorted() -> None:
    store = MetricsStore()
    store.record("val/ap", 0.5, epoch=0, step=1)
    store.record("train/loss", 1.0, epoch=0, step=1)
    assert store.names() == ("train/loss", "val/ap")


def test_epoch_summary_takes_the_last_value_per_epoch() -> None:
    store = MetricsStore()
    store.record("train/loss", 2.0, epoch=0, step=1)
    store.record("train/loss", 1.5, epoch=0, step=2)
    store.record("train/loss", 1.0, epoch=1, step=3)
    store.record("val/ap", 0.4, epoch=1, step=3)

    assert store.epoch_summary(0) == {"train/loss": 1.5}
    assert store.epoch_summary(1) == {"train/loss": 1.0, "val/ap": 0.4}
    assert store.epoch_summary(9) == {}


def test_invalid_metric_name_is_rejected() -> None:
    store = MetricsStore()
    with pytest.raises(InvalidMetricNameError):
        store.record("Val Loss", 1.0, epoch=0, step=1)


def test_nan_and_inf_values_are_rejected() -> None:
    store = MetricsStore()
    with pytest.raises(InvalidMetricValueError):
        store.record("train/loss", float("nan"), epoch=0, step=1)
    with pytest.raises(InvalidMetricValueError):
        store.record("train/loss", float("inf"), epoch=0, step=1)


def test_unknown_metric_raises() -> None:
    with pytest.raises(MetricNotFoundError, match="never"):
        MetricsStore().history("never/recorded")
    with pytest.raises(MetricNotFoundError):
        MetricsStore().latest("never/recorded")


def test_dump_is_deterministic_across_recording_order() -> None:
    a = MetricsStore()
    a.record("alpha", 1.0, epoch=0, step=1)
    a.record("beta", 2.0, epoch=0, step=1)
    b = MetricsStore()
    b.record("beta", 2.0, epoch=0, step=1)
    b.record("alpha", 1.0, epoch=0, step=1)
    assert a.to_json() == b.to_json()  # series sorted by name


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = MetricsStore()
    store.record("train/loss", 1.5, epoch=0, step=1)
    store.record("val/ap", 0.6, epoch=0, step=1)
    path = store.save(tmp_path / "metrics" / "metrics.json")

    reloaded = MetricsStore.load_or_new(path)
    assert reloaded.to_json() == store.to_json()


def test_load_or_new_returns_empty_for_missing_file(tmp_path: Path) -> None:
    store = MetricsStore.load_or_new(tmp_path / "absent.json")
    assert store.names() == ()


# --- event sinks -------------------------------------------------------------
def _event(sequence: int) -> LogEvent:
    return LogEvent(
        sequence=sequence, kind=LogEventKind.EPOCH_START, data={"epoch": sequence}
    )


def test_memory_sink_preserves_order() -> None:
    sink = MemoryLogSink()
    sink.emit(_event(0))
    sink.emit(_event(1))
    assert [e.sequence for e in sink.events] == [0, 1]


def test_jsonl_sink_appends_parseable_lines(tmp_path: Path) -> None:
    sink = JsonlLogSink(tmp_path / "logs" / "events.jsonl")
    sink.emit(_event(0))
    sink.emit(_event(1))

    lines = sink.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["sequence"] for p in parsed] == [0, 1]
    assert parsed[0]["kind"] == "epoch_start"


def test_null_sink_discards() -> None:
    NullLogSink().emit(_event(0))  # must not raise


def test_build_sink_maps_backends(tmp_path: Path) -> None:
    assert isinstance(build_sink(LoggingConfig(backend="memory"), tmp_path), MemoryLogSink)
    assert isinstance(build_sink(LoggingConfig(backend="null"), tmp_path), NullLogSink)
    jsonl = build_sink(LoggingConfig(backend="jsonl"), tmp_path)
    assert isinstance(jsonl, JsonlLogSink)
    assert jsonl.path == tmp_path / "events.jsonl"


def test_event_timestamps_default_to_none() -> None:
    """Deterministic by default: no wall-clock unless a clock was injected."""

    assert _event(0).at is None

"""Metrics recorder: deterministic counters, probe-gated measurements (H6)."""

from __future__ import annotations

import importlib.util

import pytest
from _engine_helpers import scripted_perf

from trafficpulse.engine import (
    EngineMetrics,
    LatencyKind,
    MetricsRecorder,
    torch_cuda_memory_probe,
)


# --- counters ---------------------------------------------------------------------
def test_counters_start_at_zero_and_accumulate() -> None:
    recorder = MetricsRecorder()
    snapshot = recorder.snapshot()
    assert snapshot.frames_read == 0
    assert snapshot.frames_processed == 0
    recorder.frames_read += 3
    recorder.frames_processed += 2
    recorder.detections += 5
    recorder.observe_queue_depth(4)
    recorder.observe_queue_depth(2)
    snapshot = recorder.snapshot()
    assert (snapshot.frames_read, snapshot.frames_processed) == (3, 2)
    assert snapshot.detections == 5
    assert snapshot.queue_peak == 4


def test_reset_zeroes_everything() -> None:
    recorder = MetricsRecorder(perf=scripted_perf())
    recorder.frames_processed = 7
    recorder.observe_latency(LatencyKind.FRAME, 0.5)
    recorder.reset()
    snapshot = recorder.snapshot()
    assert snapshot.frames_processed == 0
    assert snapshot.latencies == {}


# --- media-time FPS (deterministic) -------------------------------------------------
def test_media_fps_derives_from_pts_span() -> None:
    recorder = MetricsRecorder()
    for index in range(4):
        recorder.frames_processed += 1
        recorder.observe_media_timestamp(index * 0.1)
    assert recorder.snapshot().media_fps == pytest.approx(10.0)


def test_media_fps_is_none_below_two_frames() -> None:
    recorder = MetricsRecorder()
    recorder.frames_processed = 1
    recorder.observe_media_timestamp(0.0)
    assert recorder.snapshot().media_fps is None


# --- probe-gated measurements --------------------------------------------------------
def test_no_probes_means_no_measurements() -> None:
    recorder = MetricsRecorder()
    recorder.run_started()
    recorder.frames_processed = 5
    recorder.sample_resources()
    recorder.run_ended()
    snapshot = recorder.snapshot()
    assert snapshot.wall_fps is None
    assert snapshot.memory_bytes_current is None
    assert snapshot.memory_bytes_peak is None
    assert snapshot.gpu_memory_bytes_current is None


def test_wall_fps_from_the_perf_probe() -> None:
    recorder = MetricsRecorder(perf=scripted_perf(2.0))
    recorder.run_started()  # perf -> 2.0
    recorder.frames_processed = 8
    recorder.run_ended()  # perf -> 4.0
    assert recorder.snapshot().wall_fps == pytest.approx(8 / 2.0)


def test_latency_aggregation() -> None:
    recorder = MetricsRecorder(perf=scripted_perf())
    recorder.observe_latency(LatencyKind.INFERENCE, 0.2)
    recorder.observe_latency(LatencyKind.INFERENCE, 0.4)
    recorder.observe_latency(LatencyKind.TRACKING, 0.1)
    latencies = recorder.snapshot().latencies
    inference = latencies[LatencyKind.INFERENCE.value]
    assert inference.count == 2
    assert inference.total_seconds == pytest.approx(0.6)
    assert inference.min_seconds == pytest.approx(0.2)
    assert inference.max_seconds == pytest.approx(0.4)
    assert inference.mean_seconds == pytest.approx(0.3)
    assert list(latencies) == sorted(latencies)  # deterministic key order


def test_memory_and_gpu_probes_track_current_and_peak() -> None:
    memory_values = iter([100, 300, 200])
    gpu_values = iter([10, 5])
    recorder = MetricsRecorder(
        memory_probe=lambda: next(memory_values), gpu_probe=lambda: next(gpu_values)
    )
    recorder.sample_resources()
    recorder.sample_resources()
    snapshot = recorder.snapshot()
    assert snapshot.memory_bytes_current == 300
    assert snapshot.memory_bytes_peak == 300
    assert snapshot.gpu_memory_bytes_current == 5
    assert snapshot.gpu_memory_bytes_peak == 10


def test_snapshot_is_frozen() -> None:
    snapshot = MetricsRecorder().snapshot()
    assert isinstance(snapshot, EngineMetrics)
    with pytest.raises(Exception):  # pydantic frozen-instance error  # noqa: B017
        snapshot.frames_read = 1  # type: ignore[misc]


# --- optional torch probe --------------------------------------------------------------
def test_torch_cuda_probe_degrades_to_none_without_cuda() -> None:
    if importlib.util.find_spec("torch") is None:
        assert torch_cuda_memory_probe() is None  # no torch: honest absence
        return
    import torch

    probe = torch_cuda_memory_probe()
    if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
        assert probe is not None and probe() >= 0
    else:
        assert probe is None  # torch present, no CUDA: honest absence

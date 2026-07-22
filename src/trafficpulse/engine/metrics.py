"""Runtime metrics for the engine: counters, latencies, resource probes (H6).

Two strictly separated kinds of measurement:

* **Counters** (frames read/admitted/skipped/dropped/processed, detections,
  track states, events, batches, queue high-water) are pure functions of the
  frame stream and configuration -- fully deterministic, always populated, and
  asserted exactly in tests.
* **Environmental measurements** (wall-clock FPS, inference/tracking latency,
  memory, GPU utilization) depend on the machine. They are taken **only**
  through injected probes -- a ``perf`` monotonic counter, a ``memory_probe``,
  a ``gpu_probe`` -- and stay ``None`` when no probe is injected. Nothing is
  fabricated (the repo-wide time-honesty rule), and the deterministic replay
  tests simply run without probes.

``media_fps`` is the exception that stays deterministic: it is derived from the
processed frames' **PTS media time** (frames / media span), the same media-time
discipline the ingestion layer establishes, so it is replayable.

Probes
------
``perf`` is any ``() -> float`` monotonic-seconds callable (``time.perf_counter``
in production, a scripted fake in tests). ``memory_probe`` returns currently
used bytes (operators may inject e.g. a psutil RSS reader; none is bundled --
psutil is not a project dependency). :func:`torch_cuda_memory_probe` builds a
GPU probe over the optional ``rtdetr`` extra's torch, lazily; it returns
``None`` when torch or CUDA is unavailable, so callers can pass its result
straight through.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class LatencyKind(StrEnum):
    """The instrumented engine stages."""

    INFERENCE = "inference"
    TRACKING = "tracking"
    FRAME = "frame"  # whole per-frame pipeline step (detect+track+observe)


class LatencySummary(BaseModel):
    """Aggregated latency of one stage (present only when ``perf`` was injected)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    total_seconds: float
    min_seconds: float
    max_seconds: float

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0


class EngineMetrics(BaseModel):
    """One immutable snapshot of an engine run's metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # deterministic counters
    frames_read: int
    frames_skipped_stride: int
    frames_skipped_fps: int
    frames_dropped_backpressure: int
    frames_admitted: int
    frames_processed: int
    batches_processed: int
    detections: int
    track_states: int
    events_confirmed: int
    queue_peak: int
    # deterministic media-time rate (PTS-derived; None until 2+ frames processed)
    media_fps: float | None
    # environmental (None without the matching injected probe)
    wall_fps: float | None
    latencies: dict[str, LatencySummary]
    memory_bytes_current: int | None
    memory_bytes_peak: int | None
    gpu_memory_bytes_current: int | None
    gpu_memory_bytes_peak: int | None


@dataclass
class _LatencyAccumulator:
    count: int = 0
    total: float = 0.0
    minimum: float = field(default=float("inf"))
    maximum: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.minimum = min(self.minimum, seconds)
        self.maximum = max(self.maximum, seconds)

    def summary(self) -> LatencySummary:
        return LatencySummary(
            count=self.count,
            total_seconds=self.total,
            min_seconds=self.minimum,
            max_seconds=self.maximum,
        )


class MetricsRecorder:
    """Mutable accumulator behind :class:`EngineMetrics` snapshots.

    Counter methods are unconditional; measurement methods are no-ops without
    their probe, so instrumented code never branches on probe presence.
    """

    def __init__(
        self,
        *,
        perf: Callable[[], float] | None = None,
        memory_probe: Callable[[], int] | None = None,
        gpu_probe: Callable[[], int] | None = None,
    ) -> None:
        self._perf = perf
        self._memory_probe = memory_probe
        self._gpu_probe = gpu_probe
        self.reset()

    def reset(self) -> None:
        """Zero every counter and measurement for a fresh run."""

        self.frames_read = 0
        self.frames_skipped_stride = 0
        self.frames_skipped_fps = 0
        self.frames_dropped_backpressure = 0
        self.frames_admitted = 0
        self.frames_processed = 0
        self.batches_processed = 0
        self.detections = 0
        self.track_states = 0
        self.events_confirmed = 0
        self.queue_peak = 0
        self._latencies: dict[LatencyKind, _LatencyAccumulator] = {}
        self._first_media_ts: float | None = None
        self._last_media_ts: float | None = None
        self._run_started_perf: float | None = None
        self._run_ended_perf: float | None = None
        self._memory_current: int | None = None
        self._memory_peak: int | None = None
        self._gpu_current: int | None = None
        self._gpu_peak: int | None = None

    # --- timing ------------------------------------------------------------------
    @property
    def perf(self) -> Callable[[], float] | None:
        return self._perf

    def run_started(self) -> None:
        if self._perf is not None:
            self._run_started_perf = self._perf()

    def run_ended(self) -> None:
        if self._perf is not None:
            self._run_ended_perf = self._perf()

    def observe_latency(self, kind: LatencyKind, seconds: float) -> None:
        self._latencies.setdefault(kind, _LatencyAccumulator()).add(seconds)

    def observe_media_timestamp(self, timestamp_seconds: float) -> None:
        if self._first_media_ts is None:
            self._first_media_ts = timestamp_seconds
        self._last_media_ts = timestamp_seconds

    def observe_queue_depth(self, depth: int) -> None:
        self.queue_peak = max(self.queue_peak, depth)

    def sample_resources(self) -> None:
        """Read the memory/GPU probes (no-op without them)."""

        if self._memory_probe is not None:
            current = self._memory_probe()
            self._memory_current = current
            self._memory_peak = max(self._memory_peak or 0, current)
        if self._gpu_probe is not None:
            current = self._gpu_probe()
            self._gpu_current = current
            self._gpu_peak = max(self._gpu_peak or 0, current)

    # --- snapshot -------------------------------------------------------------------
    def snapshot(self) -> EngineMetrics:
        media_fps: float | None = None
        if (
            self.frames_processed > 1
            and self._first_media_ts is not None
            and self._last_media_ts is not None
            and self._last_media_ts > self._first_media_ts
        ):
            span = self._last_media_ts - self._first_media_ts
            media_fps = (self.frames_processed - 1) / span

        wall_fps: float | None = None
        if (
            self._run_started_perf is not None
            and self._run_ended_perf is not None
            and self._run_ended_perf > self._run_started_perf
            and self.frames_processed
        ):
            wall_fps = self.frames_processed / (self._run_ended_perf - self._run_started_perf)

        return EngineMetrics(
            frames_read=self.frames_read,
            frames_skipped_stride=self.frames_skipped_stride,
            frames_skipped_fps=self.frames_skipped_fps,
            frames_dropped_backpressure=self.frames_dropped_backpressure,
            frames_admitted=self.frames_admitted,
            frames_processed=self.frames_processed,
            batches_processed=self.batches_processed,
            detections=self.detections,
            track_states=self.track_states,
            events_confirmed=self.events_confirmed,
            queue_peak=self.queue_peak,
            media_fps=media_fps,
            wall_fps=wall_fps,
            latencies={
                kind.value: acc.summary() for kind, acc in sorted(self._latencies.items())
            },
            memory_bytes_current=self._memory_current,
            memory_bytes_peak=self._memory_peak,
            gpu_memory_bytes_current=self._gpu_current,
            gpu_memory_bytes_peak=self._gpu_peak,
        )


def torch_cuda_memory_probe() -> Callable[[], int] | None:
    """Build a GPU memory probe over the optional torch backend, or ``None``.

    Lazily imports torch (the ``rtdetr`` extra); returns ``None`` when torch is
    absent or CUDA is unavailable, so the result can be passed straight to
    :class:`MetricsRecorder`. The probe reports ``torch.cuda.memory_allocated()``
    -- allocated bytes on the current device, the truthful utilization signal
    available without adding a dependency.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - torch present in this env
        return None
    if not torch.cuda.is_available():
        return None

    def probe() -> int:  # pragma: no cover - needs CUDA hardware
        allocated: int = torch.cuda.memory_allocated()
        return allocated

    return probe  # pragma: no cover - needs CUDA hardware

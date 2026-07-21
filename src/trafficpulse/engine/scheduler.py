"""Deterministic frame scheduling: decimation + bounded back-pressure (H6).

The scheduler is a **pure policy object** between a frame source and the
processing core. For every submitted frame it makes exactly one decision:

* ``SKIPPED_STRIDE`` -- the frame's read ordinal fails the ``frame_stride`` gate;
* ``SKIPPED_FPS`` -- admitting it would exceed ``target_fps`` in **media time**
  (PTS seconds; wall-clock never participates, so decisions replay exactly);
* ``DROPPED_QUEUE_FULL`` -- the bounded pending queue is full (back-pressure).
  The *incoming* frame is dropped: the queued frames are older and already
  admitted, so rejecting the newest keeps the admitted stream's frame order
  strictly monotonic -- the discipline the tracker seam requires;
* ``ADMITTED`` -- enqueued for processing.

Back-pressure model
-------------------
``submit`` and ``take`` are deliberately decoupled: a synchronous run drains
after every submit (the queue never exceeds one), while a live producer may
submit many frames between drains -- exactly when the bounded queue and the
drop counter become load-bearing. The engine's metrics record the queue's
high-water mark and every drop.

Determinism
-----------
No wall-clock, no randomness, no thread. Decisions are a pure function of the
construction config and the submitted stream (ordinals + PTS timestamps), so a
replayed stream schedules identically. The FPS gate compares against the last
**admitted** frame's timestamp with a small epsilon so exact-interval streams
(e.g. 10 fps decimated to 5) admit every other frame without float jitter.
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from ..ingestion.video import FrameRecord
from .config import SchedulerConfig

# Absorbs float noise in PTS arithmetic at the FPS gate; one microsecond of
# media time, far below any real inter-frame interval.
_FPS_EPSILON_S = 1e-6


class ScheduleDecision(StrEnum):
    """What the scheduler did with one submitted frame."""

    ADMITTED = "admitted"
    SKIPPED_STRIDE = "skipped_stride"
    SKIPPED_FPS = "skipped_fps"
    DROPPED_QUEUE_FULL = "dropped_queue_full"


class FrameScheduler:
    """Bounded, deterministic frame admission (see module docstring)."""

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._queue: deque[FrameRecord] = deque()
        self._read_ordinal = 0
        self._last_admitted_ts: float | None = None

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    @property
    def queue_depth(self) -> int:
        """Frames currently admitted but not yet taken."""

        return len(self._queue)

    def reset(self) -> None:
        """Return to the initial state (empty queue, ordinal 0) for replay."""

        self._queue.clear()
        self._read_ordinal = 0
        self._last_admitted_ts = None

    def submit(self, record: FrameRecord) -> ScheduleDecision:
        """Decide one frame's fate; enqueue it when admitted."""

        ordinal = self._read_ordinal
        self._read_ordinal += 1

        if ordinal % self._config.frame_stride != 0:
            return ScheduleDecision.SKIPPED_STRIDE

        if self._config.target_fps is not None and self._last_admitted_ts is not None:
            interval = 1.0 / self._config.target_fps
            elapsed = record.timestamp_seconds - self._last_admitted_ts
            if elapsed < interval - _FPS_EPSILON_S:
                return ScheduleDecision.SKIPPED_FPS

        if len(self._queue) >= self._config.queue_capacity:
            return ScheduleDecision.DROPPED_QUEUE_FULL

        self._queue.append(record)
        self._last_admitted_ts = record.timestamp_seconds
        return ScheduleDecision.ADMITTED

    def take(self, limit: int) -> tuple[FrameRecord, ...]:
        """Dequeue up to ``limit`` admitted frames, oldest first."""

        if limit < 1:
            raise ValueError(f"take limit must be >= 1, got {limit}")
        taken: list[FrameRecord] = []
        while self._queue and len(taken) < limit:
            taken.append(self._queue.popleft())
        return tuple(taken)

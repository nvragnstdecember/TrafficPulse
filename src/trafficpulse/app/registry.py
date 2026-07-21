"""In-memory registries + job execution seam for the API layer (H7A).

The application keeps two small, thread-safe registries -- one of uploaded
videos, one of processing jobs -- plus the seam that decides *how* a job runs.
Neither registry holds domain logic: they are the request-scoped state a stateless
HTTP layer needs between calls. The persisted events themselves live in the H6
``EventStore`` (the durable source of truth); these registries only address them.

Job execution is injected, not hardcoded
-----------------------------------------
:class:`JobExecutor` is the seam that runs a job's work function. Production uses
:class:`ThreadJobExecutor` (a job returns its id immediately and processes in the
background); tests use :class:`SynchronousJobExecutor` (the job runs inline, so
the whole lifecycle is deterministic with no threads). This is what lets the
processing lifecycle be tested without concurrency while production stays
non-blocking.

Thread-safety
-------------
Both stores guard their state with a lock; job mutation goes through
:class:`JobStore` methods so a background writer and a status reader never race on
a half-updated record. Live in-flight metrics are read from the engine's own
thread-safe snapshot, outside the store lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..engine import EngineMetrics, EngineRunResult, InferenceEngine


# --- videos --------------------------------------------------------------------
@dataclass(frozen=True)
class VideoRecord:
    """An uploaded, validated, stored source video (metadata only)."""

    video_id: str
    filename: str
    path: Path
    size_bytes: int
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    codec: str


class VideoStore:
    """Thread-safe in-memory index of uploaded videos, keyed by content id."""

    def __init__(self) -> None:
        self._videos: dict[str, VideoRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: VideoRecord) -> None:
        with self._lock:
            self._videos[record.video_id] = record

    def get(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            return self._videos.get(video_id)

    def contains(self, video_id: str) -> bool:
        with self._lock:
            return video_id in self._videos


# --- jobs ----------------------------------------------------------------------
class JobStatus(StrEnum):
    """Processing-job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobRecord:
    """One processing job's mutable state (guarded by :class:`JobStore`)."""

    job_id: str
    video_id: str
    status: JobStatus = JobStatus.PENDING
    frames_total: int | None = None
    event_ids: tuple[str, ...] = ()
    error: str | None = None
    engine: InferenceEngine | None = field(default=None, repr=False)
    result: EngineRunResult | None = field(default=None, repr=False)

    def metrics(self) -> EngineMetrics | None:
        """The best available metrics snapshot: final if done, else live, else none.

        Reads the engine's own thread-safe snapshot for an in-flight job -- never a
        fabricated value.
        """

        if self.result is not None:
            return self.result.metrics
        if self.engine is not None:
            return self.engine.metrics
        return None


class JobStore:
    """Thread-safe registry of jobs + an ``event_id -> job_id`` reverse index."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._event_index: dict[str, str] = {}
        self._lock = threading.Lock()

    def add(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str, *, frames_total: int | None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.RUNNING
            record.frames_total = frames_total

    def mark_succeeded(self, job_id: str, result: EngineRunResult) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.SUCCEEDED
            record.result = result
            record.event_ids = tuple(event.event_id for event in result.events)
            for event in result.events:
                self._event_index[event.event_id] = job_id

    def mark_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.FAILED
            record.error = message

    def job_for_event(self, event_id: str) -> str | None:
        with self._lock:
            return self._event_index.get(event_id)

    def jobs(self) -> tuple[JobRecord, ...]:
        """All jobs, in insertion order (deterministic)."""

        with self._lock:
            return tuple(self._jobs.values())

    def succeeded_for_video(self, video_id: str | None) -> tuple[JobRecord, ...]:
        """Succeeded jobs, optionally filtered to one video, in insertion order."""

        with self._lock:
            return tuple(
                record
                for record in self._jobs.values()
                if record.status is JobStatus.SUCCEEDED
                and (video_id is None or record.video_id == video_id)
            )


# --- execution seam ------------------------------------------------------------
JobWork = Callable[[], None]


class JobExecutor(Protocol):
    """Runs a job's work function; the injectable async/sync policy."""

    def submit(self, work: JobWork) -> None:
        """Run (or schedule) ``work``. May return before ``work`` completes."""
        ...


class SynchronousJobExecutor:
    """Runs the work inline before returning -- deterministic, thread-free."""

    def submit(self, work: JobWork) -> None:
        work()


class ThreadJobExecutor:
    """Runs the work on a daemon thread -- non-blocking for production."""

    def submit(self, work: JobWork) -> None:
        threading.Thread(target=work, name="trafficpulse-job", daemon=True).start()

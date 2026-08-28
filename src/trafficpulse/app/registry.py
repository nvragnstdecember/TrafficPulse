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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..engine import EngineMetrics, EngineRunResult, InferenceEngine


def _utc_now() -> datetime:
    """The wall-clock instant a lifecycle transition happened.

    Confined to job lifecycle timing. Nothing in the *decision* path may read a
    clock -- media time is PTS-derived and epoch-anchored so replay stays
    byte-deterministic -- and these timestamps are observational metadata about a
    run, never an input to one.
    """

    return datetime.now(UTC)


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
    uploaded_at: datetime | None = None
    """When the upload was accepted (H11), so a library can order by recency.

    Optional because it is genuinely unknown for a video stored before this field
    existed: recovery falls back to the media file's modification time, and a
    record with neither reports ``None`` rather than inventing an instant."""

    scene_hash: str | None = None
    """The scene revision this video is calibrated against (H12), or ``None``.

    The binding that replaces the application-wide scene: processing resolves the
    scene *of the video* rather than of the server. It is a content hash, so it
    names one immutable revision -- re-calibrating rebinds to a new one and leaves
    every event already reasoned under the old one still able to resolve it.

    ``None`` means "not calibrated": the video falls back to the server's
    configured scene, if it has one, and only the geometry-free rules apply."""


class VideoStore:
    """Thread-safe in-memory index of uploaded videos, keyed by content id.

    ``on_change`` is an optional observer invoked after a record is added, used by
    startup recovery (H10) to write the record's snapshot. The store itself stays
    in-memory and knows nothing about storage: it calls a function. The callback
    runs **outside** the lock, so persisting never serialises readers behind I/O.
    """

    def __init__(self, on_change: Callable[[VideoRecord], None] | None = None) -> None:
        self._videos: dict[str, VideoRecord] = {}
        self._lock = threading.Lock()
        self._on_change = on_change

    def add(self, record: VideoRecord) -> None:
        with self._lock:
            self._videos[record.video_id] = record
        self._notify(record)

    def restore(self, record: VideoRecord) -> None:
        """Insert a record rebuilt from disk, without re-persisting it."""

        with self._lock:
            self._videos[record.video_id] = record

    def bind_scene(self, video_id: str, scene_hash: str | None) -> VideoRecord | None:
        """Point a video at a scene revision (H12); return the updated record.

        Returns ``None`` for an unknown video. The record is frozen, so this
        replaces it rather than mutating -- and because the replacement goes
        through the change observer, the video's snapshot is rewritten and the
        binding survives a restart with no extra call site to forget.
        """

        with self._lock:
            existing = self._videos.get(video_id)
            if existing is None:
                return None
            updated = replace(existing, scene_hash=scene_hash)
            self._videos[video_id] = updated
        self._notify(updated)
        return updated

    def _notify(self, record: VideoRecord) -> None:
        if self._on_change is not None:
            self._on_change(record)

    def get(self, video_id: str) -> VideoRecord | None:
        with self._lock:
            return self._videos.get(video_id)

    def contains(self, video_id: str) -> bool:
        with self._lock:
            return video_id in self._videos

    def videos(self) -> tuple[VideoRecord, ...]:
        """Every stored video, in insertion order (H11).

        The accessor that makes the store enumerable rather than merely
        addressable -- the historical library needs to *browse* uploads, and
        ``get``/``contains`` can only answer questions about an id you already
        hold. Recovery restores in sorted id order, so a restarted process yields
        a deterministic sequence; ordering for presentation is the caller's.
        """

        with self._lock:
            return tuple(self._videos.values())


# --- jobs ----------------------------------------------------------------------
class JobStatus(StrEnum):
    """Processing-job lifecycle states.

    ``CANCELLED`` is a terminal state distinct from ``FAILED``: the job was
    stopped on request before completing, which is not an error. It is additive
    -- clients that only know the original four states still receive a valid
    status string.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True once the job can no longer change state."""

        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class OverlayStatus(StrEnum):
    """Lifecycle of a job's annotated (overlay) video, independent of ``JobStatus``.

    The overlay is a *presentation* artifact rendered after inference finishes:
    events and evidence are already persisted, and the annotated MP4 is produced by
    a second decode+encode pass that can easily outlast the inference itself. Its
    readiness therefore cannot be inferred from the job status -- a job is
    ``succeeded`` (events queryable, review can start) while its overlay is still
    ``PENDING``. Publishing this axis is what lets a client keep polling until the
    artifact resolves instead of concluding, from a terminal job status, that no
    overlay will ever exist.

    ``NONE`` -- the run produced no overlay metadata, so there is nothing to draw
    (e.g. no rule with a pixel observer ran); ``FAILED`` -- a render was attempted
    and did not complete. Both are terminal and mean "play the original video";
    they are distinguished so an operator can tell an absent overlay from a broken
    one.
    """

    NONE = "none"
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True once the overlay outcome can no longer change (stop polling)."""

        return self is not OverlayStatus.PENDING


class EvidenceStatus(StrEnum):
    """Lifecycle of a run's rendered evidence frames (H16).

    The counterpart of :class:`OverlayStatus`, and added for the same reason: a
    render happens *after* the run's events and manifests are durably persisted,
    so its outcome cannot be inferred from ``JobStatus``. Until H16 evidence
    rendering had no status at all, which meant a render interrupted by a restart
    left a partial artifact set that was **indistinguishable from a complete one** --
    the dashboard reported the shortfall as the truth and nothing could tell an
    operator otherwise.

    ``NONE`` -- nothing to render (the run confirmed no events, or the manifests
    predate engine frame picking); ``PENDING`` -- a render is in flight;
    ``READY`` -- every event that could be rendered was; ``FAILED`` -- a render was
    attempted and did not complete, so the recorded artifacts may be partial.

    ``PENDING`` must never survive a restart: nothing resumes it. Recovery settles
    it to ``FAILED``, which is what makes an interrupted render visible and
    repairable rather than silently partial.
    """

    NONE = "none"
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True once the evidence outcome can no longer change on its own."""

        return self is not EvidenceStatus.PENDING

    @property
    def needs_repair(self) -> bool:
        """Whether a repair pass could still produce missing artifacts."""

        return self is EvidenceStatus.FAILED


@dataclass
class JobRecord:
    """One processing job's mutable state (guarded by :class:`JobStore`)."""

    job_id: str
    video_id: str
    status: JobStatus = JobStatus.PENDING
    frames_total: int | None = None
    event_ids: tuple[str, ...] = ()
    error: str | None = None
    cancel_requested: bool = False
    engine: InferenceEngine | None = field(default=None, repr=False)
    result: EngineRunResult | None = field(default=None, repr=False)
    # Path to the rendered overlay (annotated) video, set after a successful run
    # when the overlay framework produced one. The original upload is untouched.
    overlay_video: Path | None = field(default=None, repr=False)
    # Where the overlay render has got to; see :class:`OverlayStatus`. Starts at
    # NONE (nothing attempted) and is advanced by the processing service.
    overlay_status: OverlayStatus = OverlayStatus.NONE
    # Where the evidence-frame render has got to; see :class:`EvidenceStatus`.
    evidence_status: EvidenceStatus = EvidenceStatus.NONE
    # Metrics read back from a run snapshot at startup (H10). A recovered job has
    # neither a live engine nor an in-process result, so this is the only truthful
    # source left for its frame counts and FPS.
    restored_metrics: EngineMetrics | None = field(default=None, repr=False)

    # --- H15 wall-clock lifecycle timing -------------------------------------
    # The *only* wall-clock instants a run records. Everything else in this system
    # is media time anchored at a fixed epoch (deliberately, so replay is
    # byte-deterministic), which means a run's real duration and the calendar day
    # it happened on were previously unrecoverable. These three fields exist so
    # analytics can answer "when" without ever reinterpreting media time as a date.
    #
    # Each is optional and stays ``None`` when unknown: a job recovered from a
    # snapshot written before H15 has no timing, and reports that honestly rather
    # than substituting the recovery instant.
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Confirmed events per violation type, computed once when the run succeeds.
    # The same trick H10 used for the event index: a value that would otherwise
    # require re-reading every event file is written down at the moment it is
    # already in hand, so repository-wide analytics costs O(jobs), not O(events).
    # Empty for a job that confirmed nothing *and* for one recovered from a
    # pre-H15 snapshot -- see ``JobRecord.has_violation_counts``.
    violation_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_violation_counts(self) -> bool:
        """Whether this job's histogram is a measurement rather than an absence.

        A pre-H15 snapshot and a run that confirmed nothing both present an empty
        mapping, but they mean different things: "not recorded" versus "recorded,
        and it was zero". Event ids are recovered independently (from the run's
        ``events/`` listing), so their presence is what distinguishes the two.
        """

        return bool(self.violation_counts) or not self.event_ids

    def duration_seconds(self) -> float | None:
        """Wall-clock seconds from start to finish, or ``None`` if not both known."""

        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, (self.finished_at - self.started_at).total_seconds())

    def metrics(self) -> EngineMetrics | None:
        """The best available metrics snapshot: final if done, else live, else none.

        Reads the engine's own thread-safe snapshot for an in-flight job -- never a
        fabricated value. A job recovered from disk falls through to the metrics its
        run recorded; a job with none reports ``None`` rather than zeros.
        """

        if self.result is not None:
            return self.result.metrics
        if self.engine is not None:
            return self.engine.metrics
        return self.restored_metrics


class JobStore:
    """Thread-safe registry of jobs + an ``event_id -> job_id`` reverse index.

    ``on_change`` is an optional observer invoked after **every** state transition,
    used by startup recovery (H10) to keep each job's snapshot current. Routing it
    through the store rather than through the calling service is deliberate: a job
    has eight mutators, and a snapshot call added at each site is one refactor away
    from being forgotten. The store still knows nothing about storage -- it calls a
    function, and does so **outside** the lock so persistence never serialises
    status readers behind I/O.

    Ordering note: one job is driven by a single worker thread, so its transitions
    -- and therefore its snapshots -- are sequential. Cross-job writes touch
    different files and cannot interleave.
    """

    def __init__(self, on_change: Callable[[JobRecord], None] | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._event_index: dict[str, str] = {}
        self._lock = threading.Lock()
        self._on_change = on_change

    def add(self, record: JobRecord) -> None:
        with self._lock:
            if record.submitted_at is None:
                record.submitted_at = _utc_now()
            self._jobs[record.job_id] = record
        self._notify(record.job_id)

    def restore(self, record: JobRecord, event_ids: Sequence[str]) -> None:
        """Insert a job rebuilt from disk and index its events.

        Bypasses ``on_change``: recovery is reading what was already written, and
        echoing it straight back would be pointless I/O on every boot.
        """

        with self._lock:
            self._jobs[record.job_id] = record
            for event_id in event_ids:
                self._event_index[event_id] = record.job_id

    def _notify(self, job_id: str) -> None:
        if self._on_change is None:
            return
        record = self.get(job_id)
        if record is not None:
            self._on_change(record)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_engine(self, job_id: str, engine: InferenceEngine) -> None:
        """Attach the engine a job will run, once it is known.

        A job whose scene is derived inside its own work function cannot be given
        an engine at submit time -- the engine is built *from* that scene. The
        attachment goes through the store, under the lock, for the same reason
        every other mutation does: ``status`` reads ``job.engine`` for live
        metrics from a different thread than the one that builds it.
        """

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.engine = engine
        # Deliberately no ``on_change``: attaching the engine is wiring, not a
        # state transition, and the snapshot records nothing that changed here.

    def mark_running(self, job_id: str, *, frames_total: int | None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.RUNNING
            record.frames_total = frames_total
            record.started_at = _utc_now()
        self._notify(job_id)

    def mark_succeeded(self, job_id: str, result: EngineRunResult) -> None:
        """Record a run's outcome, folding the result to one entry per ``event_id``.

        Deduplication happens **here**, at the boundary where a run's events first
        become job-level metadata, rather than in the engine: the engine's per-rule
        fan-out is deliberately un-deduplicated (two identical rule declarations
        genuinely run twice, which is what makes a one-rule engine event-identical
        to the standalone pipeline), and the write-once ``EventStore`` already
        absorbs the repeat as a byte-identical no-op. Only the *counts* were wrong,
        so only the counts are fixed, and only where they are formed.

        ``event_id`` is content-derived, so two entries sharing one are the same
        confirmed violation reasoned twice -- never two violations. First occurrence
        wins, and because ``result.events`` arrives sorted by
        ``(trigger_at, event_id)`` the retained order is that sort, unchanged.
        """

        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.SUCCEEDED
            record.result = result
            # The histogram is computed here because this is the one moment the
            # events are already in memory; every later reader gets it for free.
            event_ids: list[str] = []
            counts: dict[str, int] = {}
            seen: set[str] = set()
            for event in result.events:
                if event.event_id in seen:
                    continue  # the same violation, confirmed by a duplicate rule
                seen.add(event.event_id)
                event_ids.append(event.event_id)
                self._event_index[event.event_id] = job_id
                key = event.violation_type.value
                counts[key] = counts.get(key, 0) + 1
            record.event_ids = tuple(event_ids)
            record.violation_counts = counts
            record.finished_at = _utc_now()
        self._notify(job_id)

    def mark_overlay_pending(self, job_id: str) -> None:
        """Declare that an overlay render is about to start for this job.

        Called **before** the job is marked succeeded, so the first terminal status
        a client observes already says an overlay is coming -- which is what keeps
        it polling across the render window instead of settling on "no overlay".
        """

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.overlay_status = OverlayStatus.PENDING
        self._notify(job_id)

    def set_overlay_video(self, job_id: str, path: Path) -> None:
        """Record the rendered overlay-video artifact and mark it ready."""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.overlay_video = path
                record.overlay_status = OverlayStatus.READY
        self._notify(job_id)

    def resolve_overlay(self, job_id: str, status: OverlayStatus) -> None:
        """Settle a pending overlay on a terminal outcome (``NONE`` / ``FAILED``)."""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.overlay_status = status
        self._notify(job_id)

    def set_evidence_status(self, job_id: str, status: EvidenceStatus) -> None:
        """Record where this run's evidence render has got to (H16).

        Persisted through the same change observer as every other transition, so a
        render marked ``PENDING`` is on disk *before* the work starts -- which is
        exactly what lets recovery notice that a restart interrupted it.
        """

        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.evidence_status = status
        self._notify(job_id)

    def mark_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.FAILED
            record.error = message
            record.finished_at = _utc_now()
        self._notify(job_id)

    def mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.CANCELLED
            record.finished_at = _utc_now()
        self._notify(job_id)

    def request_cancel(self, job_id: str) -> bool:
        """Flag a non-terminal job for cooperative cancellation.

        Returns ``True`` when the flag was newly set on a still-cancellable job,
        ``False`` for an unknown or already-terminal job (a safe no-op, so
        cancelling a finished job never changes its recorded outcome). The
        background run observes the flag between frames (see
        :meth:`is_cancel_requested`).
        """

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.status.is_terminal:
                return False
            record.cancel_requested = True
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        """Whether cancellation has been requested for a job (thread-safe read)."""

        with self._lock:
            record = self._jobs.get(job_id)
            return record is not None and record.cancel_requested

    def job_for_event(self, event_id: str) -> str | None:
        with self._lock:
            return self._event_index.get(event_id)

    def jobs(self) -> tuple[JobRecord, ...]:
        """All jobs, in insertion order (deterministic)."""

        with self._lock:
            return tuple(self._jobs.values())

    def for_video(self, video_id: str) -> tuple[JobRecord, ...]:
        """Every job for one video, whatever its status, in insertion order (H11).

        Distinct from :meth:`succeeded_for_video`, which answers "where are this
        video's events". The library has to describe a video that was uploaded and
        then failed, or is still running, so it needs the unfiltered history.
        """

        with self._lock:
            return tuple(
                record for record in self._jobs.values() if record.video_id == video_id
            )

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

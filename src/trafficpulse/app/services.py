"""Application services -- the business logic between routers and H6 (H7A).

Five services hold everything the thin routers delegate to. They **compose** the
existing layers and add no reasoning, detection, tracking, or persistence logic of
their own:

* :class:`VideoService` -- validates and stores uploads (readability via P1-U5
  ingestion, addressed by content hash).
* :class:`ProcessingService` -- creates jobs and drives the H6 engine
  (``engine.run`` + ``engine.persist``) through the injected executor + provider.
* :class:`EventService` / :class:`EvidenceService` -- read persisted events and
  manifests back from the H6 ``EventStore`` (the durable source of truth).
* :class:`MetricsService` -- aggregates job counts and surfaces the latest job's
  H6 ``EngineMetrics`` verbatim.

Every lower-layer failure is translated to a typed :class:`AppError`, so the HTTP
contract never leaks an internal exception or a traceback.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from ..contracts import ConfirmedEvent, EvidenceManifest, SceneConfig
from ..detector.errors import DetectorError
from ..engine import FileFrameSource, InferenceEngine, RuleConfig
from ..engine.errors import EngineConfigurationError, UnsupportedRuleError
from ..ingestion.video import VideoIngestionError
from ..persistence import EventStore, RunNotFoundError
from ..pipeline.errors import SceneConfigurationError
from .config import AppConfig
from .engine_provider import EngineProvider
from .errors import (
    AppError,
    DuplicateVideoError,
    EngineUnavailableError,
    EventNotFoundError,
    InvalidConfigurationError,
    JobNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    VideoNotFoundError,
)
from .models import (
    EventListResponse,
    EventSort,
    EventSummary,
    JobStatusResponse,
    MetricsResponse,
)
from .registry import (
    JobExecutor,
    JobRecord,
    JobStatus,
    JobStore,
    VideoRecord,
    VideoStore,
)

_logger = logging.getLogger("trafficpulse.app")

JobIdFactory = Callable[[], str]


def _default_job_id() -> str:
    return "job-" + uuid.uuid4().hex[:16]


# --- videos --------------------------------------------------------------------
class VideoService:
    """Validates and stores uploaded videos, addressed by content hash."""

    def __init__(self, config: AppConfig, store: VideoStore) -> None:
        self._config = config
        self._store = store

    def assert_supported_extension(self, filename: str) -> None:
        """Fast-fail before reading a body: reject an unsupported extension."""

        if not self._config.is_supported_extension(Path(filename).suffix):
            allowed = ", ".join(sorted(self._config.allowed_extensions))
            raise UnsupportedMediaError(
                f"unsupported file extension {Path(filename).suffix!r}; "
                f"supported: {allowed}"
            )

    def store_upload(self, filename: str, data: bytes) -> VideoRecord:
        """Validate + persist one upload; return its record.

        Validates extension, non-emptiness, size, content-uniqueness, and
        readability (by actually opening the file through P1-U5 ingestion). A file
        that fails the readability check is removed, so no half-valid upload
        lingers on disk.
        """

        self.assert_supported_extension(filename)
        if not data:
            raise UnsupportedMediaError("uploaded file is empty")
        if len(data) > self._config.max_upload_bytes:
            raise PayloadTooLargeError(
                f"upload is {len(data)} bytes; the limit is "
                f"{self._config.max_upload_bytes} bytes"
            )

        video_id = "vid-" + hashlib.sha256(data).hexdigest()[:16]
        if self._store.contains(video_id):
            raise DuplicateVideoError(
                f"an identical video already exists as {video_id}", video_id=video_id
            )

        self._config.videos_dir.mkdir(parents=True, exist_ok=True)
        path = self._config.videos_dir / f"{video_id}{Path(filename).suffix.lower()}"
        path.write_bytes(data)

        try:
            source = FileFrameSource(path)
            metadata = source.metadata
        except VideoIngestionError as exc:
            path.unlink(missing_ok=True)
            raise UnsupportedMediaError(
                f"uploaded file is not a readable video: {exc}"
            ) from exc

        record = VideoRecord(
            video_id=video_id,
            filename=filename,
            path=path,
            size_bytes=len(data),
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            duration_seconds=metadata.duration_seconds,
            codec=metadata.codec,
        )
        self._store.add(record)
        return record

    def require(self, video_id: str) -> VideoRecord:
        """Return the record for ``video_id`` or raise :class:`VideoNotFoundError`."""

        record = self._store.get(video_id)
        if record is None:
            raise VideoNotFoundError(f"no uploaded video with id {video_id!r}")
        return record


# --- processing ----------------------------------------------------------------
class ProcessingService:
    """Creates and drives processing jobs over the injected H6 engine."""

    def __init__(
        self,
        *,
        config: AppConfig,
        scene: SceneConfig | None,
        provider: EngineProvider,
        store: EventStore,
        job_store: JobStore,
        executor: JobExecutor,
        videos: VideoService,
        job_id_factory: JobIdFactory = _default_job_id,
    ) -> None:
        self._config = config
        self._scene = scene
        self._provider = provider
        self._store = store
        self._jobs = job_store
        self._executor = executor
        self._videos = videos
        self._job_id_factory = job_id_factory

    def submit(
        self, *, video_id: str, rules: tuple[RuleConfig, ...] | None
    ) -> JobRecord:
        """Validate the request, create a job, and schedule its execution.

        Validation is eager: the engine is *built* here (so an invalid scene/rule
        combination or an unavailable backend fails as a clean HTTP error) before
        the job is scheduled. Only the actual inference runs on the executor.
        """

        video = self._videos.require(video_id)
        if self._scene is None:
            raise EngineUnavailableError(
                "no scene is configured; the server cannot process video yet"
            )
        resolved = rules if rules is not None else self._config.default_rules
        if not resolved:
            raise InvalidConfigurationError(
                "no rules were specified and the server has no default rule set"
            )

        engine = self._build_engine(resolved)
        job = JobRecord(job_id=self._job_id_factory(), video_id=video_id, engine=engine)
        self._jobs.add(job)
        scene = self._scene
        self._executor.submit(lambda: self._run(job, video, scene))
        return job

    def _build_engine(self, rules: tuple[RuleConfig, ...]) -> InferenceEngine:
        assert self._scene is not None
        try:
            return self._provider.create(scene=self._scene, rules=rules)
        except (
            SceneConfigurationError,
            EngineConfigurationError,
            UnsupportedRuleError,
            ValueError,
        ) as exc:
            raise InvalidConfigurationError(str(exc)) from exc
        except EngineUnavailableError:
            raise
        except DetectorError as exc:
            raise EngineUnavailableError(
                f"the inference backend is unavailable: {exc}"
            ) from exc

    def _run(self, job: JobRecord, video: VideoRecord, scene: SceneConfig) -> None:
        """Execute one job: run the H6 engine and persist its output."""

        assert job.engine is not None
        try:
            source = FileFrameSource(video.path, camera_id=scene.scene.camera_id)
            self._jobs.mark_running(job.job_id, frames_total=source.metadata.frame_count)
            result = job.engine.run(source)
            job.engine.persist(result, store=self._store, run_id=job.job_id)
            self._jobs.mark_succeeded(job.job_id, result)
        except Exception as exc:  # noqa: BLE001 - a job thread must never crash silently
            _logger.exception("processing job %s failed", job.job_id)
            self._jobs.mark_failed(job.job_id, str(exc))

    def status(self, job_id: str) -> JobStatusResponse:
        """Return one job's live status (unavailable values are null, not faked)."""

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")

        metrics = job.metrics()
        processed = metrics.frames_processed if metrics is not None else 0
        total = job.frames_total
        fps = metrics.media_fps if metrics is not None else None

        progress: float | None
        if job.status is JobStatus.SUCCEEDED:
            progress = 1.0
        elif total and metrics is not None:
            progress = min(1.0, processed / total)
        else:
            progress = None

        remaining: float | None = None
        if (
            job.status is JobStatus.RUNNING
            and metrics is not None
            and metrics.wall_fps
            and total
        ):
            remaining = max(0.0, (total - processed) / metrics.wall_fps)

        return JobStatusResponse(
            job_id=job.job_id,
            video_id=job.video_id,
            status=job.status,
            progress=progress,
            frames_processed=processed,
            frames_total=total,
            fps=fps,
            estimated_remaining_seconds=remaining,
            event_count=len(job.event_ids),
            error=job.error,
        )


# --- events + evidence ---------------------------------------------------------
class EventService:
    """Reads persisted confirmed events back from the H6 ``EventStore``."""

    def __init__(self, store: EventStore, job_store: JobStore) -> None:
        self._store = store
        self._jobs = job_store

    def list(
        self,
        *,
        video_id: str | None,
        limit: int,
        offset: int,
        sort: EventSort,
    ) -> EventListResponse:
        """Return a deterministic page of event summaries."""

        summaries: list[EventSummary] = []
        seen: set[str] = set()
        for job in self._jobs.succeeded_for_video(video_id):
            if not job.event_ids:
                continue  # a succeeded job that confirmed nothing persisted no run
            for pair in self._store.load(job.job_id):
                event = pair.event
                if event.event_id in seen:
                    continue
                seen.add(event.event_id)
                summaries.append(
                    EventSummary(
                        event_id=event.event_id,
                        video_id=job.video_id,
                        job_id=job.job_id,
                        violation_type=event.violation_type,
                        camera_id=event.camera_id,
                        track_ids=event.track_ids,
                        trigger_at=event.trigger_at,
                        rule_id=event.rule_id,
                    )
                )

        ordered = _sorted_summaries(summaries, sort)
        page = ordered[offset : offset + limit]
        return EventListResponse(
            items=tuple(page), total=len(ordered), limit=limit, offset=offset
        )

    def get(self, event_id: str) -> ConfirmedEvent:
        """Return the full contract for one event or raise :class:`EventNotFoundError`."""

        return self.locate(event_id)[0]

    def locate(self, event_id: str) -> tuple[ConfirmedEvent, EvidenceManifest]:
        """Find one event + its manifest across succeeded runs (raises if unknown)."""

        job_id = self._jobs.job_for_event(event_id)
        if job_id is None:
            raise EventNotFoundError(f"no confirmed event with id {event_id!r}")
        try:
            stored = self._store.load(job_id)
        except RunNotFoundError as exc:  # pragma: no cover - index implies persistence
            raise EventNotFoundError(
                f"event {event_id!r} is indexed but its run is missing"
            ) from exc
        for pair in stored:
            if pair.event.event_id == event_id:
                return pair.event, pair.manifest
        raise EventNotFoundError(  # pragma: no cover - index guarantees membership
            f"event {event_id!r} is indexed to a run that does not contain it"
        )


class EvidenceService:
    """Returns the evidence manifest (references only) for a confirmed event."""

    def __init__(self, event_service: EventService) -> None:
        self._events = event_service

    def get(self, event_id: str) -> EvidenceManifest:
        """Return the manifest for ``event_id`` (frame references, no media)."""

        return self._events.locate(event_id)[1]


# --- metrics -------------------------------------------------------------------
class MetricsService:
    """Aggregates job counts and surfaces the latest H6 ``EngineMetrics``."""

    def __init__(self, job_store: JobStore) -> None:
        self._jobs = job_store

    def snapshot(self) -> MetricsResponse:
        jobs = self._jobs.jobs()
        by_status = {status: 0 for status in JobStatus}
        events_total = 0
        for job in jobs:
            by_status[job.status] += 1
            events_total += len(job.event_ids)

        latest = None
        for job in reversed(jobs):  # most recent job carrying metrics
            metrics = job.metrics()
            if metrics is not None:
                latest = metrics
                break

        return MetricsResponse(
            jobs_total=len(jobs),
            jobs_pending=by_status[JobStatus.PENDING],
            jobs_running=by_status[JobStatus.RUNNING],
            jobs_succeeded=by_status[JobStatus.SUCCEEDED],
            jobs_failed=by_status[JobStatus.FAILED],
            events_total=events_total,
            latest=latest,
        )


def _sorted_summaries(
    summaries: list[EventSummary], sort: EventSort
) -> list[EventSummary]:
    """Deterministically order summaries (event_id is always the final tie-break)."""

    if sort in (EventSort.EVENT_ID_ASC, EventSort.EVENT_ID_DESC):
        return sorted(
            summaries,
            key=lambda s: s.event_id,
            reverse=sort is EventSort.EVENT_ID_DESC,
        )
    return sorted(
        summaries,
        key=lambda s: (s.trigger_at, s.event_id),
        reverse=sort is EventSort.TRIGGER_AT_DESC,
    )


# The public AppError base is imported for handlers/tests; re-export keeps the
# service module the single import surface for the application error taxonomy.
__all__ = [
    "VideoService",
    "ProcessingService",
    "EventService",
    "EvidenceService",
    "MetricsService",
    "AppError",
]

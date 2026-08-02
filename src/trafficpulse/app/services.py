"""Application services -- the business logic between routers and H6 (H7A).

A small set of services holds everything the thin routers delegate to. They
**compose** the existing layers and add no reasoning, detection, tracking, or
persistence logic of their own:

* :class:`VideoService` -- validates and stores uploads (readability via P1-U5
  ingestion, addressed by content hash), and serves the stored file back.
* :class:`VideoLibraryService` -- joins the video, job, and review indices into
  browsable per-video metadata (H11), loading no event, evidence, or overlay.
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
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..contracts import (
    ConfirmedEvent,
    EvidenceManifest,
    ReviewEntry,
    SceneConfig,
    can_transition,
    next_status,
)
from ..contracts.enums import ReviewAction, ReviewStatus, ViolationType
from ..contracts.scene import ZoneType, scene_config_hash
from ..detector.errors import DetectorError
from ..engine import EngineRunResult, FileFrameSource, InferenceEngine, RuleConfig
from ..engine.errors import (
    EngineConfigurationError,
    RunCancelledError,
    UnsupportedRuleError,
)
from ..ingestion.video import VideoIngestionError
from ..persistence import CorruptRecordError, EventStore, ReviewStore, RunNotFoundError, SceneStore
from ..pipeline.errors import SceneConfigurationError
from ..scenes import SceneDraft, build_scene
from .capabilities import supported_violations
from .config import AppConfig
from .engine_provider import EngineProvider
from .errors import (
    AppError,
    DuplicateVideoError,
    EngineUnavailableError,
    EventNotFoundError,
    InvalidConfigurationError,
    InvalidTransitionError,
    JobNotFoundError,
    OverlayNotFoundError,
    PayloadTooLargeError,
    SceneNotFoundError,
    UnsupportedMediaError,
    VideoMediaNotFoundError,
    VideoNotFoundError,
)
from .models import (
    EventListResponse,
    EventSort,
    EventSummary,
    JobStatusResponse,
    MetricsResponse,
    ReviewDecisionRequest,
    ReviewResponse,
    SceneSummary,
    SceneValidationResponse,
    VideoListResponse,
    VideoSort,
    VideoSummary,
)
from .overlay_video import render_job_overlay
from .registry import (
    JobExecutor,
    JobRecord,
    JobStatus,
    JobStore,
    OverlayStatus,
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
            uploaded_at=datetime.now(UTC),
        )
        self._store.add(record)
        return record

    def require(self, video_id: str) -> VideoRecord:
        """Return the record for ``video_id`` or raise :class:`VideoNotFoundError`."""

        record = self._store.get(video_id)
        if record is None:
            raise VideoNotFoundError(f"no uploaded video with id {video_id!r}")
        return record

    def media_path(self, video_id: str) -> Path:
        """The stored source file for a video, for playback (H11).

        The read side of ``store_upload``. Uploads are addressed by content id and
        stored under it, so nothing but this service can turn an id back into a
        path -- which is why a video that was not uploaded in the current browser
        session had no playable source before H11.
        """

        record = self.require(video_id)
        if not record.path.is_file():
            raise VideoMediaNotFoundError(
                f"the stored file for video {video_id!r} is no longer on disk"
            )
        return record.path


class SceneService:
    """Authors, stores, and binds per-video scenes (H12).

    The service that ends the application-wide scene. It owns three things and
    delegates the rest:

    * **authoring** -- delegated wholly to
      :func:`~trafficpulse.scenes.builder.build_scene`, which is deterministic and
      pure, so this service invents no geometry and stamps no provenance;
    * **storage** -- delegated to the content-addressed
      :class:`~trafficpulse.persistence.SceneStore`, so an edit is a new revision
      rather than an overwrite and every historical event's ``scene_config_hash``
      stays resolvable;
    * **binding** -- one call to ``VideoStore.bind_scene``, which persists through
      the same recovery observer every other video change uses.

    Scene *validity* is never decided here: the ``SceneConfig`` contract validates
    its own structure and references, and the shipped rules decide what they can
    reason over (see :mod:`trafficpulse.app.capabilities`). This service only
    reports what they said.
    """

    def __init__(
        self,
        store: SceneStore,
        videos: VideoService,
        video_store: VideoStore,
        *,
        classifier_available: bool = False,
        fallback: SceneConfig | None = None,
    ) -> None:
        self._store = store
        self._videos = videos
        self._video_store = video_store
        self._classifier_available = classifier_available
        # The server's configured scene, used only for videos nobody has
        # calibrated. Retained so an operator with one fixed camera keeps working
        # exactly as before H12; it is a default, no longer the only answer.
        self._fallback = fallback

    # --- reading -------------------------------------------------------------
    def get(self, scene_hash: str) -> SceneConfig:
        """One stored scene revision, verbatim (404 when this repository has none).

        Addressable by any ``scene_config_hash``, including one read off a
        historical ``ConfirmedEvent`` -- which is what makes an event's declared
        provenance something a client can actually fetch and inspect.
        """

        try:
            scene = self._store.get(scene_hash)
        except CorruptRecordError as exc:
            raise SceneNotFoundError(f"scene {scene_hash!r} is stored but unreadable") from exc
        if scene is None:
            raise SceneNotFoundError(f"no stored scene with hash {scene_hash!r}")
        return scene

    def for_video(self, video_id: str) -> SceneConfig | None:
        """The scene processing should use for a video: its own, else the fallback.

        Resolution order is the whole point of H12 -- the video's binding wins, and
        the server's configured scene is consulted only when there is no binding.
        Returns ``None`` when neither exists, which the caller reports as a clean
        "this video has no scene" rather than processing against unrelated geometry.
        """

        record = self._video_store.get(video_id)
        if record is not None and record.scene_hash is not None:
            stored = self._store.get(record.scene_hash)
            if stored is not None:
                return stored
            # The binding survived but its revision did not. Falling through to the
            # server default silently would reason over another camera's geometry.
            _logger.warning(
                "video %s is bound to missing scene %s; treating as uncalibrated",
                video_id,
                record.scene_hash,
            )
        return self._fallback

    def summary_for_video(self, video_id: str) -> SceneSummary:
        """The bound scene's summary, or 404 when the video is not calibrated."""

        record = self._videos.require(video_id)
        if record.scene_hash is None:
            raise SceneNotFoundError(f"video {video_id!r} has no calibrated scene")
        return self.summarise(self.get(record.scene_hash))

    def summarise(self, scene: SceneConfig) -> SceneSummary:
        """Present a scene for browsing (see :class:`SceneSummary`)."""

        return SceneSummary(
            scene_hash=scene_config_hash(scene),
            scene_id=scene.scene.scene_id,
            scene_name=scene.scene.scene_name,
            camera_id=scene.scene.camera_id,
            site_id=scene.scene.site_id,
            status=scene.scene.status,
            calibration_status=scene.calibration.status,
            frame_width=scene.frame.reference_width,
            frame_height=scene.frame.reference_height,
            zone_count=len(scene.zones),
            has_legal_direction=bool(scene.legal_directions),
            has_no_stopping_zone=any(
                zone.enabled and zone.zone_type is ZoneType.NO_STOPPING
                for zone in scene.zones
            ),
            supported_violations=self.supported_violations(scene),
        )

    def supported_violations(self, scene: SceneConfig) -> tuple[ViolationType, ...]:
        return supported_violations(scene, classifier_available=self._classifier_available)

    def violations_for_video(self, video_id: str) -> tuple[ViolationType, ...]:
        """What can be run for a video, given whichever scene resolves for it."""

        scene = self.for_video(video_id)
        return () if scene is None else self.supported_violations(scene)

    # --- writing -------------------------------------------------------------
    def calibrate(self, video_id: str, draft: SceneDraft) -> SceneSummary:
        """Author this video's scene from a draft, store it, and bind it.

        One call because the three steps have no independent meaning: an authored
        scene nobody bound is orphaned data, and a binding to a scene nobody stored
        is dangling. Idempotent by construction -- an unchanged drawing rebuilds
        the same content, hashes to the same address, and rewrites the same bytes.

        The draft's frame size must match the video's decoded dimensions: geometry
        drawn against a different frame would land in the wrong place, and the
        contract's in-bounds check would only catch the cases that overflow.
        """

        video = self._videos.require(video_id)
        if (draft.frame_width, draft.frame_height) != (video.width, video.height):
            raise InvalidConfigurationError(
                f"the drawing is against a {draft.frame_width}x{draft.frame_height} "
                f"frame but video {video_id!r} is {video.width}x{video.height}; "
                "geometry must be drawn in the video's own pixel space"
            )

        scene = self._build(draft, video_id=video_id)
        scene_hash = self._store.put(scene)
        self._video_store.bind_scene(video_id, scene_hash)
        return self.summarise(scene)

    def validate(self, video_id: str, draft: SceneDraft) -> SceneValidationResponse:
        """Check a draft without storing it, reporting what it would unlock.

        The calibration surface's live feedback. A draft that cannot form a valid
        scene comes back with the contract's own messages rather than a 422, so a
        half-finished drawing is a *state* the UI can render, not a request error.
        """

        video = self._videos.require(video_id)
        if (draft.frame_width, draft.frame_height) != (video.width, video.height):
            return SceneValidationResponse(
                valid=False,
                errors=(
                    f"drawing frame {draft.frame_width}x{draft.frame_height} does not "
                    f"match the video's {video.width}x{video.height}",
                ),
            )
        try:
            scene = self._build(draft, video_id=video_id)
        except ValidationError as exc:
            return SceneValidationResponse(valid=False, errors=_validation_messages(exc))
        return SceneValidationResponse(
            valid=True,
            supported_violations=self.supported_violations(scene),
            scene_hash=scene_config_hash(scene),
        )

    def _build(self, draft: SceneDraft, *, video_id: str) -> SceneConfig:
        """Expand a draft, giving the scene a logical id derived from the video.

        The logical ``scene_id`` is stable across edits of the same video's scene
        (the *hash* is what changes), so successive revisions are recognisably the
        same site rather than looking like unrelated scenes.
        """

        return build_scene(draft, scene_id=f"scene-{video_id}")


def _validation_messages(error: ValidationError) -> tuple[str, ...]:
    """Flatten a pydantic error into client-safe lines naming the offending field."""

    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "scene"
        lines.append(f"{location}: {detail['msg']}")
    return tuple(lines)


class VideoLibraryService:
    """Browsable metadata for every stored video -- the historical library (H11).

    A **join over the existing registries**, holding no state and adding no
    persistence of its own: :class:`~trafficpulse.app.registry.VideoStore` supplies
    the uploads (rebuilt from disk by H10 recovery), ``JobStore`` supplies each
    video's runs and its event index, and ``ReviewStore`` supplies review progress.
    A recovered repository therefore lists exactly as a live one does, because both
    are read through the same indices.

    What it deliberately does not do
    --------------------------------
    It never opens a ``ConfirmedEvent``, an ``EvidenceManifest``, an overlay, or a
    review journal. Event counts come from the in-memory event index (filenames,
    per H10), overlay availability from a path already held on the job record, and
    review progress from a single directory listing. Listing a repository of any
    size therefore costs no deserialisation -- the detail endpoints, which already
    exist, are what load a video's actual analysis once one is selected.
    """

    def __init__(
        self,
        videos: VideoStore,
        jobs: JobStore,
        reviews: ReviewStore | None = None,
        scenes: SceneService | None = None,
    ) -> None:
        self._videos = videos
        self._jobs = jobs
        # Optional for the same reason EventService's is: a library can be built
        # without the review layer, and then every video honestly reports that no
        # event has been reviewed.
        self._reviews = reviews
        # Optional likewise: without the scene layer a row reports no calibration
        # and no supported violations, which is the truthful answer for a library
        # built without it rather than a guess.
        self._scenes = scenes

    def list(self, *, limit: int, offset: int, sort: VideoSort) -> VideoListResponse:
        """Return a deterministic page of video summaries."""

        reviewed = self._reviewed_ids()
        summaries = [
            self._summarise(record, reviewed) for record in self._videos.videos()
        ]
        ordered = _sorted_videos(summaries, sort)
        return VideoListResponse(
            items=tuple(ordered[offset : offset + limit]),
            total=len(ordered),
            limit=limit,
            offset=offset,
        )

    def get(self, video_id: str) -> VideoSummary:
        """One video's library summary, or raise :class:`VideoNotFoundError`."""

        record = self._videos.get(video_id)
        if record is None:
            raise VideoNotFoundError(f"no uploaded video with id {video_id!r}")
        return self._summarise(record, self._reviewed_ids())

    def _reviewed_ids(self) -> frozenset[str]:
        return self._reviews.reviewed_event_ids() if self._reviews is not None else frozenset()

    def _summarise(self, record: VideoRecord, reviewed: frozenset[str]) -> VideoSummary:
        jobs = self._jobs.for_video(record.video_id)
        opening = _opening_job(jobs)
        # Deduplicated across runs, matching what GET /api/events?video_id= returns:
        # reprocessing a video produces the same content-derived event ids, so
        # summing per-run counts would report the same violation several times.
        event_ids = {
            event_id
            for job in jobs
            if job.status is JobStatus.SUCCEEDED
            for event_id in job.event_ids
        }
        return VideoSummary(
            video_id=record.video_id,
            filename=record.filename,
            uploaded_at=record.uploaded_at,
            size_bytes=record.size_bytes,
            width=record.width,
            height=record.height,
            fps=record.fps,
            duration_seconds=record.duration_seconds,
            codec=record.codec,
            job_id=opening.job_id if opening is not None else None,
            status=opening.status if opening is not None else None,
            job_count=len(jobs),
            event_count=len(event_ids),
            events_reviewed=len(event_ids & reviewed),
            overlay_available=(
                opening is not None
                and opening.overlay_video is not None
                and opening.overlay_video.exists()
            ),
            media_available=record.path.is_file(),
            scene_hash=record.scene_hash,
            supported_violations=(
                self._scenes.violations_for_video(record.video_id)
                if self._scenes is not None
                else ()
            ),
        )


# --- processing ----------------------------------------------------------------
class ProcessingService:
    """Creates and drives processing jobs over the injected H6 engine."""

    def __init__(
        self,
        *,
        config: AppConfig,
        scenes: SceneService,
        provider: EngineProvider,
        store: EventStore,
        job_store: JobStore,
        executor: JobExecutor,
        videos: VideoService,
        job_id_factory: JobIdFactory = _default_job_id,
    ) -> None:
        self._config = config
        # H12: the scene is resolved *per video* at submit time, not held here.
        # A service that owns one scene is exactly the singleton this milestone
        # removed -- it made every upload reason against one camera's geometry.
        self._scenes = scenes
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

        The scene comes from the **video** (its calibration, else the server's
        configured fallback), so two videos from different cameras are reasoned
        over their own geometry in the same process.

        Validation is eager: the engine is *built* here (so an invalid scene/rule
        combination or an unavailable backend fails as a clean HTTP error) before
        the job is scheduled. Only the actual inference runs on the executor.
        """

        video = self._videos.require(video_id)
        scene = self._scenes.for_video(video_id)
        if scene is None:
            raise InvalidConfigurationError(
                f"video {video_id!r} has no calibrated scene and the server has no "
                "default scene configured; calibrate the video before processing it"
            )
        resolved = rules if rules is not None else self._config.default_rules
        if not resolved:
            raise InvalidConfigurationError(
                "no rules were specified and the server has no default rule set"
            )

        engine = self._build_engine(scene, resolved)
        job = JobRecord(job_id=self._job_id_factory(), video_id=video_id, engine=engine)
        self._jobs.add(job)
        self._executor.submit(lambda: self._run(job, video, scene))
        return job

    def _build_engine(
        self, scene: SceneConfig, rules: tuple[RuleConfig, ...]
    ) -> InferenceEngine:
        try:
            return self._provider.create(scene=scene, rules=rules)
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
        """Execute one job: run the H6 engine and persist its output.

        The run is cooperatively cancellable: the engine checks the job's cancel
        flag between frames, and a :class:`RunCancelledError` is recorded as a
        clean ``CANCELLED`` outcome (not a failure) with nothing persisted.
        """

        assert job.engine is not None
        try:
            source = FileFrameSource(video.path, camera_id=scene.scene.camera_id)
            self._jobs.mark_running(job.job_id, frames_total=source.metadata.frame_count)
            result = job.engine.run(
                source, should_cancel=lambda: self._jobs.is_cancel_requested(job.job_id)
            )
            job.engine.persist(result, store=self._store, run_id=job.job_id)
            # Declare the overlay pending BEFORE the job goes terminal. A client
            # stops polling once it sees a terminal job status, so the very first
            # succeeded status it reads must already announce that an annotated
            # video is on its way -- otherwise the render (a second decode+encode
            # pass, often longer than inference) lands after the client has stopped
            # listening and the workspace plays the raw upload forever.
            self._jobs.mark_overlay_pending(job.job_id)
            self._jobs.mark_succeeded(job.job_id, result)
            self._render_overlay_video(job, video, scene, result)
        except RunCancelledError:
            _logger.info("processing job %s cancelled", job.job_id)
            self._jobs.mark_cancelled(job.job_id)
        except Exception as exc:  # noqa: BLE001 - a job thread must never crash silently
            _logger.exception("processing job %s failed", job.job_id)
            self._jobs.mark_failed(job.job_id, str(exc))

    def _render_overlay_video(
        self, job: JobRecord, video: VideoRecord, scene: SceneConfig, result: EngineRunResult
    ) -> None:
        """Render the annotated overlay video for a finished job (best-effort).

        Presentation-only: the events + evidence are already persisted before this
        runs, so any render failure (e.g. Pillow absent, an encode error) is logged
        and swallowed -- the job stays ``succeeded``, the original video still plays,
        and every read endpoint still works. The original upload is never modified;
        the annotated video is a separate artifact under ``overlays_dir``.

        Whatever happens, the job's :class:`~trafficpulse.app.registry.OverlayStatus`
        is left **terminal** (``READY`` / ``NONE`` / ``FAILED``): a client polling for
        a pending overlay must always be released, including on the failure paths.
        """

        assert job.engine is not None
        try:
            output = self._config.overlays_dir / f"{job.job_id}.mp4"
            rendered = render_job_overlay(
                engine=job.engine,
                source_path=video.path,
                output_path=output,
                events=result.events,
                camera_id=scene.scene.camera_id,
            )
            if rendered is not None:
                self._jobs.set_overlay_video(job.job_id, rendered.output_path)
            else:
                # The run produced no overlay metadata: nothing to draw, not a fault.
                self._jobs.resolve_overlay(job.job_id, OverlayStatus.NONE)
        except Exception:  # noqa: BLE001 - overlay is a presentation aid, never fatal
            self._jobs.resolve_overlay(job.job_id, OverlayStatus.FAILED)
            _logger.exception(
                "overlay render failed for job %s; original video still plays", job.job_id
            )

    def overlay_video_path(self, job_id: str) -> Path:
        """The rendered overlay video for a job, or a 404 when none is available."""

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")
        if job.overlay_video is None or not job.overlay_video.exists():
            raise OverlayNotFoundError(f"no overlay video is available for job {job_id!r}")
        return job.overlay_video

    def cancel(self, job_id: str) -> JobStatusResponse:
        """Request cancellation of a job and return its current status.

        Cancellation is cooperative and asynchronous: for a running job this
        flags the background run, which stops at the next frame and transitions
        to ``cancelled`` (the client keeps polling until it observes that).
        Cancelling an already-finished job is a safe no-op that returns its
        existing status. An unknown job id is a 404.
        """

        if self._jobs.get(job_id) is None:
            raise JobNotFoundError(f"no processing job with id {job_id!r}")
        self._jobs.request_cancel(job_id)
        return self.status(job_id)

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
            overlay_available=job.overlay_video is not None and job.overlay_video.exists(),
            overlay_status=job.overlay_status,
        )


# --- events + evidence ---------------------------------------------------------
class EventService:
    """Reads persisted confirmed events back from the H6 ``EventStore``."""

    def __init__(
        self,
        store: EventStore,
        job_store: JobStore,
        reviews: ReviewStore | None = None,
    ) -> None:
        self._store = store
        self._jobs = job_store
        # Optional so an EventService can still be built without the review layer
        # (older tests, and any caller that only reads events). Absent it, every
        # summary reports the honest default of PENDING.
        self._reviews = reviews

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
                        start_at=event.start_at,
                        trigger_at=event.trigger_at,
                        rule_id=event.rule_id,
                        # Verbatim: the components are the contract's, un-merged.
                        confidence=event.confidence,
                    )
                )

        ordered = _sorted_summaries(summaries, sort)
        page = ordered[offset : offset + limit]
        # Fold review status only for the page actually returned: badging is a
        # per-event journal read, so doing it for the whole result set would make
        # paging cost more the deeper you go for data nobody asked for.
        if self._reviews is not None and page:
            statuses = self._reviews.statuses(summary.event_id for summary in page)
            page = [
                summary.model_copy(
                    update={
                        "review_status": statuses.get(
                            summary.event_id, ReviewStatus.PENDING
                        )
                    }
                )
                for summary in page
            ]
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


class ReviewService:
    """Records analyst decisions and serves the review case + audit history (H9).

    The application-layer guard on the review lifecycle. It owns exactly three
    responsibilities and delegates everything else:

    * the event must exist (delegated to :class:`EventService`, so a decision can
      never be recorded against an id the system cannot produce evidence for);
    * the transition must be legal (delegated to the contract's pure state
      machine, so the API and any future caller enforce one rule set);
    * the action is **appended**, never written over (delegated to
      :class:`~trafficpulse.persistence.review_store.ReviewStore`).

    It holds no state. The current status is always re-folded from the journal, so
    two concurrent reviewers cannot leave the service caching a decision that disk
    disagrees with.
    """

    def __init__(self, events: EventService, store: ReviewStore) -> None:
        self._events = events
        self._store = store

    def get(self, event_id: str) -> ReviewResponse:
        """The event's current case and its full history (raises if unknown)."""

        _, manifest = self._events.locate(event_id)
        return ReviewResponse(
            case=self._store.case(
                event_id, evidence_package_id=manifest.evidence_package_id
            ),
            history=self._store.history(event_id),
        )

    def decide(self, event_id: str, request: ReviewDecisionRequest) -> ReviewResponse:
        """Record one analyst action and return the resulting case + history.

        Raises :class:`~trafficpulse.app.errors.EventNotFoundError` for an unknown
        event and :class:`~trafficpulse.app.errors.InvalidTransitionError` when the
        action is not legal from the current status -- the latter is a 409, not a
        422, because the request is well-formed and it is the *state* that refuses
        it. Re-sending a decision after somebody else has decided therefore fails
        loudly instead of silently overwriting their call.
        """

        _, manifest = self._events.locate(event_id)
        current = self._store.status(event_id)
        action = request.action
        if not can_transition(current, action):
            raise InvalidTransitionError(
                f"cannot {action.value!r} an event that is {current.value!r}"
            )

        resulting = next_status(current, action)
        at = datetime.now(UTC)
        note = request.note.strip() if request.note else None
        reason = request.reason.strip() if request.reason else None
        entry = ReviewEntry(
            entry_id=self._entry_id(event_id, action, at),
            event_id=event_id,
            action=action,
            status_before=current,
            status_after=resulting,
            reviewer=request.reviewer,
            at=at,
            note=note or None,
            reason=reason or None,
        )
        self._store.append(entry)
        return ReviewResponse(
            case=self._store.case(
                event_id, evidence_package_id=manifest.evidence_package_id
            ),
            history=self._store.history(event_id),
        )

    def statuses_for(self, event_ids: Iterable[str]) -> dict[str, ReviewStatus]:
        """Review statuses for many events, for badging a list."""

        return self._store.statuses(event_ids)

    @staticmethod
    def _entry_id(event_id: str, action: ReviewAction, at: datetime) -> str:
        """A content-derived id for one journal entry.

        Follows the project's existing identity convention (a SHA-256 over the
        identity-bearing fields, as ``event_id`` uses) rather than a random uuid, so
        an entry's id is reproducible from the entry itself.
        """

        material = f"{event_id}|{action.value}|{at.isoformat()}"
        return "rev-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


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
            jobs_cancelled=by_status[JobStatus.CANCELLED],
            events_total=events_total,
            latest=latest,
        )


def _opening_job(jobs: Sequence[JobRecord]) -> JobRecord | None:
    """The run a client should open for a video: its analysis, not its last attempt.

    The most recent **succeeded** run wins, because that is the one that has events,
    evidence, and possibly an overlay -- the thing "reopen this video" means. Only
    when no run ever succeeded does the most recent run of any status stand in, so a
    video whose processing failed still reports why instead of looking unprocessed.
    Jobs arrive in insertion order, so "most recent" is the last match.
    """

    succeeded = [job for job in jobs if job.status is JobStatus.SUCCEEDED]
    if succeeded:
        return succeeded[-1]
    return jobs[-1] if jobs else None


def _sorted_videos(summaries: list[VideoSummary], sort: VideoSort) -> list[VideoSummary]:
    """Deterministically order video summaries (video_id is the final tie-break).

    A video with no recorded upload instant cannot be placed on the time axis, so
    it is sorted to the end of *either* direction rather than being given a
    substitute date -- the ordering says "unknown", which is what is true.
    """

    if sort in (VideoSort.FILENAME_ASC, VideoSort.FILENAME_DESC):
        return sorted(
            summaries,
            key=lambda s: (s.filename.lower(), s.video_id),
            reverse=sort is VideoSort.FILENAME_DESC,
        )
    descending = sort is VideoSort.UPLOADED_AT_DESC
    return sorted(
        summaries,
        key=lambda s: (
            s.uploaded_at is None,
            _sort_instant(s.uploaded_at, descending=descending),
            s.video_id,
        ),
    )


def _sort_instant(value: datetime | None, *, descending: bool) -> float:
    """A comparable key for an optional instant, negated for descending order.

    Negating the timestamp (rather than reversing the whole sort) keeps the
    "unknown last" flag and the ``video_id`` tie-break ascending in both
    directions, so the ordering stays stable and unknowns never lead the page.
    """

    if value is None:
        return 0.0
    seconds = value.timestamp()
    return -seconds if descending else seconds


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
    "VideoLibraryService",
    "SceneService",
    "ProcessingService",
    "EventService",
    "EvidenceService",
    "MetricsService",
    "AppError",
]

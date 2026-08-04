"""The FastAPI application factory + error handling (H7A).

:func:`create_app` is the composition root of the HTTP layer: it wires the five
services into a frozen :class:`AppContext` on ``app.state``, registers the
resource routers, installs the uniform error handlers, and stamps the OpenAPI
metadata. It takes injectable seams -- the engine provider, the job executor, and
a video/job store -- so the same app runs with the real RT-DETR backend in
production and with stub-injected engines in tests, unchanged.

Error handling is centralised here: every :class:`AppError` becomes the consistent
``{"error": {"type", "message"}}`` envelope at its declared status, request
validation errors become a 422 in the same shape, and any unexpected exception
becomes a generic 500 -- no traceback, internal path, or framework detail ever
reaches a client.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from .. import __version__
from .analytics import AnalyticsService
from .config import AppConfig, load_scene
from .dependencies import AppContext
from .engine_provider import EngineProvider, RealEngineProvider
from .errors import AppError
from .logging_config import RequestIdMiddleware, configure_logging
from .models import ErrorDetail, ErrorResponse
from .recovery import RepositoryRecovery
from .registry import JobExecutor, JobStore, ThreadJobExecutor, VideoStore
from .routers import (
    analytics,
    events,
    evidence,
    health,
    metrics,
    process,
    scenes,
    upload,
    videos,
)
from .services import (
    EventService,
    EvidenceService,
    MetricsService,
    ProcessingService,
    ReviewService,
    SceneService,
    VideoLibraryService,
    VideoService,
)

_logger = logging.getLogger("trafficpulse.app")

_DESCRIPTION = (
    "HTTP API exposing the TrafficPulse real-time inference engine: upload a "
    "video, start a processing job, and retrieve confirmed events, their "
    "evidence manifests, and engine metrics. The engine, detector, tracker, and "
    "rules stay server-side; clients depend only on this JSON contract."
)


def _error_response(
    status_code: int, error_type: str, message: str, *, video_id: str | None = None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(type=error_type, message=message, video_id=video_id))
    # exclude_none keeps the wire envelope exactly {type, message} for every error
    # except a duplicate-video conflict, which adds video_id -- so the additive
    # field is invisible to existing clients.
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def _register_error_handlers(app: FastAPI) -> None:
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        # A duplicate-video conflict carries the existing id so a client can
        # recover by opening it; other errors have no such payload.
        video_id = getattr(exc, "video_id", None)
        return _error_response(exc.status_code, exc.error_type, exc.message, video_id=video_id)

    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Compact, client-safe summary of the first validation problem.
        errors = exc.errors()
        location = ".".join(str(part) for part in errors[0]["loc"]) if errors else "request"
        message = errors[0]["msg"] if errors else "request validation failed"
        return _error_response(422, "validation_error", f"{location}: {message}")

    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        _logger.exception("unhandled error serving a request")
        return _error_response(500, "internal_error", "an internal error occurred")

    # Typed to the base classes; FastAPI dispatches subclasses to these handlers.
    app.add_exception_handler(AppError, handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unexpected)


class _SpaStaticFiles(StaticFiles):
    """Static files with an SPA fallback: a 404 serves ``index.html`` (H8).

    So a client-side route (e.g. a refresh on ``/videos``) loads the app shell
    and lets the router resolve the path, instead of returning a bare 404. The
    API routers are registered before this mount, so ``/api/*`` never reaches it.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve a built SPA from ``static_dir`` at ``/`` (mounted last)."""

    app.mount("/", _SpaStaticFiles(directory=static_dir, html=True), name="spa")


def _build_context(
    config: AppConfig,
    *,
    provider: EngineProvider,
    executor: JobExecutor,
) -> AppContext:
    from ..contracts import SceneConfig
    from ..evidence import ArtifactStore
    from ..persistence import EventStore, RenderedArtifactStore, ReviewStore, SceneStore

    # The file-configured scene is no longer *the* scene (H12): it is the fallback
    # for videos nobody has calibrated, so a single-camera deployment keeps working
    # unchanged while per-video calibration takes precedence.
    scene = load_scene(config.scene_path) if config.scene_path is not None else None
    assert scene is None or isinstance(scene, SceneConfig)  # load_scene is typed object

    # Startup recovery (H10). The registries are created with the recovery
    # observer attached, so every later state change persists itself, and are then
    # rebuilt from disk *before* anything is served -- no request can observe a
    # half-populated index. Recovery never raises: an unreadable repository
    # degrades to a smaller one rather than preventing startup.
    recovery = RepositoryRecovery(config)
    video_store = VideoStore(on_change=recovery.snapshot_video)
    job_store = JobStore(on_change=recovery.snapshot_job)
    recovery.recover(videos=video_store, jobs=job_store)

    # Named `video_service`, not `videos`: the videos *router* is imported under
    # that name at module scope, and shadowing it here is one edit away from a bug.
    video_service = VideoService(config, video_store)
    event_store = EventStore(config.runs_dir)
    # The review journal shares the runtime root with the event store but writes to
    # its own `reviews/` subtree -- it can never touch a write-once event record.
    review_store = ReviewStore(config.runs_dir)
    # H14 rendering stores. The sidecar shares the runtime root with the event store
    # but writes to its own `rendered/` subtree, so like the review journal it can
    # never touch a write-once record. Artifact bytes are content-addressed and live
    # outside the per-run tree entirely, because two runs that render the same frame
    # share one file.
    rendered_store = RenderedArtifactStore(config.runs_dir)
    artifact_store = ArtifactStore(config.artifacts_dir)
    scene_service = SceneService(
        SceneStore(config.storage_dir),
        video_service,
        video_store,
        # AppConfig is the production authority on whether a helmet classifier
        # exists: RealEngineProvider builds one from exactly this field, so a
        # deployment without it cannot run no_helmet whatever a scene declares.
        classifier_available=config.helmet_classifier is not None,
        fallback=scene,
    )
    processing = ProcessingService(
        config=config,
        scenes=scene_service,
        provider=provider,
        store=event_store,
        job_store=job_store,
        executor=executor,
        videos=video_service,
        artifacts=artifact_store,
        rendered=rendered_store,
    )
    event_service = EventService(event_store, job_store, review_store)
    return AppContext(
        config=config,
        provider=provider,
        videos=video_service,
        scenes=scene_service,
        # The library reads the same registries recovery just rebuilt, which is what
        # makes a recovered video indistinguishable from a freshly uploaded one.
        library=VideoLibraryService(video_store, job_store, review_store, scene_service),
        processing=processing,
        events=event_service,
        evidence=EvidenceService(event_service, rendered_store, artifact_store),
        reviews=ReviewService(event_service, review_store),
        metrics=MetricsService(job_store),
        # The single aggregation layer (H15). It composes the same registries every
        # other service reads -- it owns no storage and duplicates no scan.
        analytics=AnalyticsService(
            videos=video_store,
            jobs=job_store,
            provider=provider,
            reviews=review_store,
            rendered=rendered_store,
            artifacts=artifact_store,
            overlays_dir=config.overlays_dir,
        ),
    )


def create_app(
    config: AppConfig,
    *,
    engine_provider: EngineProvider | None = None,
    executor: JobExecutor | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    """Build a fully-wired FastAPI application for ``config``.

    ``engine_provider`` defaults to the production :class:`RealEngineProvider`
    (real RT-DETR, built lazily per job); ``executor`` defaults to the background
    :class:`ThreadJobExecutor`. Tests inject a stub provider and the synchronous
    executor to get a deterministic, GPU-free lifecycle.

    ``configure_logs`` installs the package's logging configuration (H16). It is on
    by default because this function is the application's composition root and an
    unconfigured application drops every ``INFO`` -- including the startup recovery
    report. An embedder that owns its own logging passes ``False``.
    """

    if configure_logs:
        level = configure_logging(config.log_level)
        _logger.info("logging configured at %s", level)

    app = FastAPI(
        title="TrafficPulse API",
        version=__version__,
        description=_DESCRIPTION,
    )
    provider = engine_provider if engine_provider is not None else RealEngineProvider(config)
    app.state.context = _build_context(
        config,
        provider=provider,
        executor=executor if executor is not None else ThreadJobExecutor(),
    )
    _register_error_handlers(app)

    # Outermost middleware: every request (including ones that fail) gets a
    # correlation id, and it is set before any handler or other middleware runs.
    # Adds a response *header* only -- no response model changes.
    app.add_middleware(RequestIdMiddleware)

    # CORS is opt-in: added only when explicit origins are configured, so the
    # default same-origin / dev-proxy deployment carries no CORS surface.
    if config.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_allow_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for router in (
        health,
        upload,
        videos,
        scenes,
        process,
        events,
        evidence,
        metrics,
        analytics,
    ):
        app.include_router(router.router)

    # The SPA mount is registered last so every /api route matches first; it is
    # opt-in and only mounted when the built assets actually exist on disk.
    if config.static_dir is not None and config.static_dir.is_dir():
        _mount_spa(app, config.static_dir)

    return app

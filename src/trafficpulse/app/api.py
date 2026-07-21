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

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from .config import AppConfig, load_scene
from .dependencies import AppContext
from .engine_provider import EngineProvider, RealEngineProvider
from .errors import AppError
from .models import ErrorDetail, ErrorResponse
from .registry import JobExecutor, JobStore, ThreadJobExecutor, VideoStore
from .routers import events, evidence, health, metrics, process, upload
from .services import (
    EventService,
    EvidenceService,
    MetricsService,
    ProcessingService,
    VideoService,
)

_logger = logging.getLogger("trafficpulse.app")

_DESCRIPTION = (
    "HTTP API exposing the TrafficPulse real-time inference engine: upload a "
    "video, start a processing job, and retrieve confirmed events, their "
    "evidence manifests, and engine metrics. The engine, detector, tracker, and "
    "rules stay server-side; clients depend only on this JSON contract."
)


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(type=error_type, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _register_error_handlers(app: FastAPI) -> None:
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.error_type, exc.message)

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


def _build_context(
    config: AppConfig,
    *,
    provider: EngineProvider,
    executor: JobExecutor,
    video_store: VideoStore,
    job_store: JobStore,
) -> AppContext:
    from ..contracts import SceneConfig
    from ..persistence import EventStore

    scene = load_scene(config.scene_path) if config.scene_path is not None else None
    assert scene is None or isinstance(scene, SceneConfig)  # load_scene is typed object

    videos = VideoService(config, video_store)
    event_store = EventStore(config.runs_dir)
    processing = ProcessingService(
        config=config,
        scene=scene,
        provider=provider,
        store=event_store,
        job_store=job_store,
        executor=executor,
        videos=videos,
    )
    event_service = EventService(event_store, job_store)
    return AppContext(
        config=config,
        provider=provider,
        videos=videos,
        processing=processing,
        events=event_service,
        evidence=EvidenceService(event_service),
        metrics=MetricsService(job_store),
    )


def create_app(
    config: AppConfig,
    *,
    engine_provider: EngineProvider | None = None,
    executor: JobExecutor | None = None,
) -> FastAPI:
    """Build a fully-wired FastAPI application for ``config``.

    ``engine_provider`` defaults to the production :class:`RealEngineProvider`
    (real RT-DETR, built lazily per job); ``executor`` defaults to the background
    :class:`ThreadJobExecutor`. Tests inject a stub provider and the synchronous
    executor to get a deterministic, GPU-free lifecycle.
    """

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
        video_store=VideoStore(),
        job_store=JobStore(),
    )
    _register_error_handlers(app)
    for router in (health, upload, process, events, evidence, metrics):
        app.include_router(router.router)
    return app

"""FastAPI dependency wiring -- no module globals (H7A).

The application's services are constructed once in :func:`create_app` and stored
on ``app.state`` inside a single frozen :class:`AppContext`. Every dependency
here reads that context off the incoming request, so routers receive their
services through ``Depends`` and nothing is a module-level singleton. Swapping the
engine provider, executor, or storage for a test is therefore just building a
different context -- no monkeypatching of globals.

The ``Annotated[..., Depends(...)]`` aliases are the types routers actually
declare, which keeps handler signatures short and the OpenAPI schema clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, WebSocket

from .analytics import AnalyticsService
from .config import AppConfig
from .engine_provider import EngineProvider
from .live import LiveSessionManager
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


@dataclass(frozen=True)
class AppContext:
    """The fully-wired application services for one running app."""

    config: AppConfig
    provider: EngineProvider
    videos: VideoService
    scenes: SceneService
    library: VideoLibraryService
    processing: ProcessingService
    events: EventService
    evidence: EvidenceService
    reviews: ReviewService
    metrics: MetricsService
    analytics: AnalyticsService
    live: LiveSessionManager


def get_context(request: Request) -> AppContext:
    """Read the wired context off ``app.state`` (typed, never Any-returning)."""

    context: AppContext = request.app.state.context
    return context


def get_config(request: Request) -> AppConfig:
    return get_context(request).config


def get_provider(request: Request) -> EngineProvider:
    return get_context(request).provider


def get_video_service(request: Request) -> VideoService:
    return get_context(request).videos


def get_scene_service(request: Request) -> SceneService:
    return get_context(request).scenes


def get_video_library_service(request: Request) -> VideoLibraryService:
    return get_context(request).library


def get_processing_service(request: Request) -> ProcessingService:
    return get_context(request).processing


def get_event_service(request: Request) -> EventService:
    return get_context(request).events


def get_evidence_service(request: Request) -> EvidenceService:
    return get_context(request).evidence


def get_review_service(request: Request) -> ReviewService:
    return get_context(request).reviews


def get_metrics_service(request: Request) -> MetricsService:
    return get_context(request).metrics


def get_analytics_service(request: Request) -> AnalyticsService:
    return get_context(request).analytics


def get_live_manager(request: Request) -> LiveSessionManager:
    return get_context(request).live


def get_live_manager_ws(websocket: WebSocket) -> LiveSessionManager:
    """The live manager for a WebSocket handler.

    A separate accessor because a WebSocket connection is not an HTTP ``Request``
    and FastAPI will not inject one into a socket route. It reads the very same
    ``app.state`` context every HTTP dependency reads, so a socket and a request in
    the same process are looking at one set of services -- there is no second
    registry a session could get lost in.
    """

    context: AppContext = websocket.app.state.context
    return context.live


ConfigDep = Annotated[AppConfig, Depends(get_config)]
ProviderDep = Annotated[EngineProvider, Depends(get_provider)]
VideoServiceDep = Annotated[VideoService, Depends(get_video_service)]
SceneServiceDep = Annotated[SceneService, Depends(get_scene_service)]
VideoLibraryServiceDep = Annotated[VideoLibraryService, Depends(get_video_library_service)]
ProcessingServiceDep = Annotated[ProcessingService, Depends(get_processing_service)]
EventServiceDep = Annotated[EventService, Depends(get_event_service)]
EvidenceServiceDep = Annotated[EvidenceService, Depends(get_evidence_service)]
ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
LiveManagerDep = Annotated[LiveSessionManager, Depends(get_live_manager)]
LiveManagerWsDep = Annotated[LiveSessionManager, Depends(get_live_manager_ws)]

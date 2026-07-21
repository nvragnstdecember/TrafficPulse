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

from fastapi import Depends, Request

from .config import AppConfig
from .engine_provider import EngineProvider
from .services import (
    EventService,
    EvidenceService,
    MetricsService,
    ProcessingService,
    VideoService,
)


@dataclass(frozen=True)
class AppContext:
    """The fully-wired application services for one running app."""

    config: AppConfig
    provider: EngineProvider
    videos: VideoService
    processing: ProcessingService
    events: EventService
    evidence: EvidenceService
    metrics: MetricsService


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


def get_processing_service(request: Request) -> ProcessingService:
    return get_context(request).processing


def get_event_service(request: Request) -> EventService:
    return get_context(request).events


def get_evidence_service(request: Request) -> EvidenceService:
    return get_context(request).evidence


def get_metrics_service(request: Request) -> MetricsService:
    return get_context(request).metrics


ConfigDep = Annotated[AppConfig, Depends(get_config)]
ProviderDep = Annotated[EngineProvider, Depends(get_provider)]
VideoServiceDep = Annotated[VideoService, Depends(get_video_service)]
ProcessingServiceDep = Annotated[ProcessingService, Depends(get_processing_service)]
EventServiceDep = Annotated[EventService, Depends(get_event_service)]
EvidenceServiceDep = Annotated[EvidenceService, Depends(get_evidence_service)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]

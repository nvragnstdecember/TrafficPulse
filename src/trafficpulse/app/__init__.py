"""TrafficPulse application API layer (H7A).

A thin FastAPI HTTP surface over the H6 real-time inference engine. Clients
upload a video, start a processing job, and read back confirmed events, evidence
manifests, and engine metrics -- over JSON only. The engine, detector, tracker,
rules, and event store all stay behind this boundary: nothing a client touches
names an engine class.

Layering
--------
``routers`` (thin HTTP handlers) -> ``services`` (business logic composing H6) ->
H6 ``engine`` + ``persistence``. Configuration, the injectable engine provider,
the in-memory registries, and the error taxonomy live in their own modules.
FastAPI is an optional extra (``trafficpulse[api]``); importing the base package
pulls in no web framework, and building the app loads no ML model.

Public surface: :func:`~trafficpulse.app.api.create_app` (the application
factory) and :class:`~trafficpulse.app.config.AppConfig`. The ASGI application is
``trafficpulse.app.asgi:app``.
"""

from __future__ import annotations

from .api import create_app
from .config import AppConfig
from .engine_provider import EngineProvider, RealEngineProvider
from .errors import (
    AppError,
    BadRequestError,
    DuplicateVideoError,
    EngineUnavailableError,
    EventNotFoundError,
    InvalidConfigurationError,
    JobNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    VideoNotFoundError,
)
from .registry import (
    JobExecutor,
    JobStatus,
    SynchronousJobExecutor,
    ThreadJobExecutor,
)

__all__ = [
    # application factory + config
    "create_app",
    "AppConfig",
    # engine provider seam
    "EngineProvider",
    "RealEngineProvider",
    # execution seam
    "JobExecutor",
    "SynchronousJobExecutor",
    "ThreadJobExecutor",
    "JobStatus",
    # error taxonomy
    "AppError",
    "BadRequestError",
    "UnsupportedMediaError",
    "InvalidConfigurationError",
    "VideoNotFoundError",
    "JobNotFoundError",
    "EventNotFoundError",
    "DuplicateVideoError",
    "PayloadTooLargeError",
    "EngineUnavailableError",
]

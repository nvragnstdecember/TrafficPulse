"""Application logging configuration + request correlation (H16).

Before this module the package emitted log records that nothing was configured to
handle: Python's last-resort handler printed ``WARNING`` and above to stderr with
no timestamp, no level and no logger name, and **dropped every ``INFO``** -- which
included the startup recovery report, the one line that tells an operator what the
repository actually contained. Configuring logging is the difference between a
system you can operate and one you can only guess at.

What this module does, and deliberately does not do
---------------------------------------------------
It configures the ``trafficpulse`` logger hierarchy **once**, from the application
composition root. It does not touch the root logger's level, does not call
``basicConfig``, and does not reconfigure on import -- a library that hijacks the
host application's logging is worse than one that stays silent. An embedding
application that has already configured logging can simply not call
:func:`configure_logging`.

Named after the subsystem, not the file
---------------------------------------
Every logger is ``trafficpulse.<subsystem>`` (:data:`LOGGERS`), so an operator can
raise or lower one area -- say ``trafficpulse.evidence`` during a rendering
investigation -- without drowning in the rest. That only works if the names are
consistent, which is why they are declared here rather than spelled out ad hoc at
each call site.

Request correlation
-------------------
:class:`RequestIdMiddleware` assigns every HTTP request an id, exposes it through a
:class:`~contextvars.ContextVar`, and injects it into every log record emitted
while that request is being served (via :class:`RequestIdFilter`). Concurrent job
failures are otherwise impossible to untangle in a shared log. The id is echoed in
the ``X-Request-ID`` response header -- a header, never a body field, so **no API
response model changes**.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import IO, Any, Final

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The subsystem logger names. Every module logs through exactly one of these, so
#: per-area log levels are a supported operation rather than an accident.
LOGGERS: Final[tuple[str, ...]] = (
    "trafficpulse.app",
    "trafficpulse.analytics",
    "trafficpulse.engine",
    "trafficpulse.evidence",
    "trafficpulse.recovery",
)

#: Root of the package's logger hierarchy; configuring it covers every subsystem.
ROOT_LOGGER: Final[str] = "trafficpulse"

#: Header carrying the correlation id back to the client.
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

DEFAULT_LOG_LEVEL: Final[str] = "INFO"

_VALID_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)-24s [%(request_id)s] %(message)s"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

#: The correlation id of the request being served on this task, if any.
_request_id: ContextVar[str] = ContextVar("trafficpulse_request_id", default="-")


def current_request_id() -> str:
    """The correlation id of the in-flight request, or ``"-"`` outside one.

    Background job threads legitimately run outside any request, so the absence of
    an id is normal and is rendered as a dash rather than treated as an error.
    """

    return _request_id.get()


def normalize_log_level(value: str | None, *, default: str = DEFAULT_LOG_LEVEL) -> str:
    """Coerce an operator-supplied level name to a valid one.

    Case-insensitive, and an unrecognised value falls back to ``default`` rather
    than raising: a typo in a deployment variable should not prevent the process
    from starting, and the resulting log line says which level was actually used.
    """

    if value is None:
        return default
    candidate = value.strip().upper()
    return candidate if candidate in _VALID_LEVELS else default


class RequestIdFilter(logging.Filter):
    """Attaches the current request id to every record.

    A :class:`~logging.Filter` rather than a custom adapter so **every** logger in
    the hierarchy carries the field without any call site opting in -- including
    third-party records routed through our handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


def configure_logging(level: str | None = None, *, stream: IO[str] | None = None) -> str:
    """Configure the ``trafficpulse`` logger hierarchy; return the level applied.

    Idempotent: repeated calls replace the package handler rather than stacking
    duplicates, so a test that builds several applications does not multiply its
    own output. ``propagate`` is disabled so records do not also reach a host
    application's root handler and appear twice.
    """

    resolved = normalize_log_level(
        level if level is not None else os.environ.get("TRAFFICPULSE_APP_LOG_LEVEL")
    )

    handler: logging.Handler = logging.StreamHandler(
        stream if stream is not None else sys.stderr
    )
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_TIME_FORMAT))
    handler.addFilter(RequestIdFilter())

    package = logging.getLogger(ROOT_LOGGER)
    for existing in list(package.handlers):
        package.removeHandler(existing)
    package.addHandler(handler)
    package.setLevel(resolved)
    package.propagate = False
    return resolved


class RequestIdMiddleware:
    """Assigns and propagates a correlation id for every HTTP request.

    A raw ASGI middleware rather than a ``BaseHTTPMiddleware`` subclass: the latter
    wraps every request in an extra task group, which is real overhead on a path
    that also streams video files. This one sets a context variable, appends one
    response header, and gets out of the way.

    An inbound ``X-Request-ID`` is honoured so an id assigned by a reverse proxy
    survives into these logs; otherwise a short random one is minted.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        inbound = Request(scope).headers.get(REQUEST_ID_HEADER)
        request_id = inbound if inbound else uuid.uuid4().hex[:12]
        token = _request_id.set(request_id)

        async def send_with_header(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                headers.append(
                    (REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("latin-1"))
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self._app(scope, receive, send_with_header)
        finally:
            _request_id.reset(token)


__all__ = [
    "LOGGERS",
    "ROOT_LOGGER",
    "REQUEST_ID_HEADER",
    "DEFAULT_LOG_LEVEL",
    "configure_logging",
    "current_request_id",
    "normalize_log_level",
    "RequestIdFilter",
    "RequestIdMiddleware",
]

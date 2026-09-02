"""HTTP routers for the application API (H7A).

One router per resource, each deliberately thin: a handler validates inputs via
its typed signature, delegates to exactly one service call, and returns a
response model. No business logic, persistence, or engine access lives here.
"""

from . import (
    analysis,
    events,
    evidence,
    health,
    live,
    metrics,
    process,
    scenes,
    upload,
    videos,
)

__all__ = [
    "analysis",
    "evidence",
    "events",
    "health",
    "live",
    "metrics",
    "process",
    "scenes",
    "upload",
    "videos",
]

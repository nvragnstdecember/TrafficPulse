"""HTTP routers for the application API (H7A).

One router per resource, each deliberately thin: a handler validates inputs via
its typed signature, delegates to exactly one service call, and returns a
response model. No business logic, persistence, or engine access lives here.
"""

from . import (
    events,
    evidence,
    health,
    metrics,
    process,
    upload,
)

__all__ = ["evidence", "events", "health", "metrics", "process", "upload"]

"""Live camera monitoring: a persistent session over the shipped inference engine.

The application's second input mode. Where
:class:`~trafficpulse.app.services.ProcessingService` runs the engine over a stored
file, this package runs the **same** engine over a browser camera stream that has
no file, no known length, and no PTS -- and it does so without a second copy of any
semantics. Detection, tracking, rider association, head crops, helmet
classification, temporal state and every violation reasoner are reached through the
existing :class:`~trafficpulse.app.engine_provider.EngineProvider`, exactly as a
processing job reaches them.

Layering (imports point one way)::

    config / errors / imaging      (pure; Pillow lazily, inside imaging only)
        ^        ^        ^
        |        |        |
    scene  ----> session <---- manager ----> protocol
    (reuses the      (holds one     (owns every    (the WebSocket
     capability       engine)        session)       message shapes)
     probe)

What live mode does **not** do is as deliberate as what it does: it derives no
scene geometry from a camera view, invents no driver attribution for a multi-rider
motorcycle, adds no live-only violation rule, and persists neither camera frames
nor live events. Each of those is documented where it is enforced, and reported to
the client rather than left to be inferred from an empty feed.
"""

from .config import LiveConfig
from .errors import (
    LiveCapacityError,
    LiveError,
    LiveFrameError,
    LiveInferenceError,
    LiveProtocolError,
    LiveUnavailableError,
)
from .manager import LiveReadiness, LiveSessionManager, LiveSessionSummary
from .scene import LiveRulePlan, live_rule_plan, provisional_live_scene
from .session import (
    LiveFrameResult,
    LiveMotorcycleView,
    LiveRiderView,
    LiveSession,
    LiveStats,
    LiveTrackView,
    PendingFrame,
)

__all__ = [
    "LiveConfig",
    "LiveError",
    "LiveProtocolError",
    "LiveFrameError",
    "LiveCapacityError",
    "LiveUnavailableError",
    "LiveInferenceError",
    "LiveSessionManager",
    "LiveReadiness",
    "LiveSessionSummary",
    "LiveSession",
    "LiveFrameResult",
    "LiveStats",
    "LiveTrackView",
    "LiveRiderView",
    "LiveMotorcycleView",
    "PendingFrame",
    "LiveRulePlan",
    "live_rule_plan",
    "provisional_live_scene",
]

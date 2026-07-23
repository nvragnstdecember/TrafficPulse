"""TrafficPulse overlay framework: explainable, production visualization.

The standard visualization layer for TrafficPulse. Inference emits **generic**
:class:`~trafficpulse.overlay.metadata.OverlayScene` metadata (via per-violation
:class:`~trafficpulse.overlay.registry.OverlayProvider`\\ s); a single
:class:`~trafficpulse.overlay.renderer.OverlayRenderer` draws it. The renderer, the
metadata model, and the theme contain **no** violation-specific logic, so a new
violation contributes only its metadata + a provider and inherits the whole look.

Layering (imports point one way)::

    metadata / theme / layout   (pure, Pillow-free, no violation logic)
        ^          ^      ^
        |          |      |
    registry (providers + compositor) -----> renderer (lazy Pillow backend)
        ^
        |
    providers/*  (violation-specific: no_helmet, and future wrong_way, speeding, …)

Only :class:`~trafficpulse.overlay.renderer.PillowOverlayRenderer` touches pixels,
and it imports Pillow lazily -- the base install (pydantic/av/numpy) can build,
serialise, and test scenes without any drawing dependency.
"""

from .metadata import (
    Corner,
    OverlayAlert,
    OverlayBanner,
    OverlayBox,
    OverlayCaption,
    OverlayElement,
    OverlayEmphasis,
    OverlayLayer,
    OverlayLink,
    OverlayPoint,
    OverlayScene,
)
from .registry import (
    OVERLAY_PROVIDERS,
    OverlayCompositor,
    OverlayFrameRef,
    OverlayProvider,
    OverlayProviderRegistry,
)
from .renderer import (
    OverlayBackendUnavailableError,
    OverlayError,
    OverlayRenderer,
    PillowOverlayRenderer,
)
from .theme import DEFAULT_THEME, OverlayTheme

__all__ = [
    # metadata
    "OverlayScene",
    "OverlayElement",
    "OverlayBox",
    "OverlayLink",
    "OverlayBanner",
    "OverlayCaption",
    "OverlayPoint",
    "OverlayEmphasis",
    "OverlayAlert",
    "OverlayLayer",
    "Corner",
    # theme
    "OverlayTheme",
    "DEFAULT_THEME",
    # registry / composition
    "OverlayProvider",
    "OverlayProviderRegistry",
    "OverlayCompositor",
    "OverlayFrameRef",
    "OVERLAY_PROVIDERS",
    # rendering
    "OverlayRenderer",
    "PillowOverlayRenderer",
    "OverlayError",
    "OverlayBackendUnavailableError",
]

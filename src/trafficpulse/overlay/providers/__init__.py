"""Violation-specific overlay providers.

Each module here maps one violation's inference metadata to the generic
:class:`~trafficpulse.overlay.metadata.OverlayScene` vocabulary. Providers are the
*only* overlay code that knows a violation exists; the metadata model, theme,
layout, and renderer stay violation-agnostic.

Two shipped providers, deliberately different in where their metadata comes from:
:class:`~trafficpulse.overlay.providers.no_helmet.NoHelmetOverlayProvider` reads a
**pixel observer's** capture (its reasoning needs the decoded image), while
:class:`~trafficpulse.overlay.providers.red_light.RedLightOverlayProvider` reads a
capture the **reasoning pass itself** produced from geometry alone. Both emit the
same generic elements, which is the point.
"""

from .no_helmet import NoHelmetOverlayProvider, register_no_helmet_overlay
from .red_light import RedLightOverlayProvider, register_red_light_overlay

__all__ = [
    "NoHelmetOverlayProvider",
    "register_no_helmet_overlay",
    "RedLightOverlayProvider",
    "register_red_light_overlay",
]

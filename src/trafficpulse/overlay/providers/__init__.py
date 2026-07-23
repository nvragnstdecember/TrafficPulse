"""Violation-specific overlay providers.

Each module here maps one violation's inference metadata to the generic
:class:`~trafficpulse.overlay.metadata.OverlayScene` vocabulary. Providers are the
*only* overlay code that knows a violation exists; the metadata model, theme,
layout, and renderer stay violation-agnostic. The shipped reference is
:class:`~trafficpulse.overlay.providers.no_helmet.NoHelmetOverlayProvider`.
"""

from .no_helmet import NoHelmetOverlayProvider, register_no_helmet_overlay

__all__ = ["NoHelmetOverlayProvider", "register_no_helmet_overlay"]

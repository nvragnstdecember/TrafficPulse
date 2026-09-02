"""Violation-specific overlay providers, and the default registration (R6).

Each module here maps one violation's inference metadata to the generic
:class:`~trafficpulse.overlay.metadata.OverlayScene` vocabulary. Providers are the
*only* overlay code that knows a violation exists; the metadata model, theme,
layout, and renderer stay violation-agnostic.

Five violation providers, drawing their metadata from two different places:
``no_helmet`` and ``triple_riding`` read a **frame observer's** capture (helmet
reasoning needs the decoded image; rider counting needs every track in a frame at
once), while ``wrong_way``, ``illegal_stopping`` and ``red_light_jumping`` read a
capture the **reasoning pass itself** produced from ``TrackState`` geometry alone.
Both shapes emit the same generic elements, which is the point.

:func:`register_defaults` is what turned the registry from a designed-but-unused
plug-in point into the actual dispatch mechanism. It binds every shipped provider
to its violation kind *and* to the capture type a finished run surfaces, so a
driver holding an engine's captures can ask for "the provider for this thing"
without naming a single violation.

The observer adapters
---------------------
Two providers consume a *tuple of captured frames*, but what the engine surfaces is
the **observer** that produced them (the same object the rule reasons through). The
adapters below bridge that in one line each, which keeps the extraction in the
registration -- where it is declared once -- instead of in every driver, and leaves
both provider constructors untouched.
"""

from collections.abc import Sequence

from ...contracts import ConfirmedEvent
from ...pipeline.helmet_observer import HelmetFrameObserver
from ...pipeline.triple_riding import RiderCountFrameObserver
from ..registry import OverlayProvider, OverlayProviderRegistry
from .helmet_analysis import (
    HelmetAnalysisOverlayProvider,
    register_helmet_analysis_overlay,
)
from .illegal_stopping import (
    IllegalStoppingOverlayProvider,
    register_illegal_stopping_overlay,
)
from .no_helmet import NoHelmetOverlayProvider, register_no_helmet_overlay
from .red_light import RedLightOverlayProvider, register_red_light_overlay
from .triple_riding import TripleRidingOverlayProvider, register_triple_riding_overlay
from .wrong_way import WrongWayOverlayProvider, register_wrong_way_overlay

# ``register_no_helmet_overlay`` / ``register_triple_riding_overlay`` remain the
# per-module registration for a caller wiring one violation on its own; the default
# wiring below registers their observer adapters instead.

__all__ = [
    "HelmetAnalysisOverlayProvider",
    "IllegalStoppingOverlayProvider",
    "NoHelmetOverlayProvider",
    "RedLightOverlayProvider",
    "TripleRidingOverlayProvider",
    "WrongWayOverlayProvider",
    "register_defaults",
    "register_helmet_analysis_overlay",
    "register_illegal_stopping_overlay",
    "register_no_helmet_overlay",
    "register_red_light_overlay",
    "register_triple_riding_overlay",
    "register_wrong_way_overlay",
]


def _no_helmet_from_observer(
    observer: HelmetFrameObserver, events: Sequence[ConfirmedEvent] = ()
) -> OverlayProvider:
    """Build the no-helmet provider from the observer the engine surfaces."""

    return NoHelmetOverlayProvider(observer.overlay_frames(), events)


def _triple_riding_from_observer(
    observer: RiderCountFrameObserver, events: Sequence[ConfirmedEvent] = ()
) -> OverlayProvider:
    """Build the triple-riding provider from the observer the engine surfaces."""

    return TripleRidingOverlayProvider(observer.overlay_frames(), events)


def register_defaults(registry: OverlayProviderRegistry) -> None:
    """Register every shipped provider, indexed by kind and by capture type.

    Registration order fixes nothing: the compositor sorts providers by kind and the
    renderer paints by layer, so this is simply where they are enumerated.

    Calling this twice on one registry raises -- deliberately, since a silent
    re-registration would hide a double-wiring bug. A caller that needs a fresh set
    builds a fresh :class:`OverlayProviderRegistry`.
    """

    # Reasoning-pass captures: each provider module owns both its kind and the
    # capture type it consumes, so these need nothing from here.
    register_wrong_way_overlay(registry)
    register_illegal_stopping_overlay(registry)
    register_red_light_overlay(registry)
    # Observer-backed providers: the engine surfaces the observer, so the factory
    # registered here is the adapter, and the source type is the observer's.
    registry.register(
        NoHelmetOverlayProvider.violation_kind,
        _no_helmet_from_observer,
        source_type=HelmetFrameObserver,
    )
    registry.register(
        TripleRidingOverlayProvider.violation_kind,
        _triple_riding_from_observer,
        source_type=RiderCountFrameObserver,
    )
    # Perception without enforcement. Registered against its own observer type, which
    # is deliberately *not* a ``HelmetFrameObserver`` subclass: ``kind_for`` resolves by
    # ``isinstance``, so a subclass would silently draw an analysis run as if the
    # no-helmet rule had confirmed something.
    register_helmet_analysis_overlay(registry)

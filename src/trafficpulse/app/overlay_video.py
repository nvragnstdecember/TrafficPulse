"""App-layer overlay integration: render a job's annotated video (H7 + overlay; R6).

The composition point between a finished H6 run and the overlay framework. After a
run it (a) collects the overlay metadata the engine exposes -- per-rule pixel
observers and per-rule geometry captures, both already produced during inference,
so **no model re-runs here** -- (b) asks the overlay provider registry for the
provider each of those captures belongs to, and (c) re-decodes the source clip to
draw every frame's scene, encoding a browser-playable annotated video.

Registry dispatch, not a violation chain (R6)
----------------------------------------------
This module used to name two violations directly: it collected the helmet frames,
built a ``NoHelmetOverlayProvider``, collected the red-light captures, built a
``RedLightOverlayProvider``, and that was the whole annotated video. Every other
confirmed violation -- wrong-way, illegal-stopping, triple-riding -- rendered
nothing, and adding one meant editing this function.

It now names **no violation at all**. Each provider is registered against the
capture type it consumes (see
:func:`trafficpulse.overlay.providers.register_defaults`), so this driver just asks
the registry to turn each capture the engine surfaced into a provider. A capture no
violation claimed resolves to ``None`` and is skipped; a violation whose rule did
not run surfaces no capture and contributes nothing -- so the annotated video always
describes exactly the rules that were actually applied.

Keeping this in the app layer is still deliberate: the engine stays
violation-agnostic (it exposes ``frame_observers()`` / ``overlay_captures()`` and
knows nothing of providers), the overlay framework stays inference-agnostic, and
this module -- which already depends on both -- does the wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..contracts import ConfirmedEvent
from ..engine import InferenceEngine
from ..overlay import OverlayCompositor
from ..overlay.providers import register_defaults
from ..overlay.registry import OverlayProvider, OverlayProviderRegistry
from ..overlay.video import OverlayVideoResult, render_overlay_video

# The application's provider set, built once at import. A module-level registry
# rather than the package-level ``OVERLAY_PROVIDERS`` singleton: registration is
# not idempotent (deliberately -- see ``register_defaults``), and owning our own
# instance keeps a second import or a test's registry from colliding with it.
OVERLAY_REGISTRY = OverlayProviderRegistry()
register_defaults(OVERLAY_REGISTRY)


def overlay_sources(engine: InferenceEngine) -> tuple[object, ...]:
    """Every overlay metadata source a finished run exposes, in one sequence.

    The two shapes the engine distinguishes -- pixel observers and geometry-only
    captures -- are the same thing to this layer: something a rule produced that a
    provider may know how to draw. Merging them here is what lets the dispatch below
    stay a single loop.
    """

    return (*engine.frame_observers(), *engine.overlay_captures())


def build_job_compositor(
    engine: InferenceEngine, events: Sequence[ConfirmedEvent]
) -> OverlayCompositor | None:
    """Build the overlay compositor for a finished run, or ``None`` if nothing to draw.

    One provider per violation that produced overlay metadata, resolved through the
    registry rather than named here. Providers are ordered by violation kind so a
    given run always composes them the same way regardless of the order its rules
    were declared in; the renderer's layer sort governs actual paint order, so this
    ordering affects reproducibility, not appearance.

    Every provider sees the run's **whole** event list and filters to its own
    violation type internally (each already does). Nothing short-circuits: one
    violation's overlay can neither suppress nor replace another's.
    """

    providers: list[OverlayProvider] = []
    for source in overlay_sources(engine):
        provider = OVERLAY_REGISTRY.create_for(source, events)
        # ``has_content`` is the provider's own answer to "did this rule publish
        # anything to draw": a rule whose capture was disabled or that observed
        # nothing contributes none, so a run can still legitimately have no
        # annotated video (the status then settles at NONE rather than pending).
        if provider is not None and provider.has_content():
            providers.append(provider)
    if not providers:
        return None
    providers.sort(key=lambda provider: provider.violation_kind)
    return OverlayCompositor(providers)


def render_job_overlay(
    *,
    engine: InferenceEngine,
    source_path: Path,
    output_path: Path,
    events: Sequence[ConfirmedEvent],
    camera_id: str,
) -> OverlayVideoResult | None:
    """Render the annotated video for one job, or ``None`` if there was nothing to draw."""

    compositor = build_job_compositor(engine, events)
    if compositor is None:
        return None
    return render_overlay_video(
        source_path=source_path,
        output_path=output_path,
        compositor=compositor,
        camera_id=camera_id,
    )

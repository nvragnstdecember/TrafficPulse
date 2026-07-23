"""App-layer overlay integration: render a job's annotated video (H7 + overlay).

The composition point between a finished H6 run and the overlay framework. After a
run it (a) reads the per-rule pixel observers the engine exposes to collect the
no-helmet **overlay capture** (already produced during inference -- no model re-runs
here), (b) builds a violation-specific
:class:`~trafficpulse.overlay.providers.no_helmet.NoHelmetOverlayProvider` over that
capture + the confirmed events, and (c) re-decodes the source clip to draw every
frame's scene, encoding a browser-playable annotated video.

Keeping this in the app layer is deliberate: the engine stays violation-agnostic
(it only exposes ``frame_observers()``), the overlay framework stays
inference-agnostic, and this module -- which already depends on both -- does the
wiring. Future violations that gain a pixel observer + provider extend
:func:`build_job_compositor` here, and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..contracts import ConfirmedEvent
from ..engine import InferenceEngine
from ..overlay import OverlayCompositor
from ..overlay.providers.no_helmet import NoHelmetOverlayProvider
from ..overlay.video import OverlayVideoResult, render_overlay_video
from ..pipeline.helmet_observer import HelmetFrameObserver, HelmetOverlayFrame


def collect_helmet_overlay_frames(engine: InferenceEngine) -> tuple[HelmetOverlayFrame, ...]:
    """Gather the no-helmet overlay capture from an engine's pixel observers.

    Empty when the run had no no-helmet rule (or capture was off), so the caller
    simply skips annotation and the original video is served unchanged.
    """

    frames: list[HelmetOverlayFrame] = []
    for observer in engine.frame_observers():
        if isinstance(observer, HelmetFrameObserver):
            frames.extend(observer.overlay_frames())
    return tuple(frames)


def build_job_compositor(
    engine: InferenceEngine, events: Sequence[ConfirmedEvent]
) -> OverlayCompositor | None:
    """Build the overlay compositor for a finished run, or ``None`` if nothing to draw.

    One provider is registered per violation that produced overlay metadata. Today
    that is no-helmet; a future violation adds its provider here (the renderer and
    metadata model are untouched).
    """

    providers = []
    helmet_frames = collect_helmet_overlay_frames(engine)
    if helmet_frames:
        providers.append(NoHelmetOverlayProvider(helmet_frames, events))
    if not providers:
        return None
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

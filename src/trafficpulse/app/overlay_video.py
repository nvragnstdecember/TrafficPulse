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
from ..overlay.providers.red_light import RedLightOverlayProvider
from ..overlay.registry import OverlayProvider
from ..overlay.video import OverlayVideoResult, render_overlay_video
from ..pipeline.helmet_observer import HelmetFrameObserver, HelmetOverlayFrame
from ..pipeline.red_light import RedLightOverlayCapture


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


def collect_red_light_captures(engine: InferenceEngine) -> tuple[RedLightOverlayCapture, ...]:
    """Gather the red-light overlay capture from an engine's geometry-only captures.

    The pixel-free counterpart of :func:`collect_helmet_overlay_frames`: red-light
    reasons over ``TrackState``s, so its metadata arrives through
    ``engine.overlay_captures()`` rather than through a frame observer. Empty when
    the run had no red-light rule.
    """

    return tuple(
        capture
        for capture in engine.overlay_captures()
        if isinstance(capture, RedLightOverlayCapture)
    )


def build_job_compositor(
    engine: InferenceEngine, events: Sequence[ConfirmedEvent]
) -> OverlayCompositor | None:
    """Build the overlay compositor for a finished run, or ``None`` if nothing to draw.

    One provider per violation that produced overlay metadata. A violation whose
    rule did not run contributes nothing, so the annotated video always describes
    exactly the rules that were actually applied.
    """

    providers: list[OverlayProvider] = []
    helmet_frames = collect_helmet_overlay_frames(engine)
    if helmet_frames:
        providers.append(NoHelmetOverlayProvider(helmet_frames, events))
    for capture in collect_red_light_captures(engine):
        # The scene geometry is worth drawing even when no vehicle was tracked
        # through the junction: a mis-drawn stop line should be visible on the
        # first frame, not deduced from an absence of events.
        providers.append(RedLightOverlayProvider(capture, events))
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

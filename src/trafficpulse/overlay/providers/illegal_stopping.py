"""The illegal-stopping overlay provider: dwell metadata -> generic elements (R6).

Maps the reasoning pass's
:class:`~trafficpulse.pipeline.illegal_stopping.IllegalStoppingOverlayCapture` plus
the finalized :class:`~trafficpulse.contracts.ConfirmedEvent`\\ s onto generic
overlay elements. Like red-light and wrong-way it needs no pixel observer -- the
rule reasons over ``TrackState`` geometry -- and it recomputes nothing.

What it draws
-------------
* the **no-stopping zones** as closed ``CONTEXT`` links, from the scene's own
  polygons -- the very geometry the rule tested membership against. Drawn on every
  frame so a mis-drawn zone is visible immediately rather than inferred from a
  missing event (the same argument the red-light provider makes for its stop line);
* the **vehicle box** per tracked frame -- ``SUBJECT``, escalating ``NONE`` ->
  ``OBSERVING`` (stationary inside a no-stopping zone, dwell accumulating) ->
  ``CONFIRMED`` (from the triggering instant);
* the **dwell** as the caption's metric (``"6.2s"``) when the derivation measured
  one, with the meter showing progress toward the scene's ``stationary_duration``;
* a **banner** from the trigger instant naming the violation, the zone, and the
  dwell reached.

The prohibited area is only implied when it exists
---------------------------------------------------
``zone_polygons`` comes from the scene, so the drawn region is the configured
no-stopping zone and not a guess. A capture with no polygons draws none: an
overlay that outlined a plausible-looking area would fabricate the single piece of
scene configuration this violation depends on.

State beats duration when the two disagree
--------------------------------------------
A vehicle can be inside the zone while still moving (approaching), and stationary
while outside it (a legal stop nearby). Both facts are carried per frame and both
are said in the caption, because the violation is their *conjunction* -- showing
only "stopped" would misrepresent a legally stopped vehicle as an offender.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...contracts import ConfirmedEvent
from ...contracts.enums import ViolationType
from ...pipeline.base import _MEDIA_TIME_EPOCH
from ...pipeline.illegal_stopping import (
    IllegalStoppingOverlayCapture,
    IllegalStoppingTrackFrame,
)
from ..metadata import (
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
)
from ..registry import OverlayFrameRef, OverlayProviderRegistry

VIOLATION_KIND = ViolationType.ILLEGAL_STOPPING.value

_DWELL_THRESHOLD = "stationary_duration"


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


@dataclass(frozen=True)
class _Episode:
    """One confirmed illegal-stopping event, reduced to what the overlay needs."""

    track_ids: frozenset[str]
    start_seconds: float
    trigger_seconds: float
    threshold_seconds: float | None


def _episodes(events: Sequence[ConfirmedEvent]) -> tuple[_Episode, ...]:
    episodes: list[_Episode] = []
    for event in events:
        if event.violation_type is not ViolationType.ILLEGAL_STOPPING:
            continue
        threshold = next(
            (t.value for t in event.thresholds if t.name == _DWELL_THRESHOLD), None
        )
        episodes.append(
            _Episode(
                track_ids=frozenset(event.track_ids),
                start_seconds=(event.start_at - _MEDIA_TIME_EPOCH).total_seconds(),
                trigger_seconds=(event.trigger_at - _MEDIA_TIME_EPOCH).total_seconds(),
                threshold_seconds=threshold,
            )
        )
    return tuple(episodes)


class IllegalStoppingOverlayProvider:
    """Maps an illegal-stopping capture + its confirmed events to overlay elements."""

    violation_kind = VIOLATION_KIND

    def __init__(
        self,
        capture: IllegalStoppingOverlayCapture,
        events: Sequence[ConfirmedEvent] = (),
    ) -> None:
        self._capture = capture
        self._episodes = _episodes(events)
        by_frame: dict[int, list[IllegalStoppingTrackFrame]] = {}
        for frame in capture.frames:
            by_frame.setdefault(frame.frame_index, []).append(frame)
        self._by_frame = {
            index: sorted(frames, key=lambda f: f.track_id)
            for index, frames in by_frame.items()
        }

    def has_content(self) -> bool:
        """The configured no-stopping zones are worth drawing on their own.

        A calibrated zone that no vehicle entered is still the thing an analyst
        needs to see to trust (or correct) the calibration.
        """

        return bool(self._capture.zone_polygons) or bool(self._by_frame)

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        elements: list[OverlayElement] = [*self._zone_elements()]
        confirmed_now: list[IllegalStoppingTrackFrame] = []

        for tracked in self._by_frame.get(frame.frame_index, ()):
            episode = self._episode_for(tracked.track_id)
            triggered = episode is not None and frame.media_seconds >= episode.trigger_seconds
            elements.append(self._vehicle_box(tracked, episode, triggered=triggered))
            if triggered:
                confirmed_now.append(tracked)

        if confirmed_now:
            elements.append(self._banner(confirmed_now, frame))
        return elements

    # --- elements ---------------------------------------------------------------
    def _zone_elements(self) -> tuple[OverlayElement, ...]:
        """The no-stopping polygons the rule tested against, drawn on every frame."""

        elements: list[OverlayElement] = []
        for _zone_id, polygon in self._capture.zone_polygons:
            if len(polygon) < 3:
                continue
            # Closed ring: repeat the first vertex. A polygon needs no new element
            # type -- the same construction the red-light provider uses.
            points = tuple(OverlayPoint(x=x, y=y) for x, y in polygon) + (
                OverlayPoint(x=polygon[0][0], y=polygon[0][1]),
            )
            elements.append(
                OverlayLink(
                    points=points,
                    emphasis=OverlayEmphasis.CONTEXT,
                    layer=OverlayLayer.BACKDROP,
                )
            )
        return tuple(elements)

    def _vehicle_box(
        self,
        tracked: IllegalStoppingTrackFrame,
        episode: _Episode | None,
        *,
        triggered: bool,
    ) -> OverlayBox:
        alert = OverlayAlert.NONE
        lines: tuple[str, ...] = (f"Track: {tracked.track_id}",)
        progress: float | None = None

        if tracked.is_stationary and tracked.is_inside:
            alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.OBSERVING
            zone = tracked.zone_id or "no-stopping zone"
            lines = (f"Stopped in {zone}", f"Track: {tracked.track_id}")
            progress = self._progress(tracked, episode) if not triggered else None
        elif tracked.is_inside:
            lines = ("Moving in no-stopping zone", f"Track: {tracked.track_id}")
        elif tracked.is_stationary:
            # Stationary but outside every no-stopping zone: not an offence, and the
            # overlay says so rather than colouring a legal stop as evidence.
            lines = ("Stopped (outside zone)", f"Track: {tracked.track_id}")

        return OverlayBox(
            bounds=tracked.bbox,
            emphasis=OverlayEmphasis.SUBJECT,
            alert=alert,
            layer=OverlayLayer.SUBJECT,
            caption=OverlayCaption(
                lines=lines,
                metric=self._dwell_metric(tracked),
                prefer=Corner.TOP_LEFT,
                progress=progress,
            ),
            key=tracked.track_id,
        )

    @staticmethod
    def _dwell_metric(tracked: IllegalStoppingTrackFrame) -> str | None:
        """The measured dwell, or nothing at all -- never a substituted zero."""

        if tracked.dwell_seconds is None:
            return None
        return f"{tracked.dwell_seconds:.1f}s"

    @staticmethod
    def _progress(
        tracked: IllegalStoppingTrackFrame, episode: _Episode | None
    ) -> float | None:
        """Dwell progress toward the scene's threshold, from the published event.

        Uses the dwell the derivation measured when it has one, falling back to time
        elapsed since the episode began. A track the reasoner never confirmed gets no
        meter: inventing a denominator would draw reasoning that never happened.
        """

        if episode is None or not episode.threshold_seconds:
            return None
        elapsed = (
            tracked.dwell_seconds
            if tracked.dwell_seconds is not None
            else tracked.media_seconds - episode.start_seconds
        )
        if elapsed < 0.0:
            return None
        return min(1.0, elapsed / episode.threshold_seconds)

    def _banner(
        self, confirmed: Sequence[IllegalStoppingTrackFrame], frame: OverlayFrameRef
    ) -> OverlayBanner:
        tracks = ", ".join(sorted(t.track_id for t in confirmed))
        zones = sorted({t.zone_id for t in confirmed if t.zone_id is not None})
        dwells = [t.dwell_seconds for t in confirmed if t.dwell_seconds is not None]
        return OverlayBanner(
            title="ILLEGAL STOPPING",
            metric=f"{max(dwells):.1f}s" if dwells else None,
            lines=(
                f"Stopped in {', '.join(zones)}" if zones else "Stopped in a no-stopping zone",
            ),
            details=(f"Track(s): {tracks}", f"Media time: {_clock(frame.media_seconds)}"),
            icon="!",
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )

    def _episode_for(self, track_id: str) -> _Episode | None:
        return next((e for e in self._episodes if track_id in e.track_ids), None)


def register_illegal_stopping_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the illegal-stopping provider factory under its violation kind."""

    registry.register(
        VIOLATION_KIND,
        IllegalStoppingOverlayProvider,
        source_type=IllegalStoppingOverlayCapture,
    )

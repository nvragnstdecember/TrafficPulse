"""The red-light overlay provider: latched-entry metadata -> generic elements (H13).

The second :class:`~trafficpulse.overlay.registry.OverlayProvider`, and the first
that needs **no pixel observer at all**. Red-light reasons purely over
``TrackState`` geometry, so its capture
(:class:`~trafficpulse.pipeline.red_light.RedLightOverlayCapture`) is produced by
the reasoning pass itself. This module maps that capture plus the finalized
:class:`~trafficpulse.contracts.ConfirmedEvent`\\ s onto generic overlay elements.
It runs no detection, tracking, or classification and recomputes no geometry.

What it draws
-------------
* the **stop line** as a link, in ``CONTEXT`` -- the scene's own configured segment,
  always visible so an analyst can see where the boundary the rule used actually
  lies (a mis-drawn stop line is the most likely calibration error, and it should
  be obvious on the first frame rather than inferred from a missing event);
* the **junction polygon** as a closed link, also ``CONTEXT``. There is no polygon
  element in the metadata vocabulary and none is added: a ring is a link whose last
  point repeats its first, which the generic renderer already draws correctly;
* the **vehicle box** per tracked frame -- ``SUBJECT``, escalating through the two
  shared alert axes: ``NONE`` while approaching, ``OBSERVING`` once it has entered
  the junction on red and the reasoner is debouncing, ``CONFIRMED`` from the
  triggering instant onward;
* the **latched signal state** as the caption's metric on the entering vehicle, so
  the single fact the whole rule turns on is legible on the frame;
* a **banner** from the trigger instant, naming the violation and the latched state.

Why the caption says what was latched, not what the signal is now
------------------------------------------------------------------
The rule's entire correctness claim is that the signal is read **once**, at the
stop-line crossing, and not re-evaluated. An overlay that displayed the *current*
signal state beside a confirmed violation would contradict the reasoning it exists
to explain -- an analyst would see "GREEN" next to a red-light event and reasonably
conclude the system was broken. The capture carries the latched ``entry_state`` per
frame; that is what is drawn.

The provider owns no drawing code and names no colour: it speaks emphasis + alert,
which the theme resolves. The renderer is untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...contracts import ConfirmedEvent
from ...contracts.enums import SignalState, ViolationType
from ...pipeline.base import _MEDIA_TIME_EPOCH
from ...pipeline.red_light import RedLightOverlayCapture, RedLightTrackFrame
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

VIOLATION_KIND = ViolationType.RED_LIGHT_JUMPING.value

_STATE_DISPLAY = {
    SignalState.RED: "RED",
    SignalState.AMBER: "AMBER",
    SignalState.GREEN: "GREEN",
    SignalState.OFF: "OFF",
    SignalState.UNKNOWN: "UNKNOWN",
}

_PERSISTENCE_THRESHOLD = "min_persistence"


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


@dataclass(frozen=True)
class _Episode:
    """One confirmed red-light event, reduced to what the overlay needs."""

    track_ids: frozenset[str]
    start_seconds: float
    trigger_seconds: float
    threshold_seconds: float | None


def _episodes(events: Sequence[ConfirmedEvent]) -> tuple[_Episode, ...]:
    episodes: list[_Episode] = []
    for event in events:
        if event.violation_type is not ViolationType.RED_LIGHT_JUMPING:
            continue
        threshold = next(
            (t.value for t in event.thresholds if t.name == _PERSISTENCE_THRESHOLD), None
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


class RedLightOverlayProvider:
    """Maps a red-light capture + its confirmed events to overlay elements."""

    def __init__(
        self, capture: RedLightOverlayCapture, events: Sequence[ConfirmedEvent] = ()
    ) -> None:
        self._capture = capture
        self._episodes = _episodes(events)
        by_frame: dict[int, list[RedLightTrackFrame]] = {}
        for frame in capture.frames:
            by_frame.setdefault(frame.frame_index, []).append(frame)
        # Sorted by track id so element order is deterministic for a given frame,
        # which keeps the rendered output reproducible across runs.
        self._by_frame = {
            index: sorted(frames, key=lambda f: f.track_id)
            for index, frames in by_frame.items()
        }

    @property
    def violation_kind(self) -> str:
        return VIOLATION_KIND

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        elements: list[OverlayElement] = [*self._geometry_elements()]
        confirmed_now: list[RedLightTrackFrame] = []

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
    def _geometry_elements(self) -> tuple[OverlayElement, ...]:
        """The scene geometry the rule reasoned against, drawn on every frame."""

        (ax, ay), (bx, by) = self._capture.stop_line
        stop_line = OverlayLink(
            points=(OverlayPoint(x=ax, y=ay), OverlayPoint(x=bx, y=by)),
            emphasis=OverlayEmphasis.CONTEXT,
            layer=OverlayLayer.BACKDROP,
        )
        ring = self._capture.zone_polygon
        if len(ring) < 3:
            return (stop_line,)
        # Closed ring: repeat the first vertex. A polygon needs no new element type.
        points = tuple(OverlayPoint(x=x, y=y) for x, y in ring) + (
            OverlayPoint(x=ring[0][0], y=ring[0][1]),
        )
        junction = OverlayLink(
            points=points,
            emphasis=OverlayEmphasis.CONTEXT,
            layer=OverlayLayer.BACKDROP,
        )
        return (junction, stop_line)

    def _vehicle_box(
        self, tracked: RedLightTrackFrame, episode: _Episode | None, *, triggered: bool
    ) -> OverlayBox:
        alert = OverlayAlert.NONE
        lines: tuple[str, ...] = (f"Track: {tracked.track_id}",)
        metric: str | None = None
        progress: float | None = None

        if tracked.entered_on_red:
            alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.OBSERVING
            # The state latched at the stop line -- never the current one; see the
            # module docstring.
            metric = _STATE_DISPLAY[tracked.entry_state]
            lines = ("Entered on red", f"Track: {tracked.track_id}")
            progress = self._progress(tracked, episode) if not triggered else None
        elif tracked.is_inside:
            lines = ("In junction", f"Track: {tracked.track_id}")
            metric = _STATE_DISPLAY[tracked.entry_state]

        return OverlayBox(
            bounds=tracked.bbox,
            emphasis=OverlayEmphasis.SUBJECT,
            alert=alert,
            layer=OverlayLayer.SUBJECT,
            caption=OverlayCaption(
                lines=lines, metric=metric, prefer=Corner.TOP_LEFT, progress=progress
            ),
            key=tracked.track_id,
        )

    @staticmethod
    def _progress(tracked: RedLightTrackFrame, episode: _Episode | None) -> float | None:
        """Debounce progress, read verbatim off the event the reasoner published.

        A track the reasoner never confirmed gets no meter: there is no episode to
        measure against, and inventing a denominator would draw reasoning the system
        never performed.
        """

        if episode is None or not episode.threshold_seconds:
            return None
        elapsed = tracked.media_seconds - episode.start_seconds
        if elapsed < 0.0:
            return None
        return min(1.0, elapsed / episode.threshold_seconds)

    def _banner(
        self, confirmed: Sequence[RedLightTrackFrame], frame: OverlayFrameRef
    ) -> OverlayBanner:
        states = {_STATE_DISPLAY[t.entry_state] for t in confirmed}
        tracks = ", ".join(sorted(t.track_id for t in confirmed))
        return OverlayBanner(
            title="RED-LIGHT JUMPING",
            metric=next(iter(sorted(states))) if len(states) == 1 else None,
            lines=(f"Signal at stop line: {', '.join(sorted(states))}",),
            details=(f"Track(s): {tracks}", f"Media time: {_clock(frame.media_seconds)}"),
            icon="!",
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )

    def _episode_for(self, track_id: str) -> _Episode | None:
        return next((e for e in self._episodes if track_id in e.track_ids), None)


def register_red_light_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the red-light provider factory under its violation kind.

    The plug-in call: a driver that ran the red-light rule asks the registry for a
    ``"red_light_jumping"`` provider, passing the reasoning capture + confirmed
    events.
    """

    registry.register(VIOLATION_KIND, RedLightOverlayProvider)

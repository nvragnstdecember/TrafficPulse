"""The wrong-way overlay provider: heading metadata -> generic elements (R6).

The third :class:`~trafficpulse.overlay.registry.OverlayProvider`, and the second
that needs no pixel observer: wrong-way reasons purely over ``TrackState``
geometry, so its capture
(:class:`~trafficpulse.pipeline.wrong_way.WrongWayOverlayCapture`) is produced by
the reasoning pass itself. This module maps that capture plus the finalized
:class:`~trafficpulse.contracts.ConfirmedEvent`\\ s onto generic overlay elements.
It runs no detection, tracking, or heading calculation and recomputes no geometry.

What it draws
-------------
* the **vehicle box** per tracked frame -- ``SUBJECT``, escalating through the two
  shared alert axes: ``NONE`` while travelling legally, ``OBSERVING`` once the
  heading contradicts the lane and the reasoner is accumulating persistence,
  ``CONFIRMED`` from the triggering instant onward;
* the **measured travel direction** as a short arrow from the box centre, drawn at
  the ``heading_degrees`` the derivation actually measured -- this is the fact the
  rule turns on, so it is drawn rather than described;
* the **legal direction** as a second, ``CONTEXT`` arrow from the same centre, so
  the contradiction is visible as an angle between two lines rather than asserted
  in text. Drawn **only** when the scene stated a legal heading for that
  observation; when it did not, no arrow is invented (see below);
* the **deviation** as the caption's metric (``"143°"``), with the persistence
  meter showing how far toward ``min_persistence`` the run had got;
* a **banner** from the trigger instant naming the violation, the lane, and the
  deviation.

Why the arrows are the honest part
-----------------------------------
The temptation with wrong-way is to draw a trajectory -- a swept path showing where
the vehicle went. The engine does not produce one: it produces a per-frame heading
angle derived from consecutive box centres. Drawing a smooth path would imply a
continuous, sub-frame-accurate estimate the system never made. A short arrow at the
measured angle claims exactly what was measured and nothing more, and its length is
a fixed fraction of the box so it never reads as a distance.

The provider owns no drawing code and names no colour: it speaks emphasis + alert,
which the theme resolves. The renderer is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ...contracts import ConfirmedEvent
from ...contracts.enums import ViolationType
from ...pipeline.base import _MEDIA_TIME_EPOCH
from ...pipeline.wrong_way import WrongWayOverlayCapture, WrongWayTrackFrame
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

VIOLATION_KIND = ViolationType.WRONG_WAY.value

_PERSISTENCE_THRESHOLD = "min_persistence"
# Arrow length as a fraction of the box's larger side: proportional to the subject
# so it stays legible at any scale, and never suggestive of a travelled distance.
_ARROW_FRACTION = 0.9
_MIN_ARROW_PX = 18.0


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


@dataclass(frozen=True)
class _Episode:
    """One confirmed wrong-way event, reduced to what the overlay needs."""

    track_ids: frozenset[str]
    start_seconds: float
    trigger_seconds: float
    threshold_seconds: float | None


def _episodes(events: Sequence[ConfirmedEvent]) -> tuple[_Episode, ...]:
    episodes: list[_Episode] = []
    for event in events:
        if event.violation_type is not ViolationType.WRONG_WAY:
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


def _arrow(
    bbox: tuple[float, float, float, float], degrees: float
) -> tuple[OverlayPoint, OverlayPoint]:
    """A short arrow from the box centre at ``degrees`` (image space, +y down)."""

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    length = max(_MIN_ARROW_PX, max(x2 - x1, y2 - y1) * _ARROW_FRACTION)
    radians = math.radians(degrees)
    return (
        OverlayPoint(x=cx, y=cy),
        OverlayPoint(x=cx + length * math.cos(radians), y=cy + length * math.sin(radians)),
    )


class WrongWayOverlayProvider:
    """Maps a wrong-way capture + its confirmed events to overlay elements."""

    violation_kind = VIOLATION_KIND

    def __init__(
        self, capture: WrongWayOverlayCapture, events: Sequence[ConfirmedEvent] = ()
    ) -> None:
        self._capture = capture
        self._episodes = _episodes(events)
        by_frame: dict[int, list[WrongWayTrackFrame]] = {}
        for frame in capture.frames:
            by_frame.setdefault(frame.frame_index, []).append(frame)
        # Sorted by track id so element order is deterministic for a given frame,
        # which keeps the rendered output reproducible across runs.
        self._by_frame = {
            index: sorted(frames, key=lambda f: f.track_id)
            for index, frames in by_frame.items()
        }

    def has_content(self) -> bool:
        """Only the per-frame capture: a legal direction has no drawable anchor.

        Unlike the zone-bearing violations, wrong-way's scene fact is a *vector*,
        which is meaningless to draw without a vehicle to attach it to. A run that
        tracked nothing therefore genuinely has no wrong-way overlay.
        """

        return bool(self._by_frame)

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        elements: list[OverlayElement] = []
        confirmed_now: list[WrongWayTrackFrame] = []

        for tracked in self._by_frame.get(frame.frame_index, ()):
            episode = self._episode_for(tracked.track_id)
            triggered = episode is not None and frame.media_seconds >= episode.trigger_seconds
            elements.append(self._vehicle_box(tracked, episode, triggered=triggered))
            elements.extend(self._direction_links(tracked, triggered=triggered))
            if triggered:
                confirmed_now.append(tracked)

        if confirmed_now:
            elements.append(self._banner(confirmed_now, frame))
        return elements

    # --- elements ---------------------------------------------------------------
    def _vehicle_box(
        self, tracked: WrongWayTrackFrame, episode: _Episode | None, *, triggered: bool
    ) -> OverlayBox:
        alert = OverlayAlert.NONE
        lines: tuple[str, ...] = (f"Track: {tracked.track_id}",)
        progress: float | None = None

        if tracked.is_contradiction:
            alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.OBSERVING
            lines = ("Against lane flow", f"Track: {tracked.track_id}")
            progress = self._progress(tracked, episode) if not triggered else None

        return OverlayBox(
            bounds=tracked.bbox,
            emphasis=OverlayEmphasis.SUBJECT,
            alert=alert,
            layer=OverlayLayer.SUBJECT,
            caption=OverlayCaption(
                lines=lines,
                # The measured angle between travel and the lane's legal flow: the
                # single number the rule's threshold is compared against.
                metric=f"{tracked.deviation_degrees:.0f}°",
                prefer=Corner.TOP_LEFT,
                progress=progress,
            ),
            key=tracked.track_id,
        )

    def _direction_links(
        self, tracked: WrongWayTrackFrame, *, triggered: bool
    ) -> tuple[OverlayElement, ...]:
        """The measured heading, and the legal heading when the scene declared one."""

        alert = OverlayAlert.NONE
        if tracked.is_contradiction:
            alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.OBSERVING
        travelled = OverlayLink(
            points=_arrow(tracked.bbox, tracked.heading_degrees),
            emphasis=OverlayEmphasis.SUBJECT,
            alert=alert,
            layer=OverlayLayer.LINK,
        )
        if tracked.legal_heading_degrees is None:
            # The scene stated no legal heading for this observation; drawing a
            # second arrow would invent the very reference the comparison used.
            return (travelled,)
        legal = OverlayLink(
            points=_arrow(tracked.bbox, tracked.legal_heading_degrees),
            emphasis=OverlayEmphasis.CONTEXT,
            layer=OverlayLayer.LINK,
        )
        return (legal, travelled)

    @staticmethod
    def _progress(tracked: WrongWayTrackFrame, episode: _Episode | None) -> float | None:
        """Persistence progress, read verbatim off the event the reasoner published.

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
        self, confirmed: Sequence[WrongWayTrackFrame], frame: OverlayFrameRef
    ) -> OverlayBanner:
        tracks = ", ".join(sorted(t.track_id for t in confirmed))
        worst = max(t.deviation_degrees for t in confirmed)
        return OverlayBanner(
            title="WRONG WAY",
            metric=f"{worst:.0f}°",
            lines=(f"Against legal flow of lane {self._capture.lane_id}",),
            details=(f"Track(s): {tracks}", f"Media time: {_clock(frame.media_seconds)}"),
            icon="!",
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )

    def _episode_for(self, track_id: str) -> _Episode | None:
        return next((e for e in self._episodes if track_id in e.track_ids), None)


def register_wrong_way_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the wrong-way provider factory under its violation kind."""

    registry.register(
        VIOLATION_KIND, WrongWayOverlayProvider, source_type=WrongWayOverlayCapture
    )

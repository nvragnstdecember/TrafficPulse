"""The triple-riding overlay provider: occupancy metadata -> generic elements (R6).

Maps the rider-count observer's
:class:`~trafficpulse.pipeline.triple_riding.TripleRidingOverlayFrame` capture plus
the finalized :class:`~trafficpulse.contracts.ConfirmedEvent`\\ s onto generic
overlay elements. Like the no-helmet provider it reads an observer's capture; unlike
it, nothing here touched pixels -- rider counting is pure association geometry.

What it draws
-------------
* the **motorcycle box** -- ``SUBJECT``, escalating ``NONE`` -> ``OBSERVING`` (the
  count has reached the statutory threshold and the reasoner is accumulating
  persistence) -> ``CONFIRMED`` (from the triggering instant);
* the **rider count** as the caption's metric (``"3"``), which is the entire
  substance of this violation;
* one **box per attached rider** -- ``OBJECT``, so the theme sets them back from the
  subject -- drawn from the boxes the associator actually attached to that
  motorcycle. Riders are *evidence for the count*, so showing them lets an analyst
  check the count rather than take it on trust;
* a **link** from each rider to the motorcycle centre, making the association the
  count was derived from visible as a relationship rather than an assertion;
* a **banner** from the trigger instant naming the violation and the count.

Occupancy is drawn, never arranged
-----------------------------------
The engine associates riders to a motorcycle; it does not resolve *seat positions*,
and this provider does not pretend otherwise. Each rider box is where that rider was
tracked. A motorcycle whose riders were counted but whose individual boxes are not
in the capture is drawn with its count and no rider boxes -- the honest shape --
rather than with placeholder seats laid out along the vehicle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...contracts import ConfirmedEvent
from ...contracts.enums import ViolationType
from ...pipeline.base import _MEDIA_TIME_EPOCH
from ...pipeline.triple_riding import TripleRidingOverlayFrame
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

VIOLATION_KIND = ViolationType.TRIPLE_RIDING.value

_PERSISTENCE_THRESHOLD = "min_persistence"
_RIDER_COUNT_THRESHOLD = "rider_count_threshold"
# The statutory default the scene builder ships; used only to decide when to show a
# box as "observing" before an event exists. The scene's own value wins when the
# confirmed event carries it.
_DEFAULT_RIDER_THRESHOLD = 3.0


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def _centre(bbox: tuple[float, float, float, float]) -> OverlayPoint:
    x1, y1, x2, y2 = bbox
    return OverlayPoint(x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0)


@dataclass(frozen=True)
class _Episode:
    """One confirmed triple-riding event, reduced to what the overlay needs."""

    track_ids: frozenset[str]
    start_seconds: float
    trigger_seconds: float
    threshold_seconds: float | None
    rider_threshold: float | None


def _episodes(events: Sequence[ConfirmedEvent]) -> tuple[_Episode, ...]:
    episodes: list[_Episode] = []
    for event in events:
        if event.violation_type is not ViolationType.TRIPLE_RIDING:
            continue
        thresholds = {t.name: t.value for t in event.thresholds}
        episodes.append(
            _Episode(
                track_ids=frozenset(event.track_ids),
                start_seconds=(event.start_at - _MEDIA_TIME_EPOCH).total_seconds(),
                trigger_seconds=(event.trigger_at - _MEDIA_TIME_EPOCH).total_seconds(),
                threshold_seconds=thresholds.get(_PERSISTENCE_THRESHOLD),
                rider_threshold=thresholds.get(_RIDER_COUNT_THRESHOLD),
            )
        )
    return tuple(episodes)


class TripleRidingOverlayProvider:
    """Maps a rider-count capture + its confirmed events to overlay elements."""

    violation_kind = VIOLATION_KIND

    def __init__(
        self,
        frames: Sequence[TripleRidingOverlayFrame],
        events: Sequence[ConfirmedEvent] = (),
    ) -> None:
        self._episodes = _episodes(events)
        by_frame: dict[int, list[TripleRidingOverlayFrame]] = {}
        for captured in frames:
            by_frame.setdefault(captured.frame_index, []).append(captured)
        self._by_frame = {
            index: sorted(captured, key=lambda f: f.motorcycle_track_id)
            for index, captured in by_frame.items()
        }

    def has_content(self) -> bool:
        """Only the per-frame capture: this violation has no scene geometry."""

        return bool(self._by_frame)

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        elements: list[OverlayElement] = []
        confirmed_now: list[TripleRidingOverlayFrame] = []

        for captured in self._by_frame.get(frame.frame_index, ()):
            episode = self._episode_for(captured.motorcycle_track_id)
            triggered = episode is not None and frame.media_seconds >= episode.trigger_seconds
            over_threshold = captured.rider_count >= self._rider_threshold(episode)
            elements.append(
                self._motorcycle_box(
                    captured, episode, triggered=triggered, over_threshold=over_threshold
                )
            )
            elements.extend(self._rider_elements(captured, triggered=triggered))
            if triggered:
                confirmed_now.append(captured)

        if confirmed_now:
            elements.append(self._banner(confirmed_now, frame))
        return elements

    # --- elements ---------------------------------------------------------------
    def _motorcycle_box(
        self,
        captured: TripleRidingOverlayFrame,
        episode: _Episode | None,
        *,
        triggered: bool,
        over_threshold: bool,
    ) -> OverlayBox:
        alert = OverlayAlert.NONE
        lines: tuple[str, ...] = (f"Track: {captured.motorcycle_track_id}",)
        progress: float | None = None

        if over_threshold:
            alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.OBSERVING
            lines = (
                f"{captured.rider_count} riders",
                f"Track: {captured.motorcycle_track_id}",
            )
            progress = self._progress(captured, episode) if not triggered else None

        return OverlayBox(
            bounds=captured.motorcycle_bbox,
            emphasis=OverlayEmphasis.SUBJECT,
            alert=alert,
            layer=OverlayLayer.SUBJECT,
            caption=OverlayCaption(
                lines=lines,
                metric=str(captured.rider_count),
                prefer=Corner.TOP_LEFT,
                progress=progress,
            ),
            key=captured.motorcycle_track_id,
        )

    def _rider_elements(
        self, captured: TripleRidingOverlayFrame, *, triggered: bool
    ) -> tuple[OverlayElement, ...]:
        """One box per associated rider, each linked to the motorcycle it rides."""

        elements: list[OverlayElement] = []
        anchor = _centre(captured.motorcycle_bbox)
        alert = OverlayAlert.CONFIRMED if triggered else OverlayAlert.NONE
        for index, bbox in enumerate(captured.rider_bboxes):
            elements.append(
                OverlayBox(
                    bounds=bbox,
                    emphasis=OverlayEmphasis.OBJECT,
                    alert=alert,
                    layer=OverlayLayer.OBJECT,
                    # Numbered by capture order (track-id order), which identifies a
                    # rider within the frame without claiming a seat position.
                    caption=OverlayCaption(
                        lines=(f"Rider {index + 1}",), prefer=Corner.BOTTOM_LEFT
                    ),
                    key=f"{captured.motorcycle_track_id}:rider:{index + 1}",
                )
            )
            elements.append(
                OverlayLink(
                    points=(_centre(bbox), anchor),
                    emphasis=OverlayEmphasis.OBJECT,
                    alert=alert,
                    layer=OverlayLayer.LINK,
                )
            )
        return tuple(elements)

    @staticmethod
    def _rider_threshold(episode: _Episode | None) -> float:
        if episode is not None and episode.rider_threshold is not None:
            return episode.rider_threshold
        return _DEFAULT_RIDER_THRESHOLD

    @staticmethod
    def _progress(
        captured: TripleRidingOverlayFrame, episode: _Episode | None
    ) -> float | None:
        """Persistence progress, read verbatim off the event the reasoner published."""

        if episode is None or not episode.threshold_seconds:
            return None
        elapsed = captured.media_seconds - episode.start_seconds
        if elapsed < 0.0:
            return None
        return min(1.0, elapsed / episode.threshold_seconds)

    def _banner(
        self, confirmed: Sequence[TripleRidingOverlayFrame], frame: OverlayFrameRef
    ) -> OverlayBanner:
        tracks = ", ".join(sorted(c.motorcycle_track_id for c in confirmed))
        most = max(c.rider_count for c in confirmed)
        return OverlayBanner(
            title="TRIPLE RIDING",
            metric=str(most),
            lines=(f"{most} riders on one motorcycle",),
            details=(f"Track(s): {tracks}", f"Media time: {_clock(frame.media_seconds)}"),
            icon="!",
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )

    def _episode_for(self, track_id: str) -> _Episode | None:
        return next((e for e in self._episodes if track_id in e.track_ids), None)


def register_triple_riding_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the triple-riding provider factory under its violation kind.

    No ``source_type``: the engine surfaces the *observer*, not the frame tuple this
    provider consumes, and the observer is shared with the rule's own reasoning. The
    driver adapts it (see :func:`trafficpulse.overlay.providers.register_defaults`),
    exactly as it does for no-helmet.
    """

    registry.register(VIOLATION_KIND, TripleRidingOverlayProvider)

"""The no-helmet overlay provider: helmet inference metadata -> generic elements.

The reference :class:`~trafficpulse.overlay.registry.OverlayProvider`. It consumes
only what the no-helmet pipeline **already produced** -- the per-frame
:class:`~trafficpulse.pipeline.helmet_observer.HelmetOverlayFrame` capture (rider &
motorcycle boxes, the *exact* head-classifier crop box, the predicted label and its
confidence) plus the finalized :class:`~trafficpulse.contracts.ConfirmedEvent`\\ s --
and emits generic :class:`~trafficpulse.overlay.metadata.OverlayElement`\\ s. It runs
no detection, tracking, association, or classification and recomputes no head
coordinates: every pixel it describes traces to a value the inference pass emitted.

What it draws (per associated rider on a frame)
-----------------------------------------------
* the **motorcycle** box (``OBJECT`` -> blue) captioned ``Motorcycle / Track: <id>``;
* the **rider** box (``SUBJECT`` -> green, amber while observing) captioned
  ``Rider / Track: <id>`` plus the reasoning status;
* the exact **head region** box (``REGION`` -> yellow, red the moment the classifier
  reads ``no_helmet``) captioned with the classifier label + confidence
  (``No Helmet`` / ``97%``);
* the **association chain** head -> rider -> motorcycle as a polyline.

Temporal state drives the two shared axes, not colour choices here:

* while the reasoner is accumulating support (a ``no_helmet`` read that has not yet
  confirmed) the rider carries ``OBSERVING`` and shows "Collecting evidence…";
* once a ``no_helmet`` :class:`~trafficpulse.contracts.ConfirmedEvent` for the rider
  has triggered (frame media-time >= its ``trigger_at``), the rider, motorcycle, and
  head flip to ``CONFIRMED`` (the theme renders them red / bright red) and a banner
  pins ``⚠ NO HELMET`` with the track, timestamp, classifier confidence, temporal
  consistency, and violation id.

Visualising the reasoning progression (every number comes from the reasoner)
----------------------------------------------------------------------------
Inside a confirmed episode's own ``[start_at, trigger_at)`` window the rider shows a
live evidence meter -- ``0.63s / 1.00s`` with a 0..1 progress bar -- so an analyst
watches support accumulate and then sees it tip into a confirmation.

Both numbers are read **verbatim off the ConfirmedEvent**: the elapsed time from its
``start_at``, the bar from the ``min_persistence`` entry of its own ``thresholds``.
The provider therefore re-implements none of the rule's temporal semantics -- it does
not decide what supports a run, when uncertainty bridges a gap, or when a turban
exempts. That authority stays solely in ``rules.no_helmet`` /
``rules.temporal.TemporalRunReasoner``; this module only *replays* the window that
reasoner already published.

The corollary is deliberate: a rider the reasoner never confirmed gets no meter.
Their bare-headed frames still read ``OBSERVING`` ("Collecting evidence…"), but no
progress is drawn, because no episode exists to measure progress against and
inventing a denominator would assert reasoning the system never performed.

Persistence is measured in **seconds, not frames** because that is the unit the rule
is configured in (``rules.no_helmet``: a frame count silently changes meaning with
the clip's fps, a duration does not).

Determinism / performance
-------------------------
Pure lookups by ``frame_index`` and a timestamp comparison per rider -- O(riders on
the frame), no model inference. The provider owns no drawing code and names no
colour; it speaks only emphasis + alert, which the theme resolves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...contracts import ConfirmedEvent
from ...contracts.enums import ViolationType
from ...pipeline.base import _MEDIA_TIME_EPOCH
from ...pipeline.helmet_observer import HelmetOverlayFrame, HelmetOverlayRider
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

_LABEL_DISPLAY = {
    "helmet": "Helmet",
    "no_helmet": "No Helmet",
    "turban": "Turban",
    "uncertain": "Uncertain",
}


_PERSISTENCE_THRESHOLD = "min_persistence"


def _center(box: tuple[float, float, float, float]) -> OverlayPoint:
    return OverlayPoint(x=(box[0] + box[2]) / 2.0, y=(box[1] + box[3]) / 2.0)


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


@dataclass(frozen=True)
class _Episode:
    """One rider's confirmed episode, in media seconds, as the reasoner published it.

    ``start_seconds`` / ``trigger_seconds`` are the event's own window and
    ``threshold_seconds`` its own ``min_persistence`` -- nothing here is inferred.
    ``threshold_seconds`` is ``None`` if the event carried no such threshold, in
    which case no meter is drawn rather than one against a guessed bar.
    """

    start_seconds: float
    trigger_seconds: float
    threshold_seconds: float | None
    event: ConfirmedEvent

    def progress_at(self, media_seconds: float) -> tuple[float, str] | None:
        """The ``(fraction, "0.63s / 1.00s")`` readout at ``media_seconds``, if any."""

        threshold = self.threshold_seconds
        if threshold is None or threshold <= 0.0:
            return None
        elapsed = max(0.0, media_seconds - self.start_seconds)
        return min(1.0, elapsed / threshold), f"{elapsed:.2f}s / {threshold:.2f}s"


class NoHelmetOverlayProvider:
    """Turns captured no-helmet metadata into overlay elements (see module docstring)."""

    violation_kind = "no_helmet"

    def __init__(
        self,
        frames: Sequence[HelmetOverlayFrame],
        events: Sequence[ConfirmedEvent] = (),
    ) -> None:
        self._by_index: dict[int, HelmetOverlayFrame] = {f.frame_index: f for f in frames}
        # Per rider track: the earliest confirmed no-helmet episode, so a frame at or
        # after its trigger reads CONFIRMED and a frame inside its window can show
        # how far the reasoner had got.
        self._episodes: dict[str, _Episode] = {}
        for event in events:
            if event.violation_type is not ViolationType.NO_HELMET:
                continue
            episode = _Episode(
                start_seconds=(event.start_at - _MEDIA_TIME_EPOCH).total_seconds(),
                trigger_seconds=(event.trigger_at - _MEDIA_TIME_EPOCH).total_seconds(),
                threshold_seconds=next(
                    (t.value for t in event.thresholds if t.name == _PERSISTENCE_THRESHOLD),
                    None,
                ),
                event=event,
            )
            for track_id in event.track_ids:
                current = self._episodes.get(track_id)
                if current is None or episode.trigger_seconds < current.trigger_seconds:
                    self._episodes[track_id] = episode

    # --- OverlayProvider protocol -------------------------------------------
    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        captured = self._by_index.get(frame.frame_index)
        if captured is None:
            return ()
        elements: list[OverlayElement] = []
        banners: list[OverlayBanner] = []
        for rider in captured.riders:
            episode = self._episodes.get(rider.rider_track_id)
            if episode is not None and frame.media_seconds >= episode.trigger_seconds:
                elements.extend(self._rider_elements(rider, OverlayAlert.CONFIRMED))
                banners.append(self._banner(rider, episode))
            elif rider.helmet_label == "no_helmet":
                # Inside a confirmed episode's window we can show *its* progress,
                # taken from the event itself; outside one there is no episode to
                # measure against, so the status is shown without a meter.
                progress = (
                    episode.progress_at(frame.media_seconds)
                    if episode is not None and frame.media_seconds >= episode.start_seconds
                    else None
                )
                elements.extend(
                    self._rider_elements(rider, OverlayAlert.OBSERVING, progress=progress)
                )
            else:
                elements.extend(self._rider_elements(rider, OverlayAlert.NONE))
        # de-duplicate banners by violation id (one rider == one banner already, but
        # guard against the same event confirming twice within a frame)
        seen: set[str] = set()
        for banner in banners:
            key = banner.lines[-1] if banner.lines else banner.title
            if key not in seen:
                seen.add(key)
                elements.append(banner)
        return tuple(elements)

    # --- element construction -----------------------------------------------
    def _rider_elements(
        self,
        rider: HelmetOverlayRider,
        alert: OverlayAlert,
        *,
        progress: tuple[float, str] | None = None,
    ) -> list[OverlayElement]:
        out: list[OverlayElement] = []
        moto_box = rider.motorcycle_bbox
        rider_box = rider.rider_bbox
        head_box = rider.head_bbox

        out.append(
            OverlayBox(
                bounds=moto_box,
                emphasis=OverlayEmphasis.OBJECT,
                alert=alert,
                layer=OverlayLayer.OBJECT,
                caption=OverlayCaption(
                    lines=("Motorcycle", f"Track: {rider.motorcycle_track_id}"),
                    prefer=Corner.TOP_LEFT,
                ),
                key=rider.motorcycle_track_id,
            )
        )
        rider_lines: tuple[str, ...] = ("Rider", f"Track: {rider.rider_track_id}")
        fraction: float | None = None
        if alert is OverlayAlert.OBSERVING:
            rider_lines = (*rider_lines, "Collecting evidence…")
            if progress is not None:
                fraction, readout = progress
                rider_lines = (*rider_lines, readout)
        elif alert is OverlayAlert.CONFIRMED:
            rider_lines = (*rider_lines, "Violation confirmed")
            fraction = 1.0
        out.append(
            OverlayBox(
                bounds=rider_box,
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.SUBJECT,
                caption=OverlayCaption(
                    lines=rider_lines, prefer=Corner.TOP_RIGHT, progress=fraction
                ),
                key=rider.rider_track_id,
            )
        )
        if head_box is not None:
            metric = None if rider.confidence is None else f"{round(rider.confidence * 100)}%"
            out.append(
                OverlayBox(
                    bounds=head_box,
                    emphasis=OverlayEmphasis.REGION,
                    alert=alert,
                    layer=OverlayLayer.REGION,
                    caption=OverlayCaption(
                        lines=(_LABEL_DISPLAY.get(rider.helmet_label, rider.helmet_label),),
                        metric=metric,
                        prefer=Corner.BOTTOM_LEFT,
                    ),
                    key=f"{rider.rider_track_id}-head",
                )
            )
        # association chain: head -> rider -> motorcycle
        head_anchor = _center(head_box) if head_box is not None else _center(rider_box)
        out.append(
            OverlayLink(
                points=(head_anchor, _center(rider_box), _center(moto_box)),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.LINK,
            )
        )
        return out

    def _banner(self, rider: HelmetOverlayRider, episode: _Episode) -> OverlayBanner:
        """The pinned confirmation headline: what was confirmed, and on what evidence.

        The confidence lines are the event's **published components**, named for what
        they are rather than merged into one score: ``ConfidenceBreakdown.aggregate``
        is deliberately ``None`` in this system (an uncalibrated blend read as a
        probability of guilt is exactly what the rule layer refuses to mint), so the
        banner shows the classifier's mean score across the supporting observations
        and the episode's temporal consistency separately. A component the reasoner
        did not measure is omitted, never rendered as ``0%``.
        """

        event = episode.event
        lines = [
            f"Track: {rider.rider_track_id} · Bike: {rider.motorcycle_track_id}",
            f"Sustained {episode.trigger_seconds - episode.start_seconds:.2f}s"
            + (
                f" (needs {episode.threshold_seconds:.2f}s)"
                if episode.threshold_seconds is not None
                else ""
            ),
        ]
        confidence = event.confidence
        if confidence.classifier is not None:
            lines.append(f"Classifier confidence: {_percent(confidence.classifier)}")
        if confidence.temporal_consistency is not None:
            lines.append(f"Temporal consistency: {_percent(confidence.temporal_consistency)}")
        lines.append(f"t = {_clock(episode.trigger_seconds)}")
        lines.append(f"ID: {event.event_id}")
        return OverlayBanner(
            title="NO HELMET",
            icon="⚠",  # warning sign
            lines=tuple(lines),
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )


def register_no_helmet_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the no-helmet provider factory under its violation kind.

    The plug-in call: a driver that ran the no-helmet rule asks the registry for a
    ``"no_helmet"`` provider, passing the captured frames + confirmed events.
    """

    registry.register("no_helmet", NoHelmetOverlayProvider)

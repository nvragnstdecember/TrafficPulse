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
* the **rider** box (``SUBJECT`` -> green) captioned ``Rider / Track: <id>``;
* the exact **head region** box (``REGION`` -> yellow) captioned with the classifier
  label + confidence (``No Helmet`` / ``97%``);
* the **association chain** head -> rider -> motorcycle as a polyline.

Temporal state drives the two shared axes, not colour choices here:

* while the reasoner is accumulating support (a ``no_helmet`` read that has not yet
  confirmed) the rider carries ``OBSERVING`` and shows a "Collecting evidence…" status;
* once a ``no_helmet`` :class:`~trafficpulse.contracts.ConfirmedEvent` for the rider
  has triggered (frame media-time >= its ``trigger_at``), the rider, motorcycle, and
  head flip to ``CONFIRMED`` (the theme renders them red / bright red) and a banner
  pins ``⚠ NO HELMET`` with the track, timestamp, and violation id.

Determinism / performance
-------------------------
Pure lookups by ``frame_index`` and a timestamp comparison per rider -- O(riders on
the frame), no model inference. The provider owns no drawing code and names no
colour; it speaks only emphasis + alert, which the theme resolves.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def _center(box: tuple[float, float, float, float]) -> OverlayPoint:
    return OverlayPoint(x=(box[0] + box[2]) / 2.0, y=(box[1] + box[3]) / 2.0)


def _clock(media_seconds: float) -> str:
    minutes, seconds = divmod(max(0.0, media_seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:06.3f}"


class NoHelmetOverlayProvider:
    """Turns captured no-helmet metadata into overlay elements (see module docstring)."""

    violation_kind = "no_helmet"

    def __init__(
        self,
        frames: Sequence[HelmetOverlayFrame],
        events: Sequence[ConfirmedEvent] = (),
    ) -> None:
        self._by_index: dict[int, HelmetOverlayFrame] = {f.frame_index: f for f in frames}
        # Per rider track: the earliest confirmed no-helmet trigger (media seconds)
        # and the confirming event, so a frame at/after the trigger reads CONFIRMED.
        self._confirmed: dict[str, tuple[float, ConfirmedEvent]] = {}
        for event in events:
            if event.violation_type is not ViolationType.NO_HELMET:
                continue
            trigger = (event.trigger_at - _MEDIA_TIME_EPOCH).total_seconds()
            for track_id in event.track_ids:
                current = self._confirmed.get(track_id)
                if current is None or trigger < current[0]:
                    self._confirmed[track_id] = (trigger, event)

    # --- OverlayProvider protocol -------------------------------------------
    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        captured = self._by_index.get(frame.frame_index)
        if captured is None:
            return ()
        elements: list[OverlayElement] = []
        banners: list[OverlayBanner] = []
        for rider in captured.riders:
            confirmed = self._confirmed.get(rider.rider_track_id)
            if confirmed is not None and frame.media_seconds >= confirmed[0]:
                trigger_seconds, event = confirmed
                elements.extend(self._rider_elements(rider, OverlayAlert.CONFIRMED))
                banners.append(self._banner(rider, event, trigger_seconds))
            elif rider.helmet_label == "no_helmet":
                elements.extend(self._rider_elements(rider, OverlayAlert.OBSERVING))
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
        self, rider: HelmetOverlayRider, alert: OverlayAlert
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
        if alert is OverlayAlert.OBSERVING:
            rider_lines = (*rider_lines, "Collecting evidence…")
        out.append(
            OverlayBox(
                bounds=rider_box,
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.SUBJECT,
                caption=OverlayCaption(lines=rider_lines, prefer=Corner.TOP_RIGHT),
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

    def _banner(
        self, rider: HelmetOverlayRider, event: ConfirmedEvent, trigger_seconds: float
    ) -> OverlayBanner:
        return OverlayBanner(
            title="NO HELMET",
            icon="⚠",  # warning sign
            lines=(
                f"Track: {rider.rider_track_id} · Bike: {rider.motorcycle_track_id}",
                f"t = {_clock(trigger_seconds)}",
                f"ID: {event.event_id}",
            ),
            alert=OverlayAlert.CONFIRMED,
            corner=Corner.TOP_LEFT,
        )


def register_no_helmet_overlay(registry: OverlayProviderRegistry) -> None:
    """Register the no-helmet provider factory under its violation kind.

    The plug-in call: a driver that ran the no-helmet rule asks the registry for a
    ``"no_helmet"`` provider, passing the captured frames + confirmed events.
    """

    registry.register("no_helmet", NoHelmetOverlayProvider)

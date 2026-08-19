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
Every rider always contributes three boxes and the association chain that joins
them -- the **motorcycle** (``OBJECT`` -> blue), the **rider** (``SUBJECT`` -> cyan,
amber while observing), and the exact **head region** the classifier saw
(``REGION`` -> yellow, red the moment it reads ``no_helmet``). Detections are never
hidden.

*Captions*, however, are rationed by reasoning state -- see
:meth:`NoHelmetOverlayProvider._rider_elements`. A frame of ordinary traffic is
mostly boxes and one classifier readout per rider; text appears where the system is
actually asserting something. Captioning all three boxes on every rider, as this
provider first did, put three chips per motorcycle on screen -- most of them
restating what the colours already said, and each one shoved away from its subject
by the collision solver until the frame was a wall of disconnected labels.

Temporal state drives the two shared axes, not colour choices here:

* while the reasoner is accumulating support (a ``no_helmet`` read that has not yet
  confirmed) the rider carries ``OBSERVING`` and shows "Observing" + the meter;
* once a ``no_helmet`` :class:`~trafficpulse.contracts.ConfirmedEvent` for the rider
  has triggered (frame media-time >= its ``trigger_at``), the rider, motorcycle, and
  head flip to ``CONFIRMED`` (the theme renders them red / bright red) and a banner
  pins ``⚠ NO HELMET``.

Stability is presentation, not data
-----------------------------------
Drawn geometry and the displayed confidence are temporally smoothed through
:mod:`~trafficpulse.overlay.smoothing` when the capture is loaded, so boxes settle
and a percentage stops churning. The capture itself is never modified and nothing
downstream sees the smoothed values -- ``_DrawnRider`` exists only to keep the two
apart. Labels are deliberately *not* smoothed: the caption must state what the
classifier said about that frame.

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
Their bare-headed frames still read ``OBSERVING``, but no progress is drawn, because
no episode exists to measure progress against and inventing a denominator would
assert reasoning the system never performed.

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
from ...pipeline.helmet_observer import HelmetOverlayFrame
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
from ..smoothing import SmoothingConfig, StreamSmoother

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


@dataclass(frozen=True)
class _DrawnRider:
    """One rider's capture as it will be *drawn*: geometry and score smoothed.

    A presentation twin of
    :class:`~trafficpulse.pipeline.helmet_observer.HelmetOverlayRider`. The labels
    and track ids are carried through untouched -- only the coordinates and the
    displayed confidence are temporally smoothed, and only here. Nothing downstream
    of the renderer ever sees these values.
    """

    rider_track_id: str
    rider_bbox: tuple[float, float, float, float]
    motorcycle_track_id: str
    motorcycle_bbox: tuple[float, float, float, float]
    head_bbox: tuple[float, float, float, float] | None
    helmet_label: str
    confidence: float | None


def _smooth_capture(
    frames: Sequence[HelmetOverlayFrame], config: SmoothingConfig | None
) -> dict[int, tuple[_DrawnRider, ...]]:
    """Precompute the drawable, temporally smoothed view of a whole capture.

    Done once up front, in frame order, so :meth:`elements_for_frame` stays a pure
    lookup: the provider remains stateless per frame, order-independent, and
    replayable, while the drawn geometry still benefits from a stateful filter.

    Smoothing is keyed by track *and role* -- a rider's own box, its motorcycle's
    box, its head crop and its score are four independent streams -- so one rider's
    jitter never bleeds into another's, and a rider that swaps motorcycle does not
    drag the old vehicle's geometry along.
    """

    smoother = StreamSmoother(config)
    out: dict[int, tuple[_DrawnRider, ...]] = {}
    for captured in sorted(frames, key=lambda f: f.frame_index):
        index = captured.frame_index
        drawn: list[_DrawnRider] = []
        for rider in captured.riders:
            key = rider.rider_track_id
            head = (
                smoother.box(f"{key}#head", index, rider.head_bbox)
                if rider.head_bbox is not None
                else None
            )
            drawn.append(
                _DrawnRider(
                    rider_track_id=rider.rider_track_id,
                    rider_bbox=smoother.box(f"{key}#rider", index, rider.rider_bbox),
                    motorcycle_track_id=rider.motorcycle_track_id,
                    motorcycle_bbox=smoother.box(
                        f"{key}#bike:{rider.motorcycle_track_id}", index, rider.motorcycle_bbox
                    ),
                    head_bbox=head,
                    helmet_label=rider.helmet_label,
                    confidence=smoother.value(f"{key}#score", index, rider.confidence),
                )
            )
        out[index] = tuple(drawn)
    return out


class NoHelmetOverlayProvider:
    """Turns captured no-helmet metadata into overlay elements (see module docstring)."""

    violation_kind = "no_helmet"

    def __init__(
        self,
        frames: Sequence[HelmetOverlayFrame],
        events: Sequence[ConfirmedEvent] = (),
        *,
        smoothing: SmoothingConfig | None = None,
    ) -> None:
        self._by_index: dict[int, tuple[_DrawnRider, ...]] = _smooth_capture(frames, smoothing)
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
    def has_content(self) -> bool:
        """Only the per-frame capture: this violation has no scene geometry.

        Empty when the run's observer had capture disabled, or classified nothing --
        which is what makes "the rule ran but there is no annotated video" a state
        the driver can still recognise.
        """

        return bool(self._by_index)

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        captured = self._by_index.get(frame.frame_index)
        if captured is None:
            return ()
        elements: list[OverlayElement] = []
        banners: list[OverlayBanner] = []
        for rider in captured:
            episode = self._episodes.get(rider.rider_track_id)
            if episode is not None and frame.media_seconds >= episode.trigger_seconds:
                elements.extend(
                    self._rider_elements(rider, OverlayAlert.CONFIRMED, episode=episode)
                )
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
            key = banner.details[-1] if banner.details else banner.title
            if key not in seen:
                seen.add(key)
                elements.append(banner)
        return tuple(elements)

    # --- element construction -----------------------------------------------
    def _rider_elements(
        self,
        rider: _DrawnRider,
        alert: OverlayAlert,
        *,
        progress: tuple[float, str] | None = None,
        episode: _Episode | None = None,
    ) -> list[OverlayElement]:
        """The boxes, captions, and chain for one rider at one reasoning state.

        Captions are allocated by **progressive disclosure**: every box is always
        drawn, but text is spent only where it tells the analyst something they do
        not already know from the visual language.

        * the *motorcycle* box carries no caption -- blue already means "vehicle",
          the chain already says which rider it belongs to, and its track id is on
          the banner if the rider is ever confirmed. A chip per vehicle per frame is
          the single largest source of clutter and none of it is new information;
        * the *rider* box carries the reasoning state, and only once there is a
          state worth reporting: silent while nothing is being argued, "Observing"
          plus the evidence meter while support accumulates, the violation call-out
          once confirmed;
        * the *head region* carries the classifier readout while the outcome is
          still open, and goes silent on confirmation -- at that point the same
          number appears on the rider caption and again on the banner, and printing
          it three times is what made the confirmed frame unreadable.

        A label the classifier abstained on (no score) is likewise not printed: an
        "Uncertain" chip with no figure costs a caption and conveys nothing the
        yellow region box does not already say.
        """

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
                key=rider.motorcycle_track_id,
            )
        )
        rider_caption: OverlayCaption | None = None
        if alert is OverlayAlert.OBSERVING:
            lines = ["Observing", f"Track {rider.rider_track_id}"]
            fraction: float | None = None
            if progress is not None:
                fraction, readout = progress
                lines.append(readout)
            rider_caption = OverlayCaption(
                lines=tuple(lines), prefer=Corner.TOP_RIGHT, progress=fraction
            )
        elif alert is OverlayAlert.CONFIRMED:
            # The *episode's* classifier confidence, not this frame's live reading.
            # Both are truthful, but they differ by a point or two, and two
            # different "confidences" a few centimetres apart on the same frame
            # read as an inconsistency rather than as two different measurements.
            # The reasoned, auditable figure is the one on the event, so the chip
            # and the banner quote the same number.
            confirmed_score = (
                episode.event.confidence.classifier if episode is not None else None
            )
            rider_caption = OverlayCaption(
                lines=("NO HELMET", f"Track {rider.rider_track_id}"),
                metric=None if confirmed_score is None else _percent(confirmed_score),
                prefer=Corner.TOP_RIGHT,
                progress=1.0,
            )
        out.append(
            OverlayBox(
                bounds=rider_box,
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.SUBJECT,
                caption=rider_caption,
                key=rider.rider_track_id,
            )
        )
        if head_box is not None:
            # On confirmation the reading has moved to the rider caption + banner.
            head_caption = (
                None
                if alert is OverlayAlert.CONFIRMED or rider.confidence is None
                else OverlayCaption(
                    lines=(_LABEL_DISPLAY.get(rider.helmet_label, rider.helmet_label),),
                    metric=_percent(rider.confidence),
                    prefer=Corner.BOTTOM_LEFT,
                )
            )
            out.append(
                OverlayBox(
                    bounds=head_box,
                    emphasis=OverlayEmphasis.REGION,
                    alert=alert,
                    layer=OverlayLayer.REGION,
                    caption=head_caption,
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

    def _banner(self, rider: _DrawnRider, episode: _Episode) -> OverlayBanner:
        """The pinned confirmation headline: what was confirmed, and on what evidence.

        Ranked into three tiers rather than one wall of text. The headline answers
        *what* and *how sure*; the body answers *who* and *for how long*; everything
        an analyst needs only when auditing a specific event -- the vehicle, the
        media timestamp, the event id -- drops into ``details``, still present but
        visually subordinate. The seven equal-weight lines this replaces made the
        banner the largest object on the frame and buried the one figure a reviewer
        actually reads first.

        The figures are the event's **published components**, named for what they
        are rather than merged: ``ConfidenceBreakdown.aggregate`` is deliberately
        ``None`` in this system (an uncalibrated blend read as a probability of guilt
        is exactly what the rule layer refuses to mint), so the classifier's mean
        score across the supporting observations is the headline metric and temporal
        consistency stays a separate, labelled figure. A component the reasoner did
        not measure is omitted, never rendered as ``0%``.
        """

        event = episode.event
        confidence = event.confidence
        sustained = episode.trigger_seconds - episode.start_seconds
        # The vehicle the *event* was attributed to, not whatever this frame's
        # association happens to say. A banner describes one fixed, already-decided
        # violation, so every figure on it must be fixed too -- reading the live
        # association made the bike id churn frame to frame under an unchanging
        # event id, which looks like the system changing its mind.
        bike = next(
            (t for t in event.track_ids if t != rider.rider_track_id),
            rider.motorcycle_track_id,
        )
        needed = (
            f" of {episode.threshold_seconds:.2f}s required"
            if episode.threshold_seconds is not None
            else ""
        )
        lines = [
            f"Track {rider.rider_track_id} · sustained {sustained:.2f}s{needed}",
        ]
        if confidence.temporal_consistency is not None:
            lines.append(
                f"Temporal consistency {_percent(confidence.temporal_consistency)}"
            )
        return OverlayBanner(
            title="NO HELMET",
            metric=(
                _percent(confidence.classifier) if confidence.classifier is not None else None
            ),
            icon="⚠",  # warning sign
            lines=tuple(lines),
            details=(
                f"Bike {bike} · t {_clock(episode.trigger_seconds)}",
                event.event_id,
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

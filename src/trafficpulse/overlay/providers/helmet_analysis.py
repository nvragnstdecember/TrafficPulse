"""The helmet-**analysis** overlay: what was classified, and what may not be claimed.

The visual counterpart to
:class:`~trafficpulse.pipeline.helmet_analysis.HelmetAnalysisObserver`. It draws the same
geometry the no-helmet provider draws -- motorcycle, rider, the exact head crop the
classifier saw, and the association chain -- from the same already-captured metadata, so
nothing is recomputed and no model re-runs.

Three things make it deliberately different from
:class:`~trafficpulse.overlay.providers.no_helmet.NoHelmetOverlayProvider`, and each is a
requirement rather than a preference.

**1. It can never draw a confirmation.** ``OverlayAlert.CONFIRMED`` is the visual language
of "a violation has been confirmed for this element", and an analysis confirms nothing.
This provider emits only ``NONE`` and ``OBSERVING``; there is no code path to
``CONFIRMED``, and no banner announcing a violation. A ``no_helmet`` reading is drawn in
the *observing* register -- noticed, not concluded -- which is precisely what it is.

**2. It shows the stabilized label, and says so.** Raw per-frame labels flip on the
majority of real tracks (P4-U10 §5.1(7)). Drawing them unfiltered produces a caption that
alternates several times a second, which is unreadable and misrepresents the system's
actual per-track reading. The label drawn is therefore the
:mod:`~trafficpulse.observations.helmet_stability` window vote, with its agreement shown
beside it so a viewer can see how well-supported it is -- and an unsettled track is marked
as such rather than presented as a firm call.

**3. Multi-rider is stated on the vehicle, in words.** When two or more riders are
associated with one motorcycle on a frame, the vehicle carries
``MULTI-RIDER / DRIVER UNRESOLVED`` and every rider on it is captioned as unresolved. No
driver is chosen -- not the front-most, not the largest, not the lowest, not the
first-tracked. ``rider_slot`` is ``UNKNOWN`` for these riders by design (the shipped
tracker has no velocity, so which end of the bike is the front is genuinely unknown), and
this provider renders that limitation instead of papering over it. 42.4% of the frozen
test corpus and 81% of crops in a real congestion clip are in this state, so it is the
common case, not an edge case.

A standing banner names the mode on every frame, so a still pulled out of the video
cannot be mistaken for enforcement output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...observations.helmet_stability import (
    HelmetSample,
    HelmetStabilizationConfig,
    StabilizedSample,
    stabilized_index,
)
from ...pipeline.helmet_analysis import HelmetAnalysisObserver
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

#: Native classifier labels -> what a viewer reads. Mirrors the no-helmet provider's
#: table rather than importing it: the two are the same today by coincidence of
#: vocabulary, not by contract, and a shared constant would couple an analysis surface
#: to a violation surface.
_LABEL_DISPLAY = {
    "helmet": "Helmet",
    "no_helmet": "No Helmet",
    "turban": "Turban",
    "uncertain": "Uncertain",
}

_MULTI_RIDER_CAPTION = ("MULTI-RIDER", "DRIVER UNRESOLVED")


def _center(box: tuple[float, float, float, float]) -> OverlayPoint:
    return OverlayPoint(x=(box[0] + box[2]) / 2.0, y=(box[1] + box[3]) / 2.0)


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


@dataclass(frozen=True)
class _DrawnRider:
    """One rider as it will be drawn: geometry smoothed, reading stabilized.

    A presentation value only. Track ids and labels are carried through untouched; the
    coordinates are temporally smoothed exactly as the no-helmet provider smooths them,
    and the *reading* is the stabilizer's window vote rather than the frame's raw label.
    """

    rider_track_id: str
    rider_bbox: tuple[float, float, float, float]
    motorcycle_track_id: str
    motorcycle_bbox: tuple[float, float, float, float]
    head_bbox: tuple[float, float, float, float] | None
    reading: StabilizedSample
    rider_count: int

    @property
    def multi_rider(self) -> bool:
        return self.rider_count > 1


def _samples(frames: Sequence[HelmetOverlayFrame]) -> list[HelmetSample]:
    """Every capture entry as a stabilizer sample (gated crops included).

    A gated crop is captured with the label ``uncertain`` and no score, and it votes:
    a rider the classifier mostly could not read must not be represented by the one
    frame where it could.
    """

    return [
        HelmetSample(
            frame_index=frame.frame_index,
            track_id=rider.rider_track_id,
            label=rider.helmet_label,
            confidence=rider.confidence,
        )
        for frame in frames
        for rider in frame.riders
    ]


def _prepare(
    frames: Sequence[HelmetOverlayFrame],
    smoothing: SmoothingConfig | None,
    stabilization: HelmetStabilizationConfig | None,
) -> dict[int, tuple[_DrawnRider, ...]]:
    """Precompute the whole drawable capture once, in frame order.

    Done up front so :meth:`HelmetAnalysisOverlayProvider.elements_for_frame` stays a
    pure lookup -- the provider is then stateless per frame, order-independent and
    replayable, while the drawn geometry still benefits from a stateful filter.
    """

    smoother = StreamSmoother(smoothing)
    readings = stabilized_index(_samples(frames), config=stabilization)
    out: dict[int, tuple[_DrawnRider, ...]] = {}
    for captured in sorted(frames, key=lambda frame: frame.frame_index):
        index = captured.frame_index
        # Riders sharing a motorcycle *on this frame* -- the same count the observation
        # layer turns into ``rider_slot``, recomputed from captured ids only.
        per_bike: dict[str, int] = {}
        for rider in captured.riders:
            per_bike[rider.motorcycle_track_id] = per_bike.get(rider.motorcycle_track_id, 0) + 1

        drawn: list[_DrawnRider] = []
        for rider in captured.riders:
            key = rider.rider_track_id
            reading = readings.get((index, key))
            if reading is None:  # unreachable: every capture entry produced a sample
                continue
            head = (
                smoother.box(f"{key}#head", index, rider.head_bbox)
                if rider.head_bbox is not None
                else None
            )
            drawn.append(
                _DrawnRider(
                    rider_track_id=key,
                    rider_bbox=smoother.box(f"{key}#rider", index, rider.rider_bbox),
                    motorcycle_track_id=rider.motorcycle_track_id,
                    motorcycle_bbox=smoother.box(
                        f"{key}#bike:{rider.motorcycle_track_id}", index, rider.motorcycle_bbox
                    ),
                    head_bbox=head,
                    reading=reading,
                    rider_count=per_bike[rider.motorcycle_track_id],
                )
            )
        out[index] = tuple(drawn)
    return out


class HelmetAnalysisOverlayProvider:
    """Draws helmet *analysis* -- classification with the enforcement limits on screen."""

    #: The registry key. Not a ``ViolationType``: this provider describes no violation,
    #: and naming it after one would make an analysis run indistinguishable from a rule
    #: run in the compositor's ordering and in anything that reads the kind.
    violation_kind = "helmet_analysis"

    def __init__(
        self,
        frames: Sequence[HelmetOverlayFrame],
        *,
        smoothing: SmoothingConfig | None = None,
        stabilization: HelmetStabilizationConfig | None = None,
    ) -> None:
        self._by_index = _prepare(frames, smoothing, stabilization)

    # --- OverlayProvider protocol -------------------------------------------
    def has_content(self) -> bool:
        """Whether the run captured anything to draw (no scene geometry is involved)."""

        return bool(self._by_index)

    def elements_for_frame(self, frame: OverlayFrameRef) -> Sequence[OverlayElement]:
        captured = self._by_index.get(frame.frame_index)
        if not captured:
            return ()
        elements: list[OverlayElement] = []
        drawn_bikes: set[str] = set()
        for rider in captured:
            elements.extend(self._rider_elements(rider, drawn_bikes))
        elements.append(self._mode_banner(captured))
        return tuple(elements)

    # --- element construction -----------------------------------------------
    def _rider_elements(
        self, rider: _DrawnRider, drawn_bikes: set[str]
    ) -> list[OverlayElement]:
        """One rider's boxes, captions and chain.

        Captions are rationed the way the no-helmet provider rations them -- every box
        is always drawn, text is spent only where it says something the colour does not.
        The one addition is the multi-rider statement, which is carried **once per
        motorcycle** rather than once per rider: it is a fact about the vehicle, and
        repeating it under every person on the bike is how the caption solver ends up
        painting a wall of identical chips over a crowd.
        """

        out: list[OverlayElement] = []
        alert = (
            OverlayAlert.OBSERVING
            if rider.reading.label == "no_helmet" and not rider.multi_rider
            else OverlayAlert.NONE
        )

        bike_caption: OverlayCaption | None = None
        if rider.multi_rider and rider.motorcycle_track_id not in drawn_bikes:
            bike_caption = OverlayCaption(
                lines=(*_MULTI_RIDER_CAPTION, f"{rider.rider_count} riders"),
                prefer=Corner.BOTTOM_RIGHT,
            )
        drawn_bikes.add(rider.motorcycle_track_id)
        out.append(
            OverlayBox(
                bounds=rider.motorcycle_bbox,
                emphasis=OverlayEmphasis.OBJECT,
                alert=OverlayAlert.NONE,
                layer=OverlayLayer.OBJECT,
                caption=bike_caption,
                key=rider.motorcycle_track_id,
            )
        )

        out.append(
            OverlayBox(
                bounds=rider.rider_bbox,
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.SUBJECT,
                caption=self._rider_caption(rider),
                key=rider.rider_track_id,
            )
        )

        if rider.head_bbox is not None:
            out.append(
                OverlayBox(
                    bounds=rider.head_bbox,
                    emphasis=OverlayEmphasis.REGION,
                    alert=alert,
                    layer=OverlayLayer.REGION,
                    caption=self._head_caption(rider),
                    key=f"{rider.rider_track_id}-head",
                )
            )

        head_anchor = (
            _center(rider.head_bbox) if rider.head_bbox is not None else _center(rider.rider_bbox)
        )
        out.append(
            OverlayLink(
                points=(head_anchor, _center(rider.rider_bbox), _center(rider.motorcycle_bbox)),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.LINK,
            )
        )
        return out

    def _rider_caption(self, rider: _DrawnRider) -> OverlayCaption:
        """The rider's identity and, when it matters, why nothing may be concluded.

        A multi-rider rider is captioned "Driver unresolved" and *not* with a helmet
        state, even though one was classified: putting a helmet call on a person whose
        role is unknown, next to a vehicle, is the exact reading a viewer would take as
        an accusation. The classification is still visible -- it is on the head crop,
        where it plainly describes a crop rather than a person's liability.
        """

        lines = [f"Track {rider.rider_track_id}"]
        if rider.multi_rider:
            lines.append("Driver unresolved")
        elif not rider.reading.settled:
            lines.append("Stabilizing")
        return OverlayCaption(lines=tuple(lines), prefer=Corner.TOP_RIGHT)

    def _head_caption(self, rider: _DrawnRider) -> OverlayCaption | None:
        """The classifier readout for the crop, or nothing when there is no measurement.

        Shows the **stabilized** label with the window's mean score, plus the agreement
        for a reading that is not unanimous -- so a 3-of-5 call never looks like a 5-of-5
        one. A crop the classifier never scored prints no chip at all: an "Uncertain"
        label with no figure costs a caption and says nothing the yellow region box does
        not already say.
        """

        reading = rider.reading
        if reading.confidence is None:
            return None
        lines = [_LABEL_DISPLAY.get(reading.label, reading.label)]
        if reading.agreement < 1.0:
            lines.append(f"{_percent(reading.agreement)} of {reading.samples} frames")
        return OverlayCaption(
            lines=tuple(lines), metric=_percent(reading.confidence), prefer=Corner.BOTTOM_LEFT
        )

    def _mode_banner(self, captured: Sequence[_DrawnRider]) -> OverlayBanner:
        """The standing "this is analysis, not enforcement" statement.

        Present on **every** drawn frame, deliberately, so a screenshot cannot be
        separated from its disclaimer. ``alert`` is ``NONE``: this is a mode notice, and
        styling it as an alert would reintroduce exactly the "something was confirmed"
        reading the provider exists to avoid.
        """

        unresolved = sum(1 for rider in captured if rider.multi_rider)
        lines = [f"{len(captured)} rider(s) classified"]
        if unresolved:
            lines.append(f"{unresolved} unresolved (multi-rider)")
        return OverlayBanner(
            title="HELMET ANALYSIS",
            lines=tuple(lines),
            details=("No violation decision is made in this mode",),
            alert=OverlayAlert.NONE,
            corner=Corner.TOP_LEFT,
        )


def _from_observer(
    observer: HelmetAnalysisObserver, events: object = ()
) -> HelmetAnalysisOverlayProvider:
    """Build the provider from the observer the engine surfaces.

    ``events`` is accepted and **ignored**: the overlay driver passes a run's whole
    event list to every provider, and this one has nothing to do with any of them. An
    analysis draws what was classified; if some other rule confirmed something on the
    same run, that rule's own provider draws it.
    """

    return HelmetAnalysisOverlayProvider(
        observer.overlay_frames(), stabilization=observer.stabilization
    )


def register_helmet_analysis_overlay(registry: OverlayProviderRegistry) -> None:
    """Register this provider against the analysis observer's type."""

    registry.register(
        HelmetAnalysisOverlayProvider.violation_kind,
        _from_observer,
        source_type=HelmetAnalysisObserver,
    )

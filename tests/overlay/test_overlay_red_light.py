"""The red-light overlay provider: capture + events -> generic elements (H13).

The provider must explain the reasoning, not contradict it. Its central obligation
is that the caption beside a confirmed violation shows the state that was **latched
at the stop line** -- an overlay reading "GREEN" next to a red-light event would make
a correct system look broken.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import ConfidenceBreakdown, ConfirmedEvent, MeasuredValue
from trafficpulse.contracts.enums import SignalState, ViolationType
from trafficpulse.overlay.metadata import OverlayAlert, OverlayBanner, OverlayBox, OverlayLink
from trafficpulse.overlay.providers.red_light import (
    VIOLATION_KIND,
    RedLightOverlayProvider,
    register_red_light_overlay,
)
from trafficpulse.overlay.registry import OverlayFrameRef, OverlayProviderRegistry
from trafficpulse.pipeline.red_light import RedLightOverlayCapture, RedLightTrackFrame

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_STOP_LINE = ((100.0, 120.0), (220.0, 120.0))
_POLYGON = ((100.0, 150.0), (220.0, 150.0), (220.0, 235.0), (100.0, 235.0))


def _frame_ref(index: int, seconds: float) -> OverlayFrameRef:
    return OverlayFrameRef(
        camera_id="cam-1", frame_index=index, media_seconds=seconds, width=320, height=240
    )


def _tracked(
    index: int,
    seconds: float,
    *,
    inside: bool = True,
    entered_on_red: bool = True,
    state: SignalState = SignalState.RED,
    track_id: str = "t-1",
) -> RedLightTrackFrame:
    return RedLightTrackFrame(
        frame_index=index,
        media_seconds=seconds,
        track_id=track_id,
        bbox=(140.0, 150.0, 180.0, 190.0),
        is_inside=inside,
        entered_on_red=entered_on_red,
        entry_state=state,
    )


def _capture(*frames: RedLightTrackFrame) -> RedLightOverlayCapture:
    return RedLightOverlayCapture(
        stop_line=_STOP_LINE, zone_polygon=_POLYGON, frames=list(frames)
    )


def _event(*, start: float, trigger: float, track_id: str = "t-1") -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id="evt-red-1",
        violation_type=ViolationType.RED_LIGHT_JUMPING,
        camera_id="cam-1",
        track_ids=(track_id,),
        start_at=_EPOCH + timedelta(seconds=start),
        trigger_at=_EPOCH + timedelta(seconds=trigger),
        rule_id="red-light-jumping-v1",
        created_at=_EPOCH + timedelta(seconds=trigger),
        confidence=ConfidenceBreakdown(),
        thresholds=(MeasuredValue(name="min_persistence", value=0.4, unit="seconds"),),
    )


def _boxes(elements: object) -> list[OverlayBox]:
    return [e for e in elements if isinstance(e, OverlayBox)]  # type: ignore[union-attr]


def _banners(elements: object) -> list[OverlayBanner]:
    return [e for e in elements if isinstance(e, OverlayBanner)]  # type: ignore[union-attr]


def _links(elements: object) -> list[OverlayLink]:
    return [e for e in elements if isinstance(e, OverlayLink)]  # type: ignore[union-attr]


# --- scene geometry ------------------------------------------------------------------
def test_the_stop_line_and_junction_are_drawn_on_every_frame() -> None:
    # A mis-drawn stop line is the most likely calibration error; it should be
    # visible on frame one rather than deduced from an absence of events.
    provider = RedLightOverlayProvider(_capture())

    links = _links(provider.elements_for_frame(_frame_ref(0, 0.0)))

    assert len(links) == 2
    # The polygon is a closed ring -- no new element type was added for it.
    ring = max(links, key=lambda link: len(link.points))
    assert len(ring.points) == len(_POLYGON) + 1
    assert (ring.points[0].x, ring.points[0].y) == (ring.points[-1].x, ring.points[-1].y)


def test_a_degenerate_polygon_still_draws_the_stop_line() -> None:
    capture = RedLightOverlayCapture(stop_line=_STOP_LINE, zone_polygon=((0.0, 0.0),))

    links = _links(RedLightOverlayProvider(capture).elements_for_frame(_frame_ref(0, 0.0)))

    assert len(links) == 1


# --- the latched state -----------------------------------------------------------------
def test_the_caption_shows_the_state_latched_at_the_stop_line() -> None:
    # Not the current signal state: an overlay that contradicts the reasoning it
    # exists to explain is worse than no overlay.
    provider = RedLightOverlayProvider(
        _capture(_tracked(10, 1.7, state=SignalState.RED)),
        [_event(start=1.7, trigger=2.1)],
    )

    box = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]

    assert box.caption is not None
    assert box.caption.metric == "RED"
    assert "Entered on red" in box.caption.lines


def test_a_vehicle_that_did_not_enter_on_red_is_not_alerted() -> None:
    provider = RedLightOverlayProvider(
        _capture(_tracked(10, 1.7, entered_on_red=False, state=SignalState.GREEN))
    )

    box = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]

    assert box.alert is OverlayAlert.NONE
    assert _banners(provider.elements_for_frame(_frame_ref(10, 1.7))) == []


# --- the observing -> confirmed progression ---------------------------------------------
def test_the_alert_escalates_from_observing_to_confirmed_at_the_trigger() -> None:
    capture = _capture(_tracked(10, 1.7), _tracked(14, 2.1), _tracked(18, 2.5))
    provider = RedLightOverlayProvider(capture, [_event(start=1.7, trigger=2.1)])

    before = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]
    at_trigger = _boxes(provider.elements_for_frame(_frame_ref(14, 2.1)))[0]
    after = _boxes(provider.elements_for_frame(_frame_ref(18, 2.5)))[0]

    assert before.alert is OverlayAlert.OBSERVING
    assert at_trigger.alert is OverlayAlert.CONFIRMED
    assert after.alert is OverlayAlert.CONFIRMED


def test_the_debounce_meter_is_read_off_the_event_not_invented() -> None:
    capture = _capture(_tracked(10, 1.7), _tracked(12, 1.9))
    provider = RedLightOverlayProvider(capture, [_event(start=1.7, trigger=2.1)])

    at_start = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]
    midway = _boxes(provider.elements_for_frame(_frame_ref(12, 1.9)))[0]

    assert at_start.caption is not None and at_start.caption.progress == 0.0
    # 0.2 s elapsed against the event's own 0.4 s min_persistence threshold.
    assert midway.caption is not None and midway.caption.progress == pytest.approx(0.5)


def test_a_track_the_reasoner_never_confirmed_gets_no_meter() -> None:
    # Inventing a denominator would draw reasoning the system never performed.
    provider = RedLightOverlayProvider(_capture(_tracked(10, 1.7)), events=[])

    box = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]

    assert box.caption is not None and box.caption.progress is None
    assert box.alert is OverlayAlert.OBSERVING  # entered on red, but never confirmed


# --- the banner --------------------------------------------------------------------------
def test_a_banner_appears_from_the_trigger_and_names_the_latched_state() -> None:
    capture = _capture(_tracked(10, 1.7), _tracked(14, 2.1))
    provider = RedLightOverlayProvider(capture, [_event(start=1.7, trigger=2.1)])

    assert _banners(provider.elements_for_frame(_frame_ref(10, 1.7))) == []
    banner = _banners(provider.elements_for_frame(_frame_ref(14, 2.1)))[0]

    assert banner.title == "RED-LIGHT JUMPING"
    assert banner.metric == "RED"
    assert any("t-1" in detail for detail in banner.details)


# --- determinism + registration --------------------------------------------------------
def test_elements_are_ordered_deterministically_for_a_frame() -> None:
    capture = _capture(
        _tracked(10, 1.7, track_id="t-b"),
        _tracked(10, 1.7, track_id="t-a"),
    )
    provider = RedLightOverlayProvider(capture)

    keys = [box.key for box in _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))]

    assert keys == ["t-a", "t-b"]


def test_a_frame_with_no_tracked_vehicle_still_draws_the_geometry() -> None:
    provider = RedLightOverlayProvider(_capture(_tracked(10, 1.7)))

    elements = provider.elements_for_frame(_frame_ref(99, 9.9))

    assert _boxes(elements) == []
    assert len(_links(elements)) == 2


def test_the_provider_registers_under_its_violation_kind() -> None:
    registry = OverlayProviderRegistry()
    register_red_light_overlay(registry)

    provider = registry.create(VIOLATION_KIND, _capture(), [])

    assert provider.violation_kind == VIOLATION_KIND
    assert provider.violation_kind == ViolationType.RED_LIGHT_JUMPING.value


def test_events_of_other_violations_are_ignored() -> None:
    other = _event(start=1.0, trigger=1.5).model_copy(
        update={"violation_type": ViolationType.WRONG_WAY}
    )
    provider = RedLightOverlayProvider(_capture(_tracked(10, 1.7)), [other])

    box = _boxes(provider.elements_for_frame(_frame_ref(10, 1.7)))[0]

    assert box.alert is OverlayAlert.OBSERVING  # never escalates on a foreign event

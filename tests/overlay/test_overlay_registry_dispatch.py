"""Registry-driven provider dispatch, and the three providers R6 added.

The claim under test: the overlay layer resolves a provider from the **capture a
rule produced**, with no violation named in the driver, and every shipped violation
has a faithful visual representation rather than a generic label.

Each provider is asserted on what it actually draws -- the vehicle box from the
tracked geometry, the measured heading, the configured zone, the associated riders
-- because "renders something" is not the requirement; "renders what the engine
concluded" is.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import ConfirmedEvent, MeasuredValue
from trafficpulse.contracts.enums import SignalState, ViolationType
from trafficpulse.overlay.metadata import (
    OverlayAlert,
    OverlayBanner,
    OverlayBox,
    OverlayEmphasis,
    OverlayLink,
)
from trafficpulse.overlay.providers import register_defaults
from trafficpulse.overlay.providers.illegal_stopping import IllegalStoppingOverlayProvider
from trafficpulse.overlay.providers.triple_riding import TripleRidingOverlayProvider
from trafficpulse.overlay.providers.wrong_way import WrongWayOverlayProvider
from trafficpulse.overlay.registry import (
    OverlayCompositor,
    OverlayFrameRef,
    OverlayProviderRegistry,
)
from trafficpulse.pipeline.illegal_stopping import (
    IllegalStoppingOverlayCapture,
    IllegalStoppingTrackFrame,
)
from trafficpulse.pipeline.red_light import RedLightOverlayCapture, RedLightTrackFrame
from trafficpulse.pipeline.triple_riding import TripleRidingOverlayFrame
from trafficpulse.pipeline.wrong_way import WrongWayOverlayCapture, WrongWayTrackFrame

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ALL_VIOLATION_KINDS = frozenset(
    {"wrong_way", "illegal_stopping", "no_helmet", "triple_riding", "red_light_jumping"}
)
#: Registered kinds that describe no violation. ``helmet_analysis`` is perception
#: without enforcement, so it has a provider but must never be counted as a shipped
#: violation -- keeping the two sets apart is what makes the assertion below still say
#: "one provider per shipped rule" rather than merely "some providers exist".
_NON_VIOLATION_KINDS = frozenset({"helmet_analysis"})


def _frame_ref(index: int, seconds: float) -> OverlayFrameRef:
    return OverlayFrameRef(
        camera_id="cam-1", frame_index=index, media_seconds=seconds, width=320, height=240
    )


def _event(
    violation: ViolationType,
    *,
    track_ids: tuple[str, ...],
    start: float,
    trigger: float,
    thresholds: tuple[MeasuredValue, ...] = (),
) -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id=f"evt-{violation.value}-{trigger}",
        violation_type=violation,
        camera_id="cam-1",
        track_ids=track_ids,
        start_at=_EPOCH + timedelta(seconds=start),
        trigger_at=_EPOCH + timedelta(seconds=trigger),
        rule_id=violation.value,
        thresholds=thresholds,
        created_at=_EPOCH + timedelta(seconds=trigger),
    )


@pytest.fixture
def registry() -> OverlayProviderRegistry:
    fresh = OverlayProviderRegistry()
    register_defaults(fresh)
    return fresh


# --- registry ----------------------------------------------------------------------
def test_every_shipped_violation_has_a_registered_provider(
    registry: OverlayProviderRegistry,
) -> None:
    """The R6 gap, closed: five shipped rules, five providers."""

    assert registry.known_kinds() == _ALL_VIOLATION_KINDS | _NON_VIOLATION_KINDS
    # Every shipped violation is covered, and nothing extra masquerades as one.
    assert registry.known_kinds() - _NON_VIOLATION_KINDS == _ALL_VIOLATION_KINDS
    # `speeding` has no shipped reasoner, so it must have no provider either.
    assert "speeding" not in registry.known_kinds()


def test_a_capture_resolves_to_its_own_violations_provider(
    registry: OverlayProviderRegistry,
) -> None:
    """Dispatch is by capture type -- the driver never names a violation."""

    assert registry.kind_for(_wrong_way_capture()) == "wrong_way"
    assert registry.kind_for(_illegal_stopping_capture()) == "illegal_stopping"
    assert (
        registry.kind_for(
            RedLightOverlayCapture(stop_line=((0.0, 0.0), (1.0, 0.0)), zone_polygon=())
        )
        == "red_light_jumping"
    )


def test_dispatch_is_deterministic_across_registries() -> None:
    """Two independently built registries answer identically."""

    first, second = OverlayProviderRegistry(), OverlayProviderRegistry()
    register_defaults(first)
    register_defaults(second)
    capture = _wrong_way_capture()

    assert first.kind_for(capture) == second.kind_for(capture)
    assert first.known_kinds() == second.known_kinds()


def test_an_unregistered_source_resolves_to_nothing_rather_than_raising(
    registry: OverlayProviderRegistry,
) -> None:
    """A driver must be able to hand over anything a rule produced."""

    assert registry.kind_for(object()) is None
    assert registry.create_for(object(), ()) is None


def test_an_unknown_violation_kind_is_a_typed_refusal(
    registry: OverlayProviderRegistry,
) -> None:
    with pytest.raises(KeyError, match="speeding"):
        registry.create("speeding")


def test_a_duplicate_registration_is_refused(registry: OverlayProviderRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register("wrong_way", WrongWayOverlayProvider)


def test_one_capture_type_cannot_serve_two_violations(
    registry: OverlayProviderRegistry,
) -> None:
    """Ambiguous dispatch is refused at registration, not resolved by luck."""

    with pytest.raises(ValueError, match="already the overlay source"):
        registry.register(
            "speeding", WrongWayOverlayProvider, source_type=WrongWayOverlayCapture
        )


# --- wrong-way provider -------------------------------------------------------------
def _wrong_way_capture(*, contradiction: bool = True) -> WrongWayOverlayCapture:
    return WrongWayOverlayCapture(
        lane_id="zone-lane",
        legal_direction=(0.0, -1.0),
        frames=[
            WrongWayTrackFrame(
                frame_index=index,
                media_seconds=index * 0.1,
                track_id="t-1",
                bbox=(140.0, 60.0 + index, 180.0, 100.0 + index),
                heading_degrees=90.0,  # travelling DOWN the frame
                legal_heading_degrees=270.0,  # the lane says UP
                deviation_degrees=180.0,
                is_contradiction=contradiction,
            )
            for index in range(4)
        ],
    )


def test_wrong_way_draws_the_vehicle_and_both_headings() -> None:
    """Not a label: the measured heading and the lane's legal heading, as geometry."""

    events = (
        _event(
            ViolationType.WRONG_WAY,
            track_ids=("t-1",),
            start=0.0,
            trigger=0.2,
            thresholds=(MeasuredValue(name="min_persistence", value=1.0, unit="s"),),
        ),
    )
    provider = WrongWayOverlayProvider(_wrong_way_capture(), events)

    elements = provider.elements_for_frame(_frame_ref(3, 0.3))
    boxes = [e for e in elements if isinstance(e, OverlayBox)]
    links = [e for e in elements if isinstance(e, OverlayLink)]

    assert [b.bounds for b in boxes] == [(140.0, 63.0, 180.0, 103.0)]  # the tracked box
    assert boxes[0].key == "t-1"  # track identity
    assert boxes[0].alert is OverlayAlert.CONFIRMED
    assert boxes[0].caption is not None
    assert boxes[0].caption.metric == "180°"  # the measured deviation
    # Two arrows from the same origin: what it did, and what the lane requires.
    assert len(links) == 2
    origins = {(round(link.points[0].x, 3), round(link.points[0].y, 3)) for link in links}
    assert origins == {(160.0, 83.0)}
    legal, travelled = links
    assert legal.emphasis is OverlayEmphasis.CONTEXT
    assert travelled.emphasis is OverlayEmphasis.SUBJECT
    # Travelling down (+y) against a legal heading that points up (-y).
    assert travelled.points[1].y > travelled.points[0].y
    assert legal.points[1].y < legal.points[0].y


def test_wrong_way_draws_no_legal_arrow_when_the_scene_declared_none() -> None:
    """Missing data stays missing: no reference arrow is invented."""

    capture = WrongWayOverlayCapture(
        lane_id="zone-lane",
        legal_direction=(0.0, -1.0),
        frames=[
            WrongWayTrackFrame(
                frame_index=0,
                media_seconds=0.0,
                track_id="t-1",
                bbox=(10.0, 10.0, 50.0, 50.0),
                heading_degrees=90.0,
                legal_heading_degrees=None,
                deviation_degrees=42.0,
                is_contradiction=True,
            )
        ],
    )
    elements = WrongWayOverlayProvider(capture).elements_for_frame(_frame_ref(0, 0.0))

    assert len([e for e in elements if isinstance(e, OverlayLink)]) == 1


def test_wrong_way_escalates_only_after_the_trigger() -> None:
    """Before confirmation the box observes; the meter shows the run's progress."""

    events = (
        _event(
            ViolationType.WRONG_WAY,
            track_ids=("t-1",),
            start=0.0,
            trigger=1.0,
            thresholds=(MeasuredValue(name="min_persistence", value=1.0, unit="s"),),
        ),
    )
    provider = WrongWayOverlayProvider(_wrong_way_capture(), events)

    box = next(
        e for e in provider.elements_for_frame(_frame_ref(2, 0.2)) if isinstance(e, OverlayBox)
    )
    assert box.alert is OverlayAlert.OBSERVING
    assert box.caption is not None and box.caption.progress == pytest.approx(0.2)


# --- illegal-stopping provider -------------------------------------------------------
def _illegal_stopping_capture(
    *, inside: bool = True, stationary: bool = True
) -> IllegalStoppingOverlayCapture:
    return IllegalStoppingOverlayCapture(
        zone_polygons=(("zone-nostop", ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0))),),
        frames=[
            IllegalStoppingTrackFrame(
                frame_index=index,
                media_seconds=index * 0.1,
                track_id="t-1",
                bbox=(20.0, 20.0, 60.0, 60.0),
                zone_id="zone-nostop" if inside else None,
                is_inside=inside,
                is_stationary=stationary,
                dwell_seconds=index * 0.1,
            )
            for index in range(4)
        ],
    )


def test_illegal_stopping_draws_the_configured_zone_and_the_stopped_vehicle() -> None:
    """The prohibited area is the scene's own polygon, not a suggestive shape."""

    events = (
        _event(
            ViolationType.ILLEGAL_STOPPING,
            track_ids=("t-1",),
            start=0.0,
            trigger=0.2,
            thresholds=(MeasuredValue(name="stationary_duration", value=2.0, unit="s"),),
        ),
    )
    provider = IllegalStoppingOverlayProvider(_illegal_stopping_capture(), events)

    elements = provider.elements_for_frame(_frame_ref(3, 0.3))
    zone = next(e for e in elements if isinstance(e, OverlayLink))
    box = next(e for e in elements if isinstance(e, OverlayBox))

    # A closed ring of the declared polygon (first vertex repeated).
    assert [(p.x, p.y) for p in zone.points] == [
        (10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 10.0)
    ]
    assert zone.emphasis is OverlayEmphasis.CONTEXT
    assert box.bounds == (20.0, 20.0, 60.0, 60.0)
    assert box.key == "t-1"
    assert box.alert is OverlayAlert.CONFIRMED
    assert box.caption is not None
    assert box.caption.metric == "0.3s"  # the measured dwell
    assert "zone-nostop" in box.caption.lines[0]


def test_illegal_stopping_does_not_incriminate_a_legal_stop() -> None:
    """Stationary outside every no-stopping zone is not an offence, and reads so."""

    provider = IllegalStoppingOverlayProvider(
        _illegal_stopping_capture(inside=False, stationary=True)
    )

    box = next(
        e for e in provider.elements_for_frame(_frame_ref(1, 0.1)) if isinstance(e, OverlayBox)
    )
    assert box.alert is OverlayAlert.NONE
    assert box.caption is not None
    assert "outside zone" in box.caption.lines[0]


def test_illegal_stopping_omits_the_dwell_metric_when_none_was_measured() -> None:
    """No substituted zero: an unmeasured dwell is simply absent."""

    capture = IllegalStoppingOverlayCapture(
        zone_polygons=(),
        frames=[
            IllegalStoppingTrackFrame(
                frame_index=0,
                media_seconds=0.0,
                track_id="t-1",
                bbox=(20.0, 20.0, 60.0, 60.0),
                zone_id="zone-nostop",
                is_inside=True,
                is_stationary=True,
                dwell_seconds=None,
            )
        ],
    )
    box = next(
        e
        for e in IllegalStoppingOverlayProvider(capture).elements_for_frame(_frame_ref(0, 0.0))
        if isinstance(e, OverlayBox)
    )
    assert box.caption is not None and box.caption.metric is None


# --- triple-riding provider ----------------------------------------------------------
def _triple_riding_frames(*, riders: int = 3) -> tuple[TripleRidingOverlayFrame, ...]:
    return tuple(
        TripleRidingOverlayFrame(
            frame_index=index,
            media_seconds=index * 0.1,
            motorcycle_track_id="m-1",
            motorcycle_bbox=(100.0, 100.0, 200.0, 150.0),
            rider_count=riders,
            rider_bboxes=tuple(
                (100.0 + slot * 30.0, 60.0, 130.0 + slot * 30.0, 140.0) for slot in range(riders)
            ),
        )
        for index in range(4)
    )


def test_triple_riding_draws_the_motorcycle_its_count_and_every_rider() -> None:
    """The count is checkable: each associated rider is drawn and linked."""

    events = (
        _event(
            ViolationType.TRIPLE_RIDING,
            track_ids=("m-1",),
            start=0.0,
            trigger=0.2,
            thresholds=(
                MeasuredValue(name="min_persistence", value=1.0, unit="s"),
                MeasuredValue(name="rider_count_threshold", value=3.0, unit="count"),
            ),
        ),
    )
    provider = TripleRidingOverlayProvider(_triple_riding_frames(), events)

    elements = provider.elements_for_frame(_frame_ref(3, 0.3))
    boxes = [e for e in elements if isinstance(e, OverlayBox)]
    links = [e for e in elements if isinstance(e, OverlayLink)]

    motorcycle = next(b for b in boxes if b.key == "m-1")
    assert motorcycle.bounds == (100.0, 100.0, 200.0, 150.0)
    assert motorcycle.alert is OverlayAlert.CONFIRMED
    assert motorcycle.caption is not None and motorcycle.caption.metric == "3"
    # One box + one association link per rider the associator actually attached.
    riders = [b for b in boxes if b.emphasis is OverlayEmphasis.OBJECT]
    assert len(riders) == 3
    assert len(links) == 3
    assert all(link.points[1] == links[0].points[1] for link in links)  # all to the bike


def test_triple_riding_draws_no_riders_when_none_were_associated() -> None:
    """A count without rider boxes is drawn as exactly that -- no invented seats."""

    frames = (
        TripleRidingOverlayFrame(
            frame_index=0,
            media_seconds=0.0,
            motorcycle_track_id="m-1",
            motorcycle_bbox=(100.0, 100.0, 200.0, 150.0),
            rider_count=3,
            rider_bboxes=(),
        ),
    )
    elements = TripleRidingOverlayProvider(frames).elements_for_frame(_frame_ref(0, 0.0))

    assert len([e for e in elements if isinstance(e, OverlayBox)]) == 1
    assert not [e for e in elements if isinstance(e, OverlayLink)]


def test_triple_riding_below_the_threshold_is_not_alerted() -> None:
    provider = TripleRidingOverlayProvider(_triple_riding_frames(riders=2))

    box = next(
        e for e in provider.elements_for_frame(_frame_ref(0, 0.0)) if isinstance(e, OverlayBox)
    )
    assert box.alert is OverlayAlert.NONE
    assert box.caption is not None and box.caption.metric == "2"


# --- has_content -------------------------------------------------------------------
def test_a_capture_that_recorded_nothing_reports_no_content() -> None:
    """What makes "the rule ran but there is no annotated video" still reachable."""

    empty = WrongWayOverlayCapture(lane_id="zone-lane", legal_direction=(0.0, -1.0))
    assert WrongWayOverlayProvider(empty).has_content() is False
    assert TripleRidingOverlayProvider(()).has_content() is False


def test_scene_geometry_alone_is_content_worth_drawing() -> None:
    """A calibrated zone nobody entered is still what an analyst needs to see."""

    capture = IllegalStoppingOverlayCapture(
        zone_polygons=(("zone-nostop", ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0))),)
    )
    assert IllegalStoppingOverlayProvider(capture).has_content() is True


# --- multi-violation composition ------------------------------------------------------
def _providers_for(*kinds: str) -> list[object]:
    built: dict[str, object] = {
        "wrong_way": WrongWayOverlayProvider(
            _wrong_way_capture(),
            (
                _event(
                    ViolationType.WRONG_WAY, track_ids=("t-1",), start=0.0, trigger=0.1
                ),
            ),
        ),
        "illegal_stopping": IllegalStoppingOverlayProvider(
            _illegal_stopping_capture(),
            (
                _event(
                    ViolationType.ILLEGAL_STOPPING,
                    track_ids=("t-1",),
                    start=0.0,
                    trigger=0.1,
                ),
            ),
        ),
        "triple_riding": TripleRidingOverlayProvider(
            _triple_riding_frames(),
            (
                _event(
                    ViolationType.TRIPLE_RIDING,
                    track_ids=("m-1",),
                    start=0.0,
                    trigger=0.1,
                    thresholds=(
                        MeasuredValue(name="rider_count_threshold", value=3.0, unit="count"),
                    ),
                ),
            ),
        ),
        "red_light_jumping": _red_light_provider(),
    }
    return [built[kind] for kind in kinds]


def _red_light_provider() -> object:
    from trafficpulse.overlay.providers.red_light import RedLightOverlayProvider

    capture = RedLightOverlayCapture(
        stop_line=((100.0, 120.0), (220.0, 120.0)),
        zone_polygon=((100.0, 150.0), (220.0, 150.0), (220.0, 235.0)),
        frames=[
            RedLightTrackFrame(
                frame_index=index,
                media_seconds=index * 0.1,
                track_id="t-1",
                bbox=(140.0, 150.0, 180.0, 190.0),
                is_inside=True,
                entered_on_red=True,
                entry_state=SignalState.RED,
            )
            for index in range(4)
        ],
    )
    return RedLightOverlayProvider(
        capture,
        (_event(ViolationType.RED_LIGHT_JUMPING, track_ids=("t-1",), start=0.0, trigger=0.1),),
    )


@pytest.mark.parametrize(
    "kinds",
    [
        ("wrong_way", "illegal_stopping"),
        ("wrong_way", "triple_riding"),
        ("illegal_stopping", "triple_riding"),
        ("wrong_way", "illegal_stopping", "triple_riding"),
        ("wrong_way", "illegal_stopping", "triple_riding", "red_light_jumping"),
    ],
)
def test_every_provider_contributes_to_one_scene(kinds: tuple[str, ...]) -> None:
    """No provider suppresses, replaces, or short-circuits another."""

    providers = _providers_for(*kinds)
    scene = OverlayCompositor(providers).scene_for(_frame_ref(3, 0.3))

    # Each provider's own contribution survives fusion, in full.
    expected = sum(
        len(provider.elements_for_frame(_frame_ref(3, 0.3)))  # type: ignore[attr-defined]
        for provider in providers
    )
    assert len(scene.elements) == expected
    # One confirmed banner per violation: none was dropped for another's.
    banners = [e for e in scene.elements if isinstance(e, OverlayBanner)]
    assert len(banners) == len(kinds)
    assert len({b.title for b in banners}) == len(kinds)


def test_overlapping_events_on_one_track_both_render() -> None:
    """Two violations, one track, disjoint windows -- both representable.

    Wrong-way triggers early and illegal-stopping later on the same ``t-1``; at a
    frame after both triggers each provider still reports its own confirmation.
    """

    wrong_way = WrongWayOverlayProvider(
        _wrong_way_capture(),
        (_event(ViolationType.WRONG_WAY, track_ids=("t-1",), start=0.0, trigger=0.1),),
    )
    stopping = IllegalStoppingOverlayProvider(
        _illegal_stopping_capture(),
        (
            _event(
                ViolationType.ILLEGAL_STOPPING, track_ids=("t-1",), start=0.1, trigger=0.2
            ),
        ),
    )
    scene = OverlayCompositor([wrong_way, stopping]).scene_for(_frame_ref(3, 0.3))

    banners = [e for e in scene.elements if isinstance(e, OverlayBanner)]
    assert {b.title for b in banners} == {"WRONG WAY", "ILLEGAL STOPPING"}
    # Both violations drew their own box for the same track: neither overwrote the
    # other, and each carries its own violation's caption.
    boxes = [e for e in scene.elements if isinstance(e, OverlayBox) and e.key == "t-1"]
    assert len(boxes) == 2
    assert all(b.alert is OverlayAlert.CONFIRMED for b in boxes)


def test_a_provider_whose_violation_did_not_confirm_still_draws_its_context() -> None:
    """An unconfirmed rule contributes observation state, never another's alert."""

    stopping = IllegalStoppingOverlayProvider(_illegal_stopping_capture(), events=())
    wrong_way = WrongWayOverlayProvider(
        _wrong_way_capture(),
        (_event(ViolationType.WRONG_WAY, track_ids=("t-1",), start=0.0, trigger=0.1),),
    )
    scene = OverlayCompositor([stopping, wrong_way]).scene_for(_frame_ref(3, 0.3))

    banners = [e for e in scene.elements if isinstance(e, OverlayBanner)]
    assert [b.title for b in banners] == ["WRONG WAY"]
    # The un-confirmed rule still drew its zone and its box, just not an alert.
    assert any(isinstance(e, OverlayLink) for e in scene.elements)

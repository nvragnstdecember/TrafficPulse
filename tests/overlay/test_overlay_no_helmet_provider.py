"""The no-helmet overlay provider: inference metadata -> generic elements."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import ConfidenceBreakdown, ConfirmedEvent, MeasuredValue
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.overlay import (
    OverlayAlert,
    OverlayEmphasis,
    OverlayFrameRef,
    OverlayProviderRegistry,
)
from trafficpulse.overlay.providers.no_helmet import (
    NoHelmetOverlayProvider,
    register_no_helmet_overlay,
)
from trafficpulse.pipeline.helmet_observer import HelmetOverlayFrame, HelmetOverlayRider

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _rider(
    label: str = "no_helmet", confidence: float | None = 0.97, gated: bool = False
) -> HelmetOverlayRider:
    return HelmetOverlayRider(
        rider_track_id="iou-1",
        rider_bbox=(100, 80, 300, 460),
        motorcycle_track_id="iou-4",
        motorcycle_bbox=(120, 180, 360, 520),
        head_bbox=(100, 80, 300, 194),
        helmet_label=label,
        confidence=confidence,
        gated=gated,
    )


def _frame(idx: int, t: float, rider: HelmetOverlayRider) -> HelmetOverlayFrame:
    return HelmetOverlayFrame(frame_index=idx, media_seconds=t, riders=(rider,))


def _ref(idx: int, t: float) -> OverlayFrameRef:
    return OverlayFrameRef(camera_id="cam", frame_index=idx, media_seconds=t, width=640, height=560)


def _event(
    *,
    start_seconds: float = 0.0,
    thresholds: tuple[MeasuredValue, ...] = (
        MeasuredValue(name="min_persistence", value=1.0, unit="seconds"),
    ),
    confidence: ConfidenceBreakdown | None = None,
) -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id="evt-abc123",
        violation_type=ViolationType.NO_HELMET,
        camera_id="cam",
        track_ids=("iou-1", "iou-4"),
        start_at=EPOCH + timedelta(seconds=start_seconds),
        trigger_at=EPOCH + timedelta(seconds=start_seconds + 1.0),
        rule_id="no_helmet",
        rule_version="0.1.0",
        source_hypothesis_id="hyp-1",
        created_at=EPOCH + timedelta(seconds=1.0),
        thresholds=thresholds,
        confidence=(
            confidence
            if confidence is not None
            else ConfidenceBreakdown(classifier=0.97, temporal_consistency=0.8)
        ),
    )


def _rider_box(elements: list[object]) -> object:
    return next(
        e
        for e in elements
        if e.kind == "box" and e.emphasis is OverlayEmphasis.SUBJECT  # type: ignore[attr-defined]
    )


def test_no_helmet_without_confirmation_is_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())])
    els = list(prov.elements_for_frame(_ref(5, 0.3)))
    boxes = {e.emphasis: e for e in els if e.kind == "box"}
    assert set(boxes) == {OverlayEmphasis.OBJECT, OverlayEmphasis.SUBJECT, OverlayEmphasis.REGION}
    assert all(b.alert is OverlayAlert.OBSERVING for b in boxes.values())
    assert boxes[OverlayEmphasis.SUBJECT].caption.lines[0] == "Observing"
    assert boxes[OverlayEmphasis.REGION].caption.lines[0] == "No Helmet"
    assert boxes[OverlayEmphasis.REGION].caption.metric == "97%"
    assert not [e for e in els if e.kind == "banner"]


# --- progressive disclosure ---------------------------------------------------
def test_the_motorcycle_box_is_never_captioned() -> None:
    # Blue already means "vehicle" and the chain says whose it is; a chip per bike
    # per frame is pure clutter. Its identity survives on the confirmation banner.
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())], [_event()])
    for media in (0.3, 1.33):
        boxes = {
            e.emphasis: e for e in prov.elements_for_frame(_ref(5, media)) if e.kind == "box"
        }
        assert boxes[OverlayEmphasis.OBJECT].caption is None


def test_an_unremarkable_rider_carries_no_state_caption() -> None:
    # Nothing is being argued about a helmeted rider, so the subject stays silent;
    # the head readout still reports what the classifier saw.
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider(label="helmet", confidence=0.93))])
    boxes = {e.emphasis: e for e in prov.elements_for_frame(_ref(5, 0.3)) if e.kind == "box"}
    assert boxes[OverlayEmphasis.SUBJECT].caption is None
    assert boxes[OverlayEmphasis.REGION].caption.lines == ("Helmet",)
    assert boxes[OverlayEmphasis.REGION].caption.metric == "93%"


def test_confirmation_states_the_reading_once_not_three_times() -> None:
    # The confidence belongs on the subject and the banner; repeating it on the head
    # box as well is what made the confirmed frame unreadable.
    prov = NoHelmetOverlayProvider([_frame(40, 1.33, _rider())], [_event()])
    els = list(prov.elements_for_frame(_ref(40, 1.33)))
    boxes = {e.emphasis: e for e in els if e.kind == "box"}
    assert boxes[OverlayEmphasis.SUBJECT].caption.lines == ("NO HELMET", "Track iou-1")
    assert boxes[OverlayEmphasis.SUBJECT].caption.metric == "97%"
    assert boxes[OverlayEmphasis.REGION].caption is None
    assert boxes[OverlayEmphasis.OBJECT].caption is None


def test_association_chain_links_head_rider_motorcycle() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())])
    links = [e for e in prov.elements_for_frame(_ref(5, 0.3)) if e.kind == "link"]
    assert len(links) == 1
    assert len(links[0].points) == 3  # head -> rider -> motorcycle


def test_confirmed_after_trigger_turns_red_and_banners() -> None:
    prov = NoHelmetOverlayProvider([_frame(40, 1.33, _rider())], [_event()])
    els = list(prov.elements_for_frame(_ref(40, 1.33)))
    assert all(e.alert is OverlayAlert.CONFIRMED for e in els if e.kind == "box")
    banners = [e for e in els if e.kind == "banner"]
    assert len(banners) == 1
    assert banners[0].title == "NO HELMET"
    assert any("iou-1" in line for line in banners[0].lines)
    assert any("evt-abc123" in detail for detail in banners[0].details)


def test_confirmed_banner_ranks_its_information() -> None:
    banner = next(
        e
        for e in NoHelmetOverlayProvider(
            [_frame(40, 1.33, _rider())], [_event()]
        ).elements_for_frame(_ref(40, 1.33))
        if e.kind == "banner"
    )
    # headline: what, and the one figure a reviewer reads first
    assert banner.title == "NO HELMET"
    assert banner.metric == "97%"
    # body: who, and for how long against the bar the rule actually required
    body = "\n".join(banner.lines)
    assert "Track iou-1 · sustained 1.00s of 1.00s required" in body
    assert "Temporal consistency 80%" in body
    # details: audit trail, present but visually subordinate
    assert banner.details == ("Bike iou-4 · t 00:01.000", "evt-abc123")


def test_confirmed_banner_omits_components_the_reasoner_never_measured() -> None:
    # An unmeasured component is absent, never rendered as a fabricated 0%.
    event = _event(confidence=ConfidenceBreakdown(classifier=0.5))
    prov = NoHelmetOverlayProvider([_frame(40, 1.33, _rider())], [event])
    banner = next(e for e in prov.elements_for_frame(_ref(40, 1.33)) if e.kind == "banner")
    assert banner.metric == "50%"
    assert "Temporal consistency" not in "\n".join(banner.lines)


def test_confirmed_banner_has_no_metric_when_no_score_was_measured() -> None:
    event = _event(confidence=ConfidenceBreakdown())
    prov = NoHelmetOverlayProvider([_frame(40, 1.33, _rider())], [event])
    banner = next(e for e in prov.elements_for_frame(_ref(40, 1.33)) if e.kind == "banner")
    assert banner.metric is None


def test_before_trigger_is_still_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(10, 0.5, _rider())], [_event()])
    els = list(prov.elements_for_frame(_ref(10, 0.5)))  # 0.5s < 1.0s trigger
    assert all(e.alert is OverlayAlert.OBSERVING for e in els if e.kind == "box")
    assert not [e for e in els if e.kind == "banner"]


# --- temporal reasoning progression -------------------------------------------
def test_evidence_meter_tracks_the_episode_window() -> None:
    # Inside the confirmed episode's [start_at, trigger_at) window the rider shows
    # elapsed-vs-threshold progress read straight off the event.
    prov = NoHelmetOverlayProvider([_frame(10, 0.6, _rider())], [_event()])
    rider = _rider_box(list(prov.elements_for_frame(_ref(10, 0.6))))
    assert rider.caption.progress == pytest.approx(0.6)
    assert "0.60s / 1.00s" in rider.caption.lines
    assert rider.caption.lines[0] == "Observing"


def test_evidence_meter_is_full_once_confirmed() -> None:
    prov = NoHelmetOverlayProvider([_frame(40, 1.33, _rider())], [_event()])
    rider = _rider_box(list(prov.elements_for_frame(_ref(40, 1.33))))
    assert rider.caption.progress == 1.0
    assert rider.caption.lines[0] == "NO HELMET"


def test_meter_never_exceeds_one_between_threshold_and_trigger() -> None:
    # The trigger fires on the first supporting observation past the threshold, so
    # elapsed can exceed it by a frame; the bar saturates rather than overflowing.
    event = _event(
        thresholds=(MeasuredValue(name="min_persistence", value=0.5, unit="seconds"),)
    )
    prov = NoHelmetOverlayProvider([_frame(20, 0.9, _rider())], [event])
    rider = _rider_box(list(prov.elements_for_frame(_ref(20, 0.9))))
    assert rider.caption.progress == 1.0


def test_no_meter_for_a_rider_the_reasoner_never_confirmed() -> None:
    # No episode exists, so there is no denominator; inventing one would assert
    # reasoning the system never performed.
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())])
    rider = _rider_box(list(prov.elements_for_frame(_ref(5, 0.3))))
    assert rider.caption.progress is None
    assert rider.caption.lines == ("Observing", "Track iou-1")


def test_no_meter_before_the_episode_started() -> None:
    # Frames earlier than the episode's start_at are outside the window it measures:
    # the rider is bare-headed, but this is not yet the run that got confirmed.
    event = _event(start_seconds=2.0)  # window is [2.0s, 3.0s)
    prov = NoHelmetOverlayProvider([_frame(3, 0.5, _rider())], [event])
    rider = _rider_box(list(prov.elements_for_frame(_ref(3, 0.5))))
    assert rider.alert is OverlayAlert.OBSERVING
    assert rider.caption.progress is None

    inside = NoHelmetOverlayProvider([_frame(60, 2.4, _rider())], [event])
    rider_inside = _rider_box(list(inside.elements_for_frame(_ref(60, 2.4))))
    assert rider_inside.caption.progress == pytest.approx(0.4)


def test_no_meter_when_the_event_declares_no_persistence_threshold() -> None:
    prov = NoHelmetOverlayProvider([_frame(10, 0.6, _rider())], [_event(thresholds=())])
    rider = _rider_box(list(prov.elements_for_frame(_ref(10, 0.6))))
    assert rider.caption.progress is None


def test_helmet_rider_is_neutral_not_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider(label="helmet", confidence=0.95))])
    boxes = [e for e in prov.elements_for_frame(_ref(5, 0.3)) if e.kind == "box"]
    assert all(b.alert is OverlayAlert.NONE for b in boxes)


def test_gated_crop_is_boxed_but_not_captioned() -> None:
    # The classifier abstained, so there is no figure to print. An "Uncertain" chip
    # with no number costs a caption and says nothing the yellow box does not.
    gated = _rider(label="uncertain", confidence=None, gated=True)
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, gated)])
    region = next(e for e in prov.elements_for_frame(_ref(5, 0.3))
                  if e.kind == "box" and e.emphasis is OverlayEmphasis.REGION)
    assert region.caption is None  # the region is still drawn -- only its label is spent


# --- presentation-only smoothing ----------------------------------------------
def _jittering(frames: int, amplitude: float) -> list[HelmetOverlayFrame]:
    """A stationary rider whose box shakes by +/- ``amplitude`` every frame."""

    out = []
    for i in range(frames):
        wobble = amplitude if i % 2 else -amplitude
        rider = HelmetOverlayRider(
            rider_track_id="iou-1",
            rider_bbox=(100 + wobble, 80, 300 + wobble, 460),
            motorcycle_track_id="iou-4",
            motorcycle_bbox=(120, 180, 360, 520),
            head_bbox=(100 + wobble, 80, 300 + wobble, 194),
            helmet_label="no_helmet",
            confidence=0.90 + (0.004 if i % 2 else -0.004),
            gated=False,
        )
        out.append(HelmetOverlayFrame(frame_index=i, media_seconds=i / 30.0, riders=(rider,)))
    return out


def test_drawn_geometry_settles_even_though_the_capture_shakes() -> None:
    capture = _jittering(20, amplitude=1.0)
    prov = NoHelmetOverlayProvider(capture)
    drawn = [
        _rider_box(list(prov.elements_for_frame(_ref(i, i / 30.0)))).bounds
        for i in range(10, 20)
    ]
    assert len(set(drawn)) == 1  # perfectly still on screen...
    # ...while the underlying capture really did move every single frame
    assert len({f.riders[0].rider_bbox for f in capture[10:20]}) == 2


def test_the_displayed_score_settles_too() -> None:
    prov = NoHelmetOverlayProvider(_jittering(20, amplitude=1.0))
    metrics = {
        next(
            e
            for e in prov.elements_for_frame(_ref(i, i / 30.0))
            if e.kind == "box" and e.emphasis is OverlayEmphasis.REGION
        ).caption.metric
        for i in range(10, 20)
    }
    assert len(metrics) == 1


def test_smoothing_never_mutates_the_captured_metadata() -> None:
    # The capture is the inference record; the overlay may only read it.
    capture = _jittering(12, amplitude=3.0)
    before = [(f.frame_index, f.riders[0].rider_bbox, f.riders[0].confidence) for f in capture]
    NoHelmetOverlayProvider(capture)
    after = [(f.frame_index, f.riders[0].rider_bbox, f.riders[0].confidence) for f in capture]
    assert before == after


def test_smoothing_follows_a_genuinely_moving_subject() -> None:
    # Calm must not become stuck: a rider crossing the frame ends up where the
    # capture says they are, within a frame or two of lag.
    moving = [
        HelmetOverlayFrame(
            frame_index=i,
            media_seconds=i / 30.0,
            riders=(
                HelmetOverlayRider(
                    rider_track_id="iou-1",
                    rider_bbox=(10.0 * i, 80, 10.0 * i + 200, 460),
                    motorcycle_track_id="iou-4",
                    motorcycle_bbox=(10.0 * i, 180, 10.0 * i + 240, 520),
                    head_bbox=None,
                    helmet_label="helmet",
                    confidence=0.9,
                    gated=False,
                ),
            ),
        )
        for i in range(40)
    ]
    prov = NoHelmetOverlayProvider(moving)
    drawn = _rider_box(list(prov.elements_for_frame(_ref(39, 1.3)))).bounds
    assert drawn[0] == pytest.approx(390.0, abs=15.0)


def test_unknown_frame_yields_nothing() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())])
    assert prov.elements_for_frame(_ref(999, 30.0)) == ()


def test_registers_under_its_violation_kind() -> None:
    reg = OverlayProviderRegistry()
    register_no_helmet_overlay(reg)
    assert reg.known_kinds() == frozenset({"no_helmet"})
    built = reg.create("no_helmet", [_frame(5, 0.3, _rider())], [])
    assert built.violation_kind == "no_helmet"
    with pytest.raises(ValueError):
        register_no_helmet_overlay(reg)  # duplicate registration is rejected

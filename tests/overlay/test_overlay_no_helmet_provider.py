"""The no-helmet overlay provider: inference metadata -> generic elements."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import ConfirmedEvent
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


def _rider(label: str = "no_helmet", confidence: float | None = 0.97, gated: bool = False) -> HelmetOverlayRider:
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


def _event() -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id="evt-abc123",
        violation_type=ViolationType.NO_HELMET,
        camera_id="cam",
        track_ids=("iou-1", "iou-4"),
        start_at=EPOCH,
        trigger_at=EPOCH + timedelta(seconds=1.0),
        rule_id="no_helmet",
        rule_version="0.1.0",
        source_hypothesis_id="hyp-1",
        created_at=EPOCH + timedelta(seconds=1.0),
    )


def test_no_helmet_without_confirmation_is_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider())])
    els = list(prov.elements_for_frame(_ref(5, 0.3)))
    boxes = {e.emphasis: e for e in els if e.kind == "box"}
    assert set(boxes) == {OverlayEmphasis.OBJECT, OverlayEmphasis.SUBJECT, OverlayEmphasis.REGION}
    assert all(b.alert is OverlayAlert.OBSERVING for b in boxes.values())
    # motorcycle & rider captions, head shows label + confidence, no banner yet
    assert boxes[OverlayEmphasis.OBJECT].caption.lines[0] == "Motorcycle"
    assert boxes[OverlayEmphasis.SUBJECT].caption.lines[0] == "Rider"
    assert "Collecting evidence…" in boxes[OverlayEmphasis.SUBJECT].caption.lines
    assert boxes[OverlayEmphasis.REGION].caption.lines[0] == "No Helmet"
    assert boxes[OverlayEmphasis.REGION].caption.metric == "97%"
    assert not [e for e in els if e.kind == "banner"]


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
    assert any("evt-abc123" in line for line in banners[0].lines)


def test_before_trigger_is_still_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(10, 0.5, _rider())], [_event()])
    els = list(prov.elements_for_frame(_ref(10, 0.5)))  # 0.5s < 1.0s trigger
    assert all(e.alert is OverlayAlert.OBSERVING for e in els if e.kind == "box")
    assert not [e for e in els if e.kind == "banner"]


def test_helmet_rider_is_neutral_not_observing() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider(label="helmet", confidence=0.95))])
    boxes = [e for e in prov.elements_for_frame(_ref(5, 0.3)) if e.kind == "box"]
    assert all(b.alert is OverlayAlert.NONE for b in boxes)


def test_gated_crop_has_no_confidence_metric() -> None:
    prov = NoHelmetOverlayProvider([_frame(5, 0.3, _rider(label="uncertain", confidence=None, gated=True))])
    region = next(e for e in prov.elements_for_frame(_ref(5, 0.3))
                  if e.kind == "box" and e.emphasis is OverlayEmphasis.REGION)
    assert region.caption.lines[0] == "Uncertain"
    assert region.caption.metric is None


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

"""The generic overlay metadata model: construction, merging, serialisation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trafficpulse.overlay import (
    OverlayAlert,
    OverlayBanner,
    OverlayBox,
    OverlayCaption,
    OverlayEmphasis,
    OverlayLayer,
    OverlayLink,
    OverlayPoint,
    OverlayScene,
)


def _box(**kw: object) -> OverlayBox:
    base = dict(bounds=(0.0, 0.0, 10.0, 10.0), emphasis=OverlayEmphasis.SUBJECT)
    base.update(kw)
    return OverlayBox(**base)  # type: ignore[arg-type]


def test_scene_holds_a_typed_element_union() -> None:
    scene = OverlayScene(
        width=100,
        height=80,
        elements=(
            _box(layer=OverlayLayer.SUBJECT),
            OverlayLink(points=(OverlayPoint(x=0, y=0), OverlayPoint(x=5, y=5))),
            OverlayBanner(title="NO HELMET"),
        ),
    )
    assert [e.kind for e in scene.elements] == ["box", "link", "banner"]


def test_link_requires_at_least_two_points() -> None:
    with pytest.raises(ValidationError):
        OverlayLink(points=(OverlayPoint(x=0, y=0),))


def test_models_are_strict_and_frozen() -> None:
    with pytest.raises(ValidationError):
        _box(unexpected=1)  # extra='forbid'
    box = _box()
    with pytest.raises(ValidationError):
        box.emphasis = OverlayEmphasis.OBJECT  # type: ignore[misc]  # frozen


def test_merged_concatenates_elements_and_keeps_identity() -> None:
    a = OverlayScene(width=100, height=80, elements=(_box(),), frame_index=7, media_seconds=1.5)
    b = OverlayScene(width=100, height=80, elements=(OverlayBanner(title="X"),))
    merged = a.merged(b)
    assert [e.kind for e in merged.elements] == ["box", "banner"]
    assert merged.frame_index == 7 and merged.media_seconds == 1.5
    # inputs are untouched (frozen, pure)
    assert len(a.elements) == 1 and len(b.elements) == 1


def test_scene_json_roundtrip_is_lossless() -> None:
    scene = OverlayScene(
        width=64,
        height=48,
        elements=(
            _box(alert=OverlayAlert.CONFIRMED, caption=OverlayCaption(lines=("Rider",), metric="98%")),
        ),
    )
    restored = OverlayScene.model_validate_json(scene.model_dump_json())
    assert restored == scene


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        OverlayScene(width=0, height=10)

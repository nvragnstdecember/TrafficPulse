"""The Pillow renderer backend (skipped where Pillow is absent)."""

from __future__ import annotations

import numpy as np
import pytest

from trafficpulse.overlay import (
    OverlayAlert,
    OverlayBanner,
    OverlayBox,
    OverlayCaption,
    OverlayEmphasis,
    OverlayLayer,
    OverlayScene,
)
from trafficpulse.overlay.renderer import FrameSizeMismatchError

pytest.importorskip("PIL", reason="overlay renderer needs Pillow (the optional 'overlay' extra)")
from trafficpulse.overlay import PillowOverlayRenderer  # noqa: E402


def _scene(alert: OverlayAlert = OverlayAlert.NONE) -> OverlayScene:
    return OverlayScene(
        width=200,
        height=160,
        elements=(
            OverlayBox(
                bounds=(20, 20, 180, 140),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=alert,
                layer=OverlayLayer.SUBJECT,
                caption=OverlayCaption(lines=("Rider", "Track: iou-1"), metric="97%"),
            ),
        ),
    )


def test_render_returns_same_shape_and_draws_pixels() -> None:
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    out = PillowOverlayRenderer().render(img, _scene())
    assert out.shape == img.shape and out.dtype == np.uint8
    assert int((out.sum(axis=2) > 0).sum()) > 0  # something was drawn
    assert not np.array_equal(out, img)


def test_frame_size_mismatch_is_rejected() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(FrameSizeMismatchError):
        PillowOverlayRenderer().render(img, _scene())


def test_confirmed_scene_paints_red() -> None:
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    out = PillowOverlayRenderer().render(img, _scene(OverlayAlert.CONFIRMED))
    # a strongly red pixel (R high, G/B low) exists on the confirmed stroke
    r, g, b = out[..., 0].astype(int), out[..., 1].astype(int), out[..., 2].astype(int)
    reddish = (r > 180) & (g < 120) & (b < 120)
    assert reddish.any()


def _painted_count(scene: OverlayScene, width: int = 200, height: int = 160) -> int:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return int((PillowOverlayRenderer().render(img, scene).sum(axis=2) > 0).sum())


def _caption_pixels(box: OverlayBox, width: int = 200, height: int = 160) -> int:
    """Pixels a box's caption contributes, over the same box drawn bare.

    Font-independent by construction: it measures *whether the caption was drawn*
    rather than where the layout solver happened to put it. Asserting a chip landed
    in some pixel band encodes the metrics of whichever font the machine has, which
    differs between a developer box and CI.
    """

    with_caption = _painted_count(
        OverlayScene(width=width, height=height, elements=(box,)), width, height
    )
    bare = _painted_count(
        OverlayScene(
            width=width, height=height, elements=(box.model_copy(update={"caption": None}),)
        ),
        width,
        height,
    )
    return with_caption - bare


def _meter_scene(progress: float | None) -> OverlayScene:
    return OverlayScene(
        width=200,
        height=160,
        elements=(
            OverlayBox(
                bounds=(20, 20, 180, 140),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=OverlayAlert.OBSERVING,
                layer=OverlayLayer.SUBJECT,
                caption=OverlayCaption(lines=("Rider",), progress=progress),
            ),
        ),
    )


def _painted_intensity(scene: OverlayScene) -> int:
    """Total drawn brightness -- the filled bar is opaque, its track translucent."""

    img = np.zeros((160, 200, 3), dtype=np.uint8)
    return int(PillowOverlayRenderer().render(img, scene).astype(np.int64).sum())


def test_caption_meter_grows_with_its_fraction() -> None:
    # The renderer treats progress as a bare fraction: more progress, more filled
    # bar. It never asks what is progressing.
    empty, half, full = (_painted_intensity(_meter_scene(p)) for p in (0.0, 0.5, 1.0))
    assert empty < half < full


def test_a_zero_meter_still_draws_its_track() -> None:
    # "Nothing accumulated yet" must be visibly distinct from "no meter at all".
    assert _painted_intensity(_meter_scene(0.0)) > _painted_intensity(_meter_scene(None))


def test_a_banner_does_not_bury_the_caption_of_the_track_it_names() -> None:
    # Banners are pinned top-left and painted over captions; a confirmed rider near
    # that corner would otherwise lose its own label. The caption text must survive.
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    box = OverlayBox(
        bounds=(10, 10, 90, 120),
        emphasis=OverlayEmphasis.SUBJECT,
        alert=OverlayAlert.CONFIRMED,
        layer=OverlayLayer.SUBJECT,
        caption=OverlayCaption(lines=("Rider", "Track: iou-7")),
    )
    banner = OverlayBanner(title="NO HELMET", lines=("Track: iou-7",), icon="⚠")
    with_banner = PillowOverlayRenderer().render(
        img, OverlayScene(width=200, height=160, elements=(box, banner))
    )
    banner_only = PillowOverlayRenderer().render(
        img, OverlayScene(width=200, height=160, elements=(banner,))
    )
    # The caption contributes its own drawn area on top of the banner-only frame;
    # were it placed underneath, the banner would cover it and the delta would be
    # driven purely by the box outline.
    extra = int(((with_banner.sum(axis=2) > 0) & ~(banner_only.sum(axis=2) > 0)).sum())
    assert extra > 400


def test_a_box_too_small_to_caption_is_still_drawn() -> None:
    # A chip beside a distant object is bigger than the object; the label is
    # dropped, but the detection itself is never hidden.
    tiny = OverlayBox(
        bounds=(80, 70, 92, 82),  # 12px -- below the caption threshold
        emphasis=OverlayEmphasis.SUBJECT,
        alert=OverlayAlert.NONE,
        layer=OverlayLayer.SUBJECT,
        caption=OverlayCaption(lines=("Rider", "Track: iou-1"), metric="97%"),
    )
    assert _painted_count(OverlayScene(width=200, height=160, elements=(tiny,))) > 0
    # The geometry is drawn; the caption contributes nothing, because it is withheld.
    assert _caption_pixels(tiny) == 0


def test_a_confirmed_box_keeps_its_caption_however_small() -> None:
    # Two distant confirmations otherwise become two identical red boxes with no
    # way to tell which banner belongs to which.
    tiny_confirmed = OverlayBox(
        bounds=(80, 70, 92, 82),
        emphasis=OverlayEmphasis.SUBJECT,
        alert=OverlayAlert.CONFIRMED,
        layer=OverlayLayer.SUBJECT,
        caption=OverlayCaption(lines=("NO HELMET", "Track iou-209"), metric="48%"),
    )
    assert _caption_pixels(tiny_confirmed) > 0


def test_a_large_box_keeps_its_caption() -> None:
    big = OverlayBox(
        bounds=(60, 60, 140, 140),
        emphasis=OverlayEmphasis.SUBJECT,
        alert=OverlayAlert.NONE,
        layer=OverlayLayer.SUBJECT,
        caption=OverlayCaption(lines=("Rider", "Track: iou-1"), metric="97%"),
    )
    assert _caption_pixels(big) > 0


def test_a_displaced_caption_is_tied_back_to_its_box() -> None:
    # Two boxes crowding one corner force a caption away from its own subject; a
    # leader line is what stops it reading as a label for whatever it landed on.
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    scene = OverlayScene(
        width=400,
        height=300,
        elements=tuple(
            OverlayBox(
                bounds=(40 + i * 6, 40 + i * 6, 130 + i * 6, 130 + i * 6),
                emphasis=OverlayEmphasis.SUBJECT,
                alert=OverlayAlert.NONE,
                layer=OverlayLayer.SUBJECT,
                caption=OverlayCaption(lines=(f"Rider {i}", "Track: x")),
                key=f"k{i}",
            )
            for i in range(4)
        ),
    )
    painted = int((PillowOverlayRenderer().render(img, scene).sum(axis=2) > 0).sum())
    bare = OverlayScene(
        width=400,
        height=300,
        elements=tuple(e.model_copy(update={"caption": None}) for e in scene.elements),
    )
    bare_painted = int((PillowOverlayRenderer().render(img, bare).sum(axis=2) > 0).sum())
    assert painted > bare_painted  # captions + their leaders were drawn


def test_banner_metric_and_details_are_rendered() -> None:
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    plain = OverlayBanner(title="NO HELMET", lines=("Track iou-1",))
    full = OverlayBanner(
        title="NO HELMET", metric="97%", lines=("Track iou-1",), details=("evt-abc123",)
    )
    lean = int(
        (
            PillowOverlayRenderer()
            .render(img, OverlayScene(width=400, height=200, elements=(plain,)))
            .sum(axis=2)
            > 0
        ).sum()
    )
    rich = int(
        (
            PillowOverlayRenderer()
            .render(img, OverlayScene(width=400, height=200, elements=(full,)))
            .sum(axis=2)
            > 0
        ).sum()
    )
    assert rich > lean


def test_banner_only_scene_renders() -> None:
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    scene = OverlayScene(
        width=200, height=160, elements=(OverlayBanner(title="NO HELMET", icon="⚠"),)
    )
    out = PillowOverlayRenderer().render(img, scene)
    assert int((out.sum(axis=2) > 0).sum()) > 0

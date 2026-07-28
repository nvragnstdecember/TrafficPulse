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

pytest.importorskip("PIL", reason="overlay renderer needs Pillow (the rtdetr extra)")
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


def test_banner_only_scene_renders() -> None:
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    scene = OverlayScene(
        width=200, height=160, elements=(OverlayBanner(title="NO HELMET", icon="⚠"),)
    )
    out = PillowOverlayRenderer().render(img, scene)
    assert int((out.sum(axis=2) > 0).sum()) > 0

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


def test_banner_only_scene_renders() -> None:
    img = np.zeros((160, 200, 3), dtype=np.uint8)
    scene = OverlayScene(width=200, height=160, elements=(OverlayBanner(title="NO HELMET", icon="⚠"),))
    out = PillowOverlayRenderer().render(img, scene)
    assert int((out.sum(axis=2) > 0).sum()) > 0

"""The annotated-video renderer: source clip + overlay metadata -> H.264 mp4."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from trafficpulse.overlay import OverlayCompositor
from trafficpulse.overlay.providers.no_helmet import NoHelmetOverlayProvider
from trafficpulse.overlay.video import render_overlay_video
from trafficpulse.pipeline.helmet_observer import HelmetOverlayFrame, HelmetOverlayRider

pytest.importorskip("PIL", reason="overlay renderer needs Pillow (the rtdetr extra)")


def _write_clip(path: Path, *, frames: int = 8, w: int = 96, h: int = 64, fps: int = 10) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(frames):
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        img[:, (i * 5) % w : (i * 5) % w + 8] = (200, 200, 200)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _rider() -> HelmetOverlayRider:
    return HelmetOverlayRider(
        rider_track_id="iou-1",
        rider_bbox=(10, 8, 60, 56),
        motorcycle_track_id="iou-2",
        motorcycle_bbox=(12, 20, 70, 60),
        head_bbox=(10, 8, 60, 22),
        helmet_label="no_helmet",
        confidence=0.95,
        gated=False,
    )


def _compositor(frame_count: int) -> OverlayCompositor:
    frames = [
        HelmetOverlayFrame(frame_index=i, media_seconds=i / 10.0, riders=(_rider(),))
        for i in range(frame_count)
    ]
    return OverlayCompositor([NoHelmetOverlayProvider(frames, [])])


def test_renders_browser_playable_h264_with_matching_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "src.mp4"
    out = tmp_path / "overlay.mp4"
    _write_clip(source, frames=8)

    result = render_overlay_video(
        source_path=source, output_path=out, compositor=_compositor(8), camera_id="cam-1"
    )

    assert result.output_path == out and out.exists()
    assert result.frames_written == 8
    container = av.open(str(out))
    vs = container.streams.video[0]
    assert vs.codec_context.name == "h264"
    assert vs.codec_context.pix_fmt == "yuv420p"  # browser-decodable
    container.close()


def test_annotations_change_the_pixels(tmp_path: Path) -> None:
    source = tmp_path / "src.mp4"
    annotated = tmp_path / "annotated.mp4"
    passthrough = tmp_path / "plain.mp4"
    _write_clip(source, frames=6)

    render_overlay_video(
        source_path=source, output_path=annotated, compositor=_compositor(6), camera_id="cam-1"
    )
    # empty compositor -> a re-encode with nothing drawn
    render_overlay_video(
        source_path=source, output_path=passthrough, compositor=OverlayCompositor([]),
        camera_id="cam-1",
    )
    assert annotated.read_bytes() != passthrough.read_bytes()


def test_odd_dimensions_are_encoded_evenly(tmp_path: Path) -> None:
    source = tmp_path / "odd.mp4"
    out = tmp_path / "odd_overlay.mp4"
    _write_clip(source, frames=4, w=96, h=64)  # even source; encoder enforces even output

    result = render_overlay_video(
        source_path=source, output_path=out, compositor=OverlayCompositor([]), camera_id="cam-1"
    )
    assert result.width % 2 == 0 and result.height % 2 == 0

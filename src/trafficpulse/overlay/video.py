"""Render an annotated H.264 video from a source clip + overlay metadata.

The production bridge between the overlay framework and a playable artifact: it
**re-decodes** a source video and draws each frame's :class:`OverlayScene` (produced
by an :class:`OverlayCompositor`) onto it, encoding the result as a browser-playable
H.264/yuv420p MP4. It runs **no** model inference -- detection, tracking,
association, and classification already happened; this pass only decodes pixels
(cheap) and draws metadata already produced by the pipeline.

It is violation-agnostic: it takes a compositor and knows nothing about helmets.
Frames the compositor has nothing to say about pass through unannotated, so *every*
displayed frame goes through the overlay system while the original video is left
untouched (a separate file).

PyAV is a base dependency (ingestion uses it); the drawing backend (Pillow) is
imported lazily by the injected renderer, so importing this module pulls in no
drawing dependency until a render actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ..ingestion.video import open_video
from .registry import OverlayCompositor, OverlayFrameRef
from .renderer import OverlayRenderer, PillowOverlayRenderer

_DEFAULT_FPS = 30


@dataclass(frozen=True)
class OverlayVideoResult:
    """Outcome of one annotated-video render."""

    output_path: Path
    frames_written: int
    width: int
    height: int


def render_overlay_video(
    *,
    source_path: str | Path,
    output_path: str | Path,
    compositor: OverlayCompositor,
    camera_id: str,
    renderer: OverlayRenderer | None = None,
    crf: int = 23,
    preset: str = "veryfast",
) -> OverlayVideoResult:
    """Draw ``compositor``'s scenes onto ``source_path`` and encode to ``output_path``.

    Decodes the source once (deterministic ``frame_index`` 0..N, matching the
    inference pass's indices, so the compositor's per-frame metadata aligns),
    renders every frame through the overlay renderer, and muxes H.264. The output
    is constant-frame-rate at the source's (rounded) fps -- correct for playback and
    identical frame count. Returns the artifact's path + frame count.

    Raises:
        Any ingestion error (unreadable source) or PyAV encoding error; the caller
        (the processing service) treats a render failure as non-fatal -- the event
        results are already persisted and the original video still plays.
    """

    renderer = renderer if renderer is not None else PillowOverlayRenderer()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_video(str(source_path)) as reader:
        meta = reader.metadata
        # libx264 + yuv420p require even dimensions; crop at most one edge pixel.
        enc_w = meta.width - (meta.width % 2)
        enc_h = meta.height - (meta.height % 2)
        rate = _encode_rate(meta.fps)

        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = enc_w
            stream.height = enc_h
            stream.pix_fmt = "yuv420p"
            # faststart moves the moov atom to the front so the browser can begin
            # playback before the whole file arrives (progressive streaming).
            stream.options = {"crf": str(crf), "preset": preset, "movflags": "+faststart"}

            written = 0
            for record in reader:
                ref = OverlayFrameRef(
                    camera_id=camera_id,
                    frame_index=record.frame_index,
                    media_seconds=record.timestamp_seconds,
                    width=record.width,
                    height=record.height,
                )
                scene = compositor.scene_for(ref)
                drawn = renderer.render(record.image, scene)
                if drawn.shape[0] != enc_h or drawn.shape[1] != enc_w:
                    drawn = np.ascontiguousarray(drawn[:enc_h, :enc_w])
                frame = av.VideoFrame.from_ndarray(drawn, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                written += 1
            for packet in stream.encode():  # flush the encoder
                container.mux(packet)
        finally:
            container.close()

    return OverlayVideoResult(
        output_path=output_path, frames_written=written, width=enc_w, height=enc_h
    )


def _encode_rate(fps: float | None) -> Fraction:
    """A sane constant output frame rate (source fps rounded; fallback 30)."""

    if fps is None or fps <= 0:
        return Fraction(_DEFAULT_FPS, 1)
    return Fraction(round(fps))

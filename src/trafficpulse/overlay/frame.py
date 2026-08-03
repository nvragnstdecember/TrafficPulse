"""Render individual annotated still frames from a source video (H14).

The still-image counterpart of :mod:`trafficpulse.overlay.video`, and deliberately
its sibling rather than its rival: both decode a source, ask an
:class:`~trafficpulse.overlay.registry.OverlayCompositor` for each frame's scene,
and hand it to the **same** :class:`~trafficpulse.overlay.renderer.OverlayRenderer`.
There is one drawing implementation in this system; this module only chooses *which*
frames it draws and encodes the result as an image instead of a video stream.

Addressing frames by media time
-------------------------------
Callers ask for media times, not frame indices, because media time is what an
evidence manifest durably records and it is independent of where the source file
lives (see :mod:`trafficpulse.evidence.frames`). Each requested time resolves to the
decoded frame whose PTS is nearest it -- normally an exact hit, since the times an
evidence manifest carries are the PTS values of frames the engine actually
processed.

One decode, in order, stopping early
------------------------------------
All requested times are satisfied in a **single sequential pass**: seeking is not
used (a keyframe seek would land on a different frame than the inference pass saw,
which is exactly the kind of silent divergence evidence must not have), and the pass
returns as soon as every request is settled. Rendering the three frames of an event
near the start of a long clip therefore decodes only up to that point, not the whole
file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..ingestion.video import open_video
from .registry import OverlayCompositor, OverlayFrameRef
from .renderer import OverlayBackendUnavailableError, OverlayRenderer, PillowOverlayRenderer

#: Encodable still formats, mapped to their media type.
STILL_MEDIA_TYPES: dict[str, str] = {"png": "image/png", "jpeg": "image/jpeg"}


@dataclass(frozen=True)
class RenderedStill:
    """One rendered still: its encoded bytes and the frame it was drawn from."""

    media_seconds: float
    """The media time that was **requested** (the manifest's recorded value)."""

    frame_index: int
    """The index of the frame actually drawn."""

    frame_media_seconds: float
    """The PTS of the frame actually drawn (equal to ``media_seconds`` on an exact hit)."""

    width: int
    height: int
    data: bytes
    media_type: str


def encode_image(
    image: NDArray[np.uint8], *, image_format: str = "png", quality: int = 90
) -> bytes:
    """Encode an RGB ``uint8`` array as PNG/JPEG bytes (lazy Pillow).

    Pillow is the optional ``overlay`` extra, imported here exactly as the renderer
    imports it, so this module adds no dependency to the base install.
    """

    if image_format not in STILL_MEDIA_TYPES:
        raise ValueError(
            f"unsupported still format {image_format!r}; "
            f"supported: {', '.join(sorted(STILL_MEDIA_TYPES))}"
        )
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise OverlayBackendUnavailableError(
            "rendering evidence stills needs Pillow (the optional 'overlay' extra); "
            "install with: pip install 'trafficpulse[overlay]'"
        ) from exc

    buffer = BytesIO()
    pillow_image = Image.fromarray(np.ascontiguousarray(image))
    if image_format == "jpeg":
        # `optimize` is deterministic and shrinks the artifact; no metadata is
        # written, so the same pixels always encode to the same bytes.
        pillow_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        pillow_image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_stills_at(
    *,
    source_path: str | Path,
    media_times: Sequence[float],
    camera_id: str,
    compositor: OverlayCompositor | None = None,
    renderer: OverlayRenderer | None = None,
    image_format: str = "png",
) -> Mapping[float, RenderedStill]:
    """Render one annotated still per requested media time, in a single decode pass.

    ``compositor`` supplies the overlay metadata already produced by inference; when
    it is ``None`` the frames are rendered **unannotated** rather than not at all --
    a run whose rules contributed no overlay metadata still has real evidence
    pixels, and showing them beats showing nothing.

    Returns a mapping keyed by the *requested* time (so a caller can pair results
    with what it asked for). Times that fall outside the clip resolve to its nearest
    decoded frame; a source with no decodable frames yields an empty mapping.
    """

    requested = sorted(set(media_times))
    if not requested:
        return {}

    active_renderer = renderer if renderer is not None else PillowOverlayRenderer()
    horizon = max(requested)

    best: dict[float, tuple[float, NDArray[np.uint8], int, float, int, int]] = {}
    with open_video(str(source_path), camera_id=camera_id) as reader:
        for record in reader:
            timestamp = record.timestamp_seconds
            for target in requested:
                distance = abs(timestamp - target)
                current = best.get(target)
                if current is None or distance < current[0]:
                    best[target] = (
                        distance,
                        record.image,
                        record.frame_index,
                        timestamp,
                        record.width,
                        record.height,
                    )
            # Frames only ever get further from every request once the stream has
            # passed the last one, so there is nothing left to improve.
            if timestamp > horizon:
                break

    stills: dict[float, RenderedStill] = {}
    for target, (_, image, frame_index, frame_seconds, width, height) in best.items():
        ref = OverlayFrameRef(
            camera_id=camera_id,
            frame_index=frame_index,
            media_seconds=frame_seconds,
            width=width,
            height=height,
        )
        scene = (
            compositor.scene_for(ref)
            if compositor is not None
            else OverlayCompositor().scene_for(ref)
        )
        drawn = active_renderer.render(image, scene)
        stills[target] = RenderedStill(
            media_seconds=target,
            frame_index=frame_index,
            frame_media_seconds=frame_seconds,
            width=width,
            height=height,
            data=encode_image(drawn, image_format=image_format),
            media_type=STILL_MEDIA_TYPES[image_format],
        )
    return stills

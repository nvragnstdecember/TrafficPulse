"""Encoded-frame <-> RGB array conversion for the live camera channel.

The one place live mode touches image codecs. Pillow is imported **lazily**, for
the same reason the overlay renderer imports it lazily: the base install stays
drawing-free, and a deployment without the ``overlay`` extra gets a typed
"live mode needs a drawing backend" message instead of an ImportError traceback.

Both guards here are on untrusted input. A browser sends whatever it sends, so the
byte limit bounds the *compressed* payload and the pixel limit bounds what that
payload is allowed to expand into -- the second is not redundant, because a small
compressed image can decode to an enormous one.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import LiveFrameError, LiveUnavailableError


def _pillow() -> Any:
    """The lazily-imported Pillow ``Image`` module, or a typed unavailability."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise LiveUnavailableError(
            "live camera monitoring needs a drawing backend (Pillow, the optional "
            "'overlay' extra) to decode camera frames and draw annotations; "
            "install with: pip install 'trafficpulse[overlay]'"
        ) from exc
    return Image


def decode_frame(
    data: bytes, *, max_bytes: int, max_pixels: int
) -> NDArray[np.uint8]:
    """Decode one browser-encoded camera frame to an RGB ``uint8`` array.

    Raises:
        LiveFrameError: the payload is empty, exceeds ``max_bytes``, is not a
            readable image, or decodes to more than ``max_pixels``.
        LiveUnavailableError: no drawing backend is installed.
    """

    if not data:
        raise LiveFrameError("received an empty camera frame")
    if len(data) > max_bytes:
        raise LiveFrameError(
            f"camera frame is {len(data)} bytes, over the {max_bytes}-byte limit"
        )
    image_module = _pillow()
    try:
        with image_module.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise LiveFrameError(
                    f"camera frame decodes to {width}x{height}, over the "
                    f"{max_pixels}-pixel limit"
                )
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except LiveFrameError:
        raise
    except Exception as exc:  # noqa: BLE001 - any codec failure is one client error
        raise LiveFrameError(f"camera frame could not be decoded: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise LiveFrameError(f"camera frame decoded to an unexpected shape {array.shape!r}")
    return array


def encode_jpeg(image: NDArray[np.uint8], *, quality: int) -> bytes:
    """Encode an RGB ``uint8`` array as JPEG bytes for the browser."""

    image_module = _pillow()
    buffer = io.BytesIO()
    image_module.fromarray(np.ascontiguousarray(image)).save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()

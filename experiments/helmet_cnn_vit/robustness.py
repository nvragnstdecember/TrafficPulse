"""Robustness slices and synthetic corruptions (P4-U5).

Architecture-review §12 asks for robustness "under blur, low light, occlusion, and
other relevant conditions where data permits", concretely: crop-height buckets,
synthetic corruptions at three severities, and day/night slices where counts allow.

What this corpus permits, and what it does not
-----------------------------------------------
* **Corruptions** -- fully supported, and applied to the *test crops only*, at
  evaluation time. Nothing corrupted is ever trained on.
* **Crop height** -- supported, but §12's absolute buckets (<32 / 32-64 / >64 px)
  are degenerate here: HELMET boxes are whole motorcycles at 1080p, so 7,391 of
  7,406 test crops land in the top bucket. Both the specified buckets and a
  train-derived tertile split are reported, the former so the required slice is not
  quietly dropped and the latter so there is a slice that can actually carry a
  claim. The tertile boundaries come from the **training** split and were fixed
  before training (see PREREGISTRATION.md section 9).
* **Day/night** -- not supported. HELMET is daytime footage with no illumination
  annotation, so the slice is reported as unavailable rather than invented.
* **Occlusion** -- not supported. There is no occlusion label to slice on.

Pillow is imported lazily inside each corruption, so this module stays importable
in a CI environment with no image library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from helmet_rtdetr.models import _Model

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

#: §12's absolute crop-height buckets. Degenerate on this corpus; reported anyway.
SPEC_HEIGHT_EDGES: tuple[float, float] = (32.0, 64.0)

#: Tertile boundaries derived from the TRAIN split before any training run.
TRAIN_HEIGHT_TERTILES: tuple[float, float] = (170.0, 287.0)

#: Three severities per corruption, as pre-registered.
SEVERITIES: tuple[int, ...] = (1, 2, 3)


class SliceCounts(_Model):
    """How many crops fall in each bucket of a slice (reported alongside metrics)."""

    name: str
    buckets: dict[str, int]


def height_bucket(height: float, edges: tuple[float, float]) -> str:
    """Name the bucket ``height`` falls into for a two-edge split."""

    low, high = edges
    if height < low:
        return f"<{low:g}"
    if height <= high:
        return f"{low:g}-{high:g}"
    return f">{high:g}"


def bucket_by_height(
    heights: Sequence[float], *, edges: tuple[float, float] = TRAIN_HEIGHT_TERTILES
) -> dict[str, list[int]]:
    """Group crop indices by height bucket, preserving order within each bucket."""

    groups: dict[str, list[int]] = {}
    for index, height in enumerate(heights):
        groups.setdefault(height_bucket(height, edges), []).append(index)
    return groups


def bucket_by_key(keys: Sequence[str]) -> dict[str, list[int]]:
    """Group crop indices by an arbitrary categorical key (used for the site slice)."""

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return dict(sorted(groups.items()))


# --- synthetic corruptions -------------------------------------------------------
# Each takes a PIL image and a severity in SEVERITIES and returns a new image.
# Parameters are fixed constants, applied identically to both model families.


def gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    """Defocus blur. Radii chosen so severity 3 is clearly degraded but not unreadable."""

    from PIL import ImageFilter

    radius = {1: 1.0, 2: 2.0, 3: 4.0}[severity]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(image: Image.Image, severity: int) -> Image.Image:
    """Horizontal motion blur -- the realistic failure for a moving motorcycle.

    Implemented as a normalised horizontal box kernel, which is what a linear
    left-to-right smear is; ``ImageFilter.BoxBlur`` would smear vertically too.
    """

    from PIL import ImageFilter

    length = {1: 3, 2: 5, 3: 9}[severity]
    kernel = [0.0] * (length * length)
    middle = length // 2
    for column in range(length):
        kernel[middle * length + column] = 1.0 / length
    return image.filter(ImageFilter.Kernel((length, length), kernel, scale=1.0, offset=0))


def jpeg_compression(image: Image.Image, severity: int) -> Image.Image:
    """Re-encode at a low quality -- the compression a real CCTV feed applies."""

    from io import BytesIO

    from PIL import Image as PilImage

    quality = {1: 30, 2: 15, 3: 7}[severity]
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with PilImage.open(buffer) as handle:
        return handle.convert("RGB")


def brightness(image: Image.Image, severity: int) -> Image.Image:
    """Darkening, the closest available proxy for the low-light condition §12 names.

    Note this is a *synthetic* darkening of daylight footage, not real night data;
    the report says so rather than presenting it as a night-time result.
    """

    from PIL import ImageEnhance

    factor = {1: 0.7, 2: 0.5, 3: 0.3}[severity]
    return ImageEnhance.Brightness(image).enhance(factor)


CORRUPTIONS: dict[str, Any] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
    "brightness": brightness,
}


def corruption_variants() -> tuple[tuple[str, int], ...]:
    """Every (corruption, severity) pair, in a deterministic order."""

    return tuple((name, severity) for name in sorted(CORRUPTIONS) for severity in SEVERITIES)


def apply_corruption(image: Image.Image, name: str, severity: int) -> Image.Image:
    """Apply one named corruption at one severity."""

    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    return CORRUPTIONS[name](image, severity)


def slice_counts(name: str, groups: Mapping[str, Sequence[int]]) -> SliceCounts:
    """Summarise a slice's bucket sizes for the report."""

    return SliceCounts(name=name, buckets={k: len(v) for k, v in sorted(groups.items())})

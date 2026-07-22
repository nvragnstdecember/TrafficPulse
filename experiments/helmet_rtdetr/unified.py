"""The canonical unified annotation schema (H2).

One deterministic intermediate representation every source dataset converts into,
so the corpus builder, validation, and export never contain dataset-specific
logic. Strongly typed, frozen, and content-addressed.

Class space
-----------
Binary ``{helmet, no_helmet}`` on rider heads, plus ``motorcycle`` as an
ignorable context class. There is deliberately **no** ``turban`` class: no
permissive dataset labels turbans, so the detector never learns to emit one, and
turban stays a rule-layer exemption in the runtime (the documented Phase-4 gap).

Determinism
-----------
:attr:`UnifiedObject.object_id` is a SHA-256 over the object's own content
(image path + quantised box + label), so identical annotations collide -- which is
exactly what duplicate detection needs -- and the id never depends on load order.
Box coordinates are quantised to 3 decimals before hashing so float noise across
converters cannot split a genuine duplicate.
"""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Self

from pydantic import model_validator

from .models import NonEmptyStr, Slug, _Model


class UnifiedClass(StrEnum):
    """The closed label space of the unified corpus."""

    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    MOTORCYCLE = "motorcycle"


class BBox(_Model):
    """Axis-aligned pixel box in COCO convention: ``(x, y, w, h)``, top-left origin."""

    x: float
    y: float
    w: float
    h: float

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError("bbox coordinates must be finite")
        if self.x < 0 or self.y < 0:
            raise ValueError("bbox x and y must be non-negative")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("bbox w and h must be positive")
        return self

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def quantised_key(self) -> str:
        """A float-noise-stable string of the coordinates (for content hashing)."""

        return "|".join(f"{v:.3f}" for v in (self.x, self.y, self.w, self.h))


class ObjectProvenance(_Model):
    """Where a unified object came from and what it was before mapping."""

    dataset_id: Slug
    dataset_version: NonEmptyStr
    adapter: NonEmptyStr  # the adapter that produced it (e.g. "coco", "helmet-track-csv")
    source_label: NonEmptyStr  # the ORIGINAL label before mapping (audit trail)


class UnifiedObject(_Model):
    """One annotated object in the canonical schema."""

    image_path: NonEmptyStr
    bbox: BBox
    label: UnifiedClass
    provenance: ObjectProvenance
    video_id: str | None = None
    site_id: str | None = None
    frame_id: str | None = None
    frame_index: int | None = None
    ignore: bool = False

    @model_validator(mode="after")
    def _frame_consistency(self) -> Self:
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.frame_id is not None and (self.video_id is None or self.frame_index is None):
            raise ValueError("frame_id requires both video_id and frame_index")
        return self

    @property
    def object_id(self) -> str:
        """Deterministic, content-derived id; identical annotations share it."""

        preimage = "\x1f".join((self.image_path, self.bbox.quantised_key(), self.label.value))
        return "obj-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]

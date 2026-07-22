"""COCO-format converter (H2) -- for Roboflow exports.

COCO detection JSON is a fully specified, stable format, so this adapter is
verified by construction (unlike the HELMET adapters). It reads ``images``,
``annotations``, and ``categories``; maps each annotation's category name through
a caller-supplied label map; and emits one unified object per kept annotation.
Still images: no video/site/frame identity.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..errors import MalformedAnnotationError
from ..unified import BBox, ObjectProvenance, UnifiedObject
from .base import AnnotationAdapter, LabelMap, map_label

# Roboflow COCO exports name their per-split annotation file this way.
DEFAULT_COCO_FILENAME = "_annotations.coco.json"


class CocoAdapter(AnnotationAdapter):
    """Converts a COCO detection JSON into unified objects."""

    def __init__(self, label_map: LabelMap, *, filename: str = DEFAULT_COCO_FILENAME) -> None:
        self._label_map = label_map
        self._filename = filename

    @property
    def name(self) -> str:
        return "coco"

    def _annotation_file(self, root: Path) -> Path | None:
        direct = root / self._filename
        if direct.is_file():
            return direct
        candidates = sorted(root.glob("*.json"))
        return candidates[0] if candidates else None

    def detect(self, root: Path) -> bool:
        return self._annotation_file(root) is not None

    def convert(
        self, root: Path, *, dataset_id: str, dataset_version: str
    ) -> Iterator[UnifiedObject]:
        path = self._annotation_file(root)
        if path is None:
            raise MalformedAnnotationError(f"no COCO annotation JSON found under {root}")
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            images = {img["id"]: img["file_name"] for img in data["images"]}
            categories = {cat["id"]: cat["name"] for cat in data["categories"]}
            annotations = data["annotations"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedAnnotationError(f"malformed COCO JSON at {path}: {exc}") from exc

        # Deterministic order: annotations sorted by (image_id, id).
        for ann in sorted(annotations, key=lambda a: (a.get("image_id"), a.get("id"))):
            try:
                image_id = ann["image_id"]
                category_id = ann["category_id"]
                box = ann["bbox"]
                image_path = images[image_id]
                raw_label = categories[category_id]
            except (KeyError, TypeError) as exc:
                raise MalformedAnnotationError(
                    f"malformed COCO annotation in {path}: {ann!r} ({exc})"
                ) from exc

            mapped = map_label(self._label_map, raw_label, adapter=self.name)
            if mapped is None:
                continue  # recognised but intentionally skipped (e.g. a plate)

            if not (isinstance(box, list | tuple) and len(box) == 4):
                raise MalformedAnnotationError(
                    f"COCO bbox must be [x, y, w, h], got {box!r} in {path}"
                )
            x, y, w, h = (float(v) for v in box)
            yield UnifiedObject(
                image_path=str(image_path),
                bbox=BBox(x=x, y=y, w=w, h=h),
                label=mapped,
                provenance=ObjectProvenance(
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    adapter=self.name,
                    source_label=raw_label,
                ),
            )

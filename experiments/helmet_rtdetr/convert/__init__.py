"""Dataset annotation converters (H2).

Each converter is an independent :class:`AnnotationAdapter` that turns one source
dataset's native annotations into the unified schema. Dataset-specific logic lives
entirely inside its adapter module -- ``coco`` (Roboflow COCO exports, a fully
specified format) and ``helmet`` (the HELMET family, whose exact layout is
**unverified** because the dataset was not available for Step-0 confirmation, so it
is handled behind a sniffed, pluggable set of layout adapters).
"""

from __future__ import annotations

from .base import AnnotationAdapter, map_label
from .coco import CocoAdapter
from .helmet import (
    HELMET_ADAPTERS,
    HELMET_LABEL_MAP,
    HelmetFlatCsvAdapter,
    HelmetLayoutAdapter,
    HelmetTrackCsvAdapter,
    sniff_helmet_layout,
)

__all__ = [
    "AnnotationAdapter",
    "map_label",
    "CocoAdapter",
    "HelmetLayoutAdapter",
    "HelmetTrackCsvAdapter",
    "HelmetFlatCsvAdapter",
    "HELMET_ADAPTERS",
    "HELMET_LABEL_MAP",
    "sniff_helmet_layout",
]

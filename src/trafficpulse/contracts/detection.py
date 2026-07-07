"""Frame-level detection contract (perception-layer output).

Data only: represents a single detector output for one object in one frame.
It performs no detection.
"""

from pydantic import AwareDatetime

from .enums import ObjectClass
from .primitives import (
    BoundingBox,
    Confidence,
    ContractModel,
    ModelRef,
    NonEmptyStr,
    NonNegativeInt,
)


class Detection(ContractModel):
    """One detector output for one object in one frame."""

    detection_id: NonEmptyStr
    camera_id: NonEmptyStr
    frame_index: NonNegativeInt
    timestamp: AwareDatetime
    object_class: ObjectClass
    confidence: Confidence
    bbox: BoundingBox
    source_model: ModelRef | None = None

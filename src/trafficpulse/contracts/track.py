"""Track-state contract (tracker-owned identity + kinematics).

Data only: represents the state of one tracked entity at one instant. It
implements no tracking.
"""

from pydantic import AwareDatetime

from .enums import ObjectClass, TrackStatus
from .primitives import (
    BoundingBox,
    Confidence,
    ContractModel,
    ModelRef,
    NonEmptyStr,
    NonNegativeInt,
    Velocity,
)


class TrackState(ContractModel):
    """State of one tracked entity at a point in time.

    ``tainted`` records the ID-switch guard from architecture-review §13
    (tainted tracks may abstain but never confirm); enforcement of that rule
    is Phase 1 behaviour, not part of this contract.
    """

    track_id: NonEmptyStr
    camera_id: NonEmptyStr
    timestamp: AwareDatetime
    frame_index: NonNegativeInt | None = None
    object_class: ObjectClass
    bbox: BoundingBox
    confidence: Confidence | None = None
    status: TrackStatus
    tainted: bool = False
    velocity: Velocity | None = None
    tracker: ModelRef | None = None

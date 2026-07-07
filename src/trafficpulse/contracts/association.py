"""Association contract (explicit entity-to-entity relationship).

Data only: represents a typed link such as rider->motorcycle, head->rider, or
plate->vehicle, with confidence. It implements no association algorithm.
"""

from pydantic import AwareDatetime

from .enums import AssociationType
from .primitives import Confidence, ContractModel, NonEmptyStr, TimeInterval


class Association(ContractModel):
    """A confidence-weighted relationship between two tracked entities."""

    association_id: NonEmptyStr
    camera_id: NonEmptyStr
    subject_track_id: NonEmptyStr
    object_track_id: NonEmptyStr
    association_type: AssociationType
    confidence: Confidence
    timestamp: AwareDatetime
    interval: TimeInterval | None = None

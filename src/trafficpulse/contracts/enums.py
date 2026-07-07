"""Shared closed-domain enumerations for TrafficPulse domain contracts.

These enumerate closed vocabularies referenced across the contract layer.
They carry no behaviour. Values are stable wire identifiers: changing a value
is a contract change. New members are additive contract revisions.
"""

from enum import StrEnum


class ViolationType(StrEnum):
    """The six locked violation types (spec §2; architecture-review §3)."""

    NO_HELMET = "no_helmet"
    TRIPLE_RIDING = "triple_riding"
    RED_LIGHT_JUMPING = "red_light_jumping"
    WRONG_WAY = "wrong_way"
    ILLEGAL_STOPPING = "illegal_stopping"
    SPEEDING = "speeding"


class ObjectClass(StrEnum):
    """Detector/tracker object classes (architecture-review §5 shared core)."""

    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    AUTO_RICKSHAW = "auto_rickshaw"
    BICYCLE = "bicycle"
    PERSON = "person"
    LICENSE_PLATE = "license_plate"


class TrackStatus(StrEnum):
    """Tracker-owned lifecycle of a single track."""

    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


class AssociationType(StrEnum):
    """Kinds of explicit entity-to-entity relationships."""

    RIDER_OF_MOTORCYCLE = "rider_of_motorcycle"
    HEAD_OF_RIDER = "head_of_rider"
    PLATE_OF_VEHICLE = "plate_of_vehicle"


class LifecycleState(StrEnum):
    """Event/hypothesis lifecycle states (architecture-review §13 FSM).

    Represented as data only; no transition logic lives in the contracts.
    """

    IDLE = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    CLOSED = "closed"
    ABSTAINED = "abstained"


class ReviewStatus(StrEnum):
    """Human-review decision states (architecture-review §21)."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


class SimulatedPenaltyStatus(StrEnum):
    """Simulated-penalty lifecycle states (architecture-review §21).

    Simulation only — never real enforcement.
    """

    ISSUED = "issued"
    PAID = "paid"
    CONTESTED = "contested"
    VOIDED = "voided"


class HelmetState(StrEnum):
    """Four-label helmet ontology (architecture-review §5e, §12).

    The rule-layer mapping (turban -> exempt, uncertain -> abstain) is U3's
    responsibility and is deliberately not encoded in the contracts.
    """

    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    TURBAN = "turban"
    UNCERTAIN = "uncertain"


class SignalState(StrEnum):
    """Traffic-signal states (architecture-review §5b)."""

    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    OFF = "off"
    UNKNOWN = "unknown"


class ZoneKind(StrEnum):
    """Configured scene-geometry zone kinds (architecture-review §16)."""

    ROAD = "road"
    LANE = "lane"
    NO_STOPPING = "no_stopping"
    PARKING = "parking"
    EXCLUSION = "exclusion"
    JUNCTION_CONFLICT = "junction_conflict"
    LIGHT_ROI = "light_roi"
    MEASUREMENT_ZONE = "measurement_zone"


class ProducerKind(StrEnum):
    """What kind of component produced an observation (provenance)."""

    MODEL = "model"
    HEURISTIC = "heuristic"
    RULE = "rule"
    TRACKER = "tracker"
    CALIBRATION = "calibration"


class ArtifactKind(StrEnum):
    """Kinds of evidence artifacts referenced by a manifest."""

    BEFORE_FRAME = "before_frame"
    TRIGGER_FRAME = "trigger_frame"
    AFTER_FRAME = "after_frame"
    CLIP = "clip"
    TRAJECTORY = "trajectory"
    PLATE_CROP = "plate_crop"
    OVERLAY = "overlay"
    OTHER = "other"


class SpeedUnit(StrEnum):
    """Units for reported speed observations."""

    KM_PER_H = "km_per_h"
    M_PER_S = "m_per_s"


class RiderSlot(StrEnum):
    """Ordered rider slot on a motorcycle (architecture-review §5e)."""

    DRIVER = "driver"
    PILLION = "pillion"
    THIRD = "third"
    UNKNOWN = "unknown"

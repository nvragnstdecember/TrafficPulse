"""Observation contracts — the load-bearing perception->reasoning boundary.

Rules consume **only** Observations (architecture-review §14, §15), which is
what makes the reasoning layer replayable deterministically without a model or
GPU. Observations are a **discriminated union** over ``obs_type`` with
explicit, typed variant payloads — never an untyped dict.

The seven variants match the approved Phase 0-F plan and architecture-review
§14: ``in_zone``, ``signal_state``, ``heading_vs_lane``, ``stationary``,
``rider_count``, ``helmet_state``, ``speed``.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter

from .enums import (
    HelmetState,
    ProducerKind,
    RiderSlot,
    SignalState,
    SpeedUnit,
    ZoneKind,
)
from .primitives import (
    Confidence,
    ContractModel,
    DeviationDegrees,
    HeadingDegrees,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
)


class Producer(ContractModel):
    """Provenance for what produced an observation (model/heuristic/rule)."""

    name: NonEmptyStr
    version: NonEmptyStr
    kind: ProducerKind


class ObservationBase(ContractModel):
    """Common provenance/context shared by every observation variant.

    ``track_id`` is optional because some observations are scene-level rather
    than track-bound (for example ``signal_state``).
    """

    observation_id: NonEmptyStr
    camera_id: NonEmptyStr
    track_id: NonEmptyStr | None = None
    timestamp: AwareDatetime
    confidence: Confidence | None = None
    producer: Producer


class InZoneObservation(ObservationBase):
    """Whether a track occupies a configured scene zone."""

    obs_type: Literal["in_zone"] = "in_zone"
    zone_id: NonEmptyStr
    zone_kind: ZoneKind
    is_inside: bool


class SignalStateObservation(ObservationBase):
    """Observed traffic-signal state for a configured light ROI."""

    obs_type: Literal["signal_state"] = "signal_state"
    signal_state: SignalState
    roi_id: NonEmptyStr | None = None


class HeadingVsLaneObservation(ObservationBase):
    """Track heading relative to a lane's legal direction (wrong-way)."""

    obs_type: Literal["heading_vs_lane"] = "heading_vs_lane"
    lane_id: NonEmptyStr
    heading_degrees: HeadingDegrees
    legal_heading_degrees: HeadingDegrees | None = None
    deviation_degrees: DeviationDegrees
    is_contradiction: bool


class StationaryObservation(ObservationBase):
    """Whether a track is stationary, with optional dwell (illegal stopping)."""

    obs_type: Literal["stationary"] = "stationary"
    is_stationary: bool
    speed_estimate: NonNegativeFloat | None = None
    dwell_seconds: NonNegativeFloat | None = None


class RiderCountObservation(ObservationBase):
    """Estimated rider count on a motorcycle (triple riding)."""

    obs_type: Literal["rider_count"] = "rider_count"
    rider_count: NonNegativeInt
    motorcycle_track_id: NonEmptyStr | None = None


class HelmetStateObservation(ObservationBase):
    """Per-rider-slot helmet state.

    Carries the four-label ontology {helmet, no_helmet, turban, uncertain}
    required later by U3. The rule-layer mapping (turban -> exempt,
    uncertain -> abstain) is U3's responsibility and is not encoded here.
    """

    obs_type: Literal["helmet_state"] = "helmet_state"
    helmet_state: HelmetState
    rider_slot: RiderSlot | None = None
    crop_height_px: NonNegativeFloat | None = None


class SpeedObservation(ObservationBase):
    """Ground-plane speed with explicit uncertainty (v +/- sigma; §17)."""

    obs_type: Literal["speed"] = "speed"
    speed_value: NonNegativeFloat
    speed_sigma: NonNegativeFloat
    unit: SpeedUnit
    measurement_zone_id: NonEmptyStr | None = None
    sample_count: NonNegativeInt | None = None


# Ordered tuple of the seven concrete observation variants.
OBSERVATION_VARIANTS: tuple[type[ObservationBase], ...] = (
    InZoneObservation,
    SignalStateObservation,
    HeadingVsLaneObservation,
    StationaryObservation,
    RiderCountObservation,
    HelmetStateObservation,
    SpeedObservation,
)

# The bare union of the seven observation variants (used as the TypeAdapter
# parameter and anywhere the discriminator metadata is not required).
ObservationUnion = (
    InZoneObservation
    | SignalStateObservation
    | HeadingVsLaneObservation
    | StationaryObservation
    | RiderCountObservation
    | HelmetStateObservation
    | SpeedObservation
)

# The discriminated-union alias, tagged by ``obs_type``. Use this as the field
# type wherever a model needs to hold an arbitrary observation.
Observation = Annotated[ObservationUnion, Field(discriminator="obs_type")]

# Adapter for validating/parsing/serializing a standalone Observation and for
# exporting its JSON Schema.
ObservationAdapter: TypeAdapter[ObservationUnion] = TypeAdapter(Observation)

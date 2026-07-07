"""TrafficPulse domain contracts (Phase 0-F, U2).

The typed, serializable layer boundary of the data flow
(architecture-review §14):

    Detection -> TrackState -> Association -> Observation -> TemporalState
    -> ViolationHypothesis -> ConfirmedEvent -> EvidenceManifest -> ReviewCase
    -> SimulatedPenalty

These models represent **data** only. They implement no pipelines, rule
engines, tracking, persistence, ML inference, geometry, or evidence
generation. Observations are the load-bearing perception->reasoning contract
and are modelled as a discriminated union so rules can consume them without
depending on detector-specific objects.
"""

from typing import TYPE_CHECKING

from .association import Association
from .detection import Detection
from .enums import (
    ArtifactKind,
    AssociationType,
    HelmetState,
    LifecycleState,
    ObjectClass,
    ProducerKind,
    ReviewStatus,
    RiderSlot,
    SignalState,
    SimulatedPenaltyStatus,
    SpeedUnit,
    TrackStatus,
    ViolationType,
    ZoneKind,
)
from .event import ConfirmedEvent
from .evidence import ArtifactReference, EvidenceManifest, OcrResult, RuleTraceStep
from .hypothesis import ViolationHypothesis
from .observations import (
    OBSERVATION_VARIANTS,
    HeadingVsLaneObservation,
    HelmetStateObservation,
    InZoneObservation,
    Observation,
    ObservationAdapter,
    ObservationBase,
    Producer,
    RiderCountObservation,
    SignalStateObservation,
    SpeedObservation,
    StationaryObservation,
)
from .penalty import SIMULATION_DISCLAIMER, SimulatedAmount, SimulatedPenalty
from .primitives import (
    BoundingBox,
    ConfidenceBreakdown,
    ContractModel,
    MeasuredValue,
    ModelRef,
    TimeInterval,
    Velocity,
)
from .review import ReviewCase
from .temporal import TemporalState
from .track import TrackState

if TYPE_CHECKING:
    # Imported for typing only; the runtime access path is __getattr__ below,
    # which defers importing schema_export until the name is first used.
    from .schema_export import TOP_LEVEL_CONTRACTS, export_schemas

__all__ = [
    "OBSERVATION_VARIANTS",
    "SIMULATION_DISCLAIMER",
    "TOP_LEVEL_CONTRACTS",
    # value objects / base
    "ContractModel",
    "BoundingBox",
    "Velocity",
    "TimeInterval",
    "MeasuredValue",
    "ModelRef",
    "ConfidenceBreakdown",
    # enums
    "ViolationType",
    "ObjectClass",
    "TrackStatus",
    "AssociationType",
    "LifecycleState",
    "ReviewStatus",
    "SimulatedPenaltyStatus",
    "HelmetState",
    "SignalState",
    "ZoneKind",
    "ProducerKind",
    "ArtifactKind",
    "SpeedUnit",
    "RiderSlot",
    # perception / tracking / association
    "Detection",
    "TrackState",
    "Association",
    # observations
    "Producer",
    "ObservationBase",
    "InZoneObservation",
    "SignalStateObservation",
    "HeadingVsLaneObservation",
    "StationaryObservation",
    "RiderCountObservation",
    "HelmetStateObservation",
    "SpeedObservation",
    "Observation",
    "ObservationAdapter",
    # temporal / hypothesis / event
    "TemporalState",
    "ViolationHypothesis",
    "ConfirmedEvent",
    # evidence / review / penalty
    "ArtifactReference",
    "OcrResult",
    "RuleTraceStep",
    "EvidenceManifest",
    "ReviewCase",
    "SimulatedAmount",
    "SimulatedPenalty",
    # schema export
    "export_schemas",
]


def __getattr__(name: str) -> object:
    """Lazily expose the schema-export helpers (PEP 562).

    Importing ``schema_export`` eagerly at package-import time would place
    ``trafficpulse.contracts.schema_export`` in ``sys.modules`` before runpy
    executes it as ``__main__``, triggering a RuntimeWarning under
    ``python -m trafficpulse.contracts.schema_export``. Deferring the import
    keeps that command warning-free while preserving the public API.
    """

    if name in {"export_schemas", "TOP_LEVEL_CONTRACTS"}:
        from . import schema_export

        return getattr(schema_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Shared value objects and constrained scalar types for TrafficPulse contracts.

These are data-only building blocks (no behaviour) composed by the domain
contracts. There is deliberately no ``common`` package: shared contract-level
types live here, inside ``contracts`` (Phase 0-F plan, U2 non-goals).
"""

from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# --- Constrained scalar aliases ---------------------------------------------
# Reused across contracts to keep validation uniform and self-documenting.
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
HeadingDegrees = Annotated[float, Field(ge=0.0, le=360.0)]
DeviationDegrees = Annotated[float, Field(ge=0.0, le=180.0)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class ContractModel(BaseModel):
    """Base for all TrafficPulse domain contracts.

    Immutable (``frozen``) and strict (``extra='forbid'``): contracts describe
    fixed data snapshots at a layer boundary, not mutable runtime objects. This
    supports deterministic replay and prevents silent field drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class BoundingBox(ContractModel):
    """Axis-aligned image-space box in pixel coordinates (top-left origin)."""

    x1: NonNegativeFloat
    y1: NonNegativeFloat
    x2: NonNegativeFloat
    y2: NonNegativeFloat

    @model_validator(mode="after")
    def _check_geometry(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box requires x2 > x1 and y2 > y1")
        return self


class Velocity(ContractModel):
    """Image-space motion vector, pixels per second (data only)."""

    vx: float
    vy: float


class TimeInterval(ContractModel):
    """Closed time interval with timezone-aware bounds."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end < self.start:
            raise ValueError("interval end must not precede start")
        return self


class MeasuredValue(ContractModel):
    """A named measured quantity with an optional unit (data only).

    Used for a rule's measured values and thresholds instead of an untyped
    mapping, so measurements stay explicit and inspectable.
    """

    name: NonEmptyStr
    value: float
    unit: str | None = None


class ModelRef(ContractModel):
    """Reference to a model/weights version (metadata only; no loading)."""

    name: NonEmptyStr
    version: NonEmptyStr
    weights_hash: Sha256Hex | None = None


class ConfidenceBreakdown(ContractModel):
    """Typed confidence components (architecture-review §13).

    Deliberately not a probability unless calibration is demonstrated. Every
    component is optional and, when present, lies in ``[0, 1]``.
    """

    detector: Confidence | None = None
    classifier: Confidence | None = None
    association: Confidence | None = None
    temporal_consistency: Confidence | None = None
    track_continuity: Confidence | None = None
    geometric_margin: Confidence | None = None
    calibration_quality: Confidence | None = None
    aggregate: Confidence | None = None

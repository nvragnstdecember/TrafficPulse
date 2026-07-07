"""Violation-hypothesis contract (candidate under accumulation).

Data only: represents a candidate violation *before* confirmation, distinct
from a ``ConfirmedEvent``. It computes no confidence and runs no rule logic.
"""

from pydantic import Field

from .enums import LifecycleState, ViolationType
from .primitives import (
    ConfidenceBreakdown,
    ContractModel,
    MeasuredValue,
    NonEmptyStr,
    TimeInterval,
)


class ViolationHypothesis(ContractModel):
    """A candidate violation accumulating evidence, pre-confirmation."""

    hypothesis_id: NonEmptyStr
    violation_type: ViolationType
    camera_id: NonEmptyStr
    track_ids: tuple[NonEmptyStr, ...] = ()
    interval: TimeInterval
    state: LifecycleState
    rule_id: NonEmptyStr
    rule_version: str | None = None
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    reasons: tuple[str, ...] = ()
    measurements: tuple[MeasuredValue, ...] = ()
    thresholds: tuple[MeasuredValue, ...] = ()

"""Confirmed-event contract (immutable confirmed violation).

Data only: represents a confirmed violation event, kept structurally distinct
from a ``ViolationHypothesis``. It generates no evidence artifacts.
"""

from pydantic import AwareDatetime, Field

from .enums import ViolationType
from .primitives import (
    ConfidenceBreakdown,
    ContractModel,
    MeasuredValue,
    ModelRef,
    NonEmptyStr,
    Sha256Hex,
)


class ConfirmedEvent(ContractModel):
    """An immutable confirmed violation event.

    All contracts are frozen, but confirmation is where immutability is
    architecturally load-bearing (architecture-review §14, §19). ``models``,
    ``code_version`` and ``scene_config_hash`` capture the provenance every
    event embeds; ``source_hypothesis_id`` links back to its origin.
    """

    event_id: NonEmptyStr
    violation_type: ViolationType
    camera_id: NonEmptyStr
    track_ids: tuple[NonEmptyStr, ...] = ()
    start_at: AwareDatetime
    trigger_at: AwareDatetime
    end_at: AwareDatetime | None = None
    rule_id: NonEmptyStr
    rule_version: str | None = None
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    measurements: tuple[MeasuredValue, ...] = ()
    thresholds: tuple[MeasuredValue, ...] = ()
    scene_config_hash: Sha256Hex | None = None
    models: tuple[ModelRef, ...] = ()
    code_version: str | None = None
    source_hypothesis_id: NonEmptyStr | None = None
    created_at: AwareDatetime

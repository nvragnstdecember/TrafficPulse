"""Temporal-state contract (per-(track, rule) accumulator snapshot).

Data only: this represents rule-engine lifecycle *state as data*. It defines
no transitions and no FSM — that behaviour is Phase 1 rule-engine work
(architecture-review §13).
"""

from pydantic import AwareDatetime

from .enums import LifecycleState
from .primitives import ContractModel, NonEmptyStr, NonNegativeInt


class TemporalState(ContractModel):
    """A snapshot of one per-(track, rule) temporal accumulator.

    Captures the candidate-accumulation vs confirmed distinction via
    ``lifecycle_state`` without implementing the transitions between them.
    """

    state_id: NonEmptyStr
    camera_id: NonEmptyStr
    track_id: NonEmptyStr
    rule_id: NonEmptyStr
    lifecycle_state: LifecycleState
    accumulated_score: float | None = None
    observation_count: NonNegativeInt = 0
    tainted: bool = False
    first_observation_at: AwareDatetime | None = None
    last_observation_at: AwareDatetime | None = None
    updated_at: AwareDatetime

"""Rule Engine Core for TrafficPulse (P1-U3).

The first reasoning unit: a deterministic, rule-agnostic engine that consumes
frozen U2 ``Observation`` contracts and manages the lifecycle of
``ViolationHypothesis`` candidates through the engine-internal state machine

    NEW -> CANDIDATE -> ACTIVE -> CLOSED   (with ABANDONED as an off-ramp)

It creates, looks up, updates, transitions, closes, and abandons hypotheses and
emits ``ViolationHypothesis`` snapshots. It knows nothing about specific traffic
rules, embeds no threshold, uses no wall clock or randomness, and never produces
a ``ConfirmedEvent`` or reaches ``CONFIRMED`` (that is P1-U4). See
``engine.py``/``states.py`` for the routing, determinism, and contract-mapping
contracts.
"""

from .engine import (
    HypothesisKey,
    HypothesisRecord,
    RuleEngine,
    UnknownHypothesisError,
    to_violation_hypothesis,
)
from .states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    EngineState,
    IllegalTransitionError,
    RuleEngineError,
    is_terminal,
    is_valid_transition,
    to_lifecycle_state,
    validate_transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    # state machine
    "EngineState",
    "is_terminal",
    "is_valid_transition",
    "validate_transition",
    "to_lifecycle_state",
    # engine
    "RuleEngine",
    "HypothesisKey",
    "HypothesisRecord",
    "to_violation_hypothesis",
    # errors
    "RuleEngineError",
    "IllegalTransitionError",
    "UnknownHypothesisError",
]

"""Engine lifecycle states, transition rules, and contract mapping (P1-U3).

The rule engine's *internal* lifecycle is the five-state machine this unit
specifies:

    NEW -> CANDIDATE -> ACTIVE -> CLOSED
                     \\-> ABANDONED  (off-ramp from any non-terminal state)

``CLOSED`` and ``ABANDONED`` are terminal. ``CONFIRMED`` is deliberately **not**
implemented here -- confirmation is P1-U4.

Relationship to the frozen contract vocabulary
----------------------------------------------
The frozen U2 ``LifecycleState`` enum is ``{IDLE, CANDIDATE, CONFIRMED, CLOSED,
ABSTAINED}`` and ``ViolationHypothesis.state`` is typed with it. The prompt's
state names ``NEW``, ``ACTIVE``, and ``ABANDONED`` are not members of that frozen
enum, so this module keeps a distinct engine-internal :class:`EngineState` and
maps it onto ``LifecycleState`` (via :func:`to_lifecycle_state`) only when a
``ViolationHypothesis`` snapshot is produced. The contract is never mutated and
never receives a non-member value. ``CONFIRMED`` is never emitted.
"""

from enum import StrEnum

from ..contracts.enums import LifecycleState


class EngineState(StrEnum):
    """The engine-internal hypothesis lifecycle (see module docstring)."""

    NEW = "new"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CLOSED = "closed"
    ABANDONED = "abandoned"


# Deterministic transition table. A transition is legal iff the target is in the
# source's allowed set; terminal states have an empty set (no outgoing edges).
ALLOWED_TRANSITIONS: dict[EngineState, frozenset[EngineState]] = {
    EngineState.NEW: frozenset({EngineState.CANDIDATE, EngineState.ABANDONED}),
    EngineState.CANDIDATE: frozenset({EngineState.ACTIVE, EngineState.ABANDONED}),
    EngineState.ACTIVE: frozenset({EngineState.CLOSED, EngineState.ABANDONED}),
    EngineState.CLOSED: frozenset(),
    EngineState.ABANDONED: frozenset(),
}

# Terminal states have no outgoing transitions.
TERMINAL_STATES: frozenset[EngineState] = frozenset(
    state for state, targets in ALLOWED_TRANSITIONS.items() if not targets
)

# Engine state -> frozen contract LifecycleState. ``ACTIVE`` and ``CANDIDATE``
# both map to the contract's pre-confirmation ``CANDIDATE`` (the frozen
# vocabulary has no distinct "active"); ``ABANDONED`` maps to ``ABSTAINED``;
# ``NEW`` maps to ``IDLE`` (pre-candidate). ``CONFIRMED`` is intentionally
# unreachable from any engine state.
_LIFECYCLE_MAP: dict[EngineState, LifecycleState] = {
    EngineState.NEW: LifecycleState.IDLE,
    EngineState.CANDIDATE: LifecycleState.CANDIDATE,
    EngineState.ACTIVE: LifecycleState.CANDIDATE,
    EngineState.CLOSED: LifecycleState.CLOSED,
    EngineState.ABANDONED: LifecycleState.ABSTAINED,
}


class RuleEngineError(Exception):
    """Base class for all rule-engine errors."""


class IllegalTransitionError(RuleEngineError):
    """Raised when a requested lifecycle transition is not permitted."""


def is_terminal(state: EngineState) -> bool:
    """Return ``True`` if ``state`` has no outgoing transitions."""

    return state in TERMINAL_STATES


def is_valid_transition(source: EngineState, target: EngineState) -> bool:
    """Return ``True`` if ``source -> target`` is a permitted transition."""

    return target in ALLOWED_TRANSITIONS[source]


def validate_transition(source: EngineState, target: EngineState) -> None:
    """Raise :class:`IllegalTransitionError` unless ``source -> target`` is legal."""

    if not is_valid_transition(source, target):
        raise IllegalTransitionError(f"illegal transition {source.value} -> {target.value}")


def to_lifecycle_state(state: EngineState) -> LifecycleState:
    """Map an engine state to the frozen contract ``LifecycleState``.

    Never returns ``LifecycleState.CONFIRMED`` (confirmation is P1-U4).
    """

    return _LIFECYCLE_MAP[state]

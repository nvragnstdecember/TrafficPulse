"""Tests for the engine state machine and contract mapping (P1-U3).

Pure, data-only tests of ``EngineState``, the transition table, terminal states,
transition validation, and the mapping to the frozen ``LifecycleState`` (which
must never yield ``CONFIRMED``).
"""

import pytest

from trafficpulse.contracts.enums import LifecycleState
from trafficpulse.rules.states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    EngineState,
    IllegalTransitionError,
    is_terminal,
    is_valid_transition,
    to_lifecycle_state,
    validate_transition,
)

_LEGAL = frozenset(
    {
        (EngineState.NEW, EngineState.CANDIDATE),
        (EngineState.NEW, EngineState.ABANDONED),
        (EngineState.CANDIDATE, EngineState.ACTIVE),
        (EngineState.CANDIDATE, EngineState.ABANDONED),
        (EngineState.ACTIVE, EngineState.CLOSED),
        (EngineState.ACTIVE, EngineState.ABANDONED),
    }
)


# --- state vocabulary --------------------------------------------------------
def test_engine_state_members() -> None:
    assert [s.value for s in EngineState] == ["new", "candidate", "active", "closed", "abandoned"]


def test_confirmed_is_not_an_engine_state() -> None:
    assert "confirmed" not in {s.value for s in EngineState}


def test_terminal_states() -> None:
    assert set(TERMINAL_STATES) == {EngineState.CLOSED, EngineState.ABANDONED}
    assert is_terminal(EngineState.CLOSED)
    assert is_terminal(EngineState.ABANDONED)
    assert not is_terminal(EngineState.NEW)
    assert not is_terminal(EngineState.CANDIDATE)
    assert not is_terminal(EngineState.ACTIVE)


# --- transition table --------------------------------------------------------
def test_transition_table_matches_legal_set() -> None:
    for source in EngineState:
        for target in EngineState:
            assert is_valid_transition(source, target) == ((source, target) in _LEGAL)


def test_terminal_states_have_no_outgoing_edges() -> None:
    assert ALLOWED_TRANSITIONS[EngineState.CLOSED] == frozenset()
    assert ALLOWED_TRANSITIONS[EngineState.ABANDONED] == frozenset()


def test_no_self_loops() -> None:
    for state in EngineState:
        assert not is_valid_transition(state, state)


def test_validate_transition_passes_for_legal() -> None:
    for source, target in _LEGAL:
        validate_transition(source, target)  # must not raise


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (EngineState.NEW, EngineState.ACTIVE),
        (EngineState.NEW, EngineState.CLOSED),
        (EngineState.CANDIDATE, EngineState.NEW),
        (EngineState.CANDIDATE, EngineState.CLOSED),
        (EngineState.ACTIVE, EngineState.CANDIDATE),
        (EngineState.ACTIVE, EngineState.NEW),
        (EngineState.CLOSED, EngineState.ABANDONED),
        (EngineState.CLOSED, EngineState.ACTIVE),
        (EngineState.ABANDONED, EngineState.CANDIDATE),
    ],
)
def test_validate_transition_raises_for_illegal(
    source: EngineState, target: EngineState
) -> None:
    with pytest.raises(IllegalTransitionError):
        validate_transition(source, target)


# --- mapping to the frozen contract vocabulary -------------------------------
def test_lifecycle_mapping() -> None:
    assert to_lifecycle_state(EngineState.NEW) is LifecycleState.IDLE
    assert to_lifecycle_state(EngineState.CANDIDATE) is LifecycleState.CANDIDATE
    assert to_lifecycle_state(EngineState.ACTIVE) is LifecycleState.CANDIDATE
    assert to_lifecycle_state(EngineState.CLOSED) is LifecycleState.CLOSED
    assert to_lifecycle_state(EngineState.ABANDONED) is LifecycleState.ABSTAINED


def test_mapping_never_produces_confirmed() -> None:
    produced = {to_lifecycle_state(s) for s in EngineState}
    assert LifecycleState.CONFIRMED not in produced

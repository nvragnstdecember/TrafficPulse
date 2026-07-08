"""Tests for the deterministic rule-engine core (P1-U3).

Exercises creation, lookup, duplicate suppression, observation attachment,
lifecycle transitions (legal and illegal), abandonment/closure, deterministic
replay and ordering, id stability, ``ViolationHypothesis`` materialization /
serialization, and absence of mutation. Uses synthetic observations only -- no
detector, tracker, or video.
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import (
    ConfidenceBreakdown,
    InZoneObservation,
    Producer,
    SignalStateObservation,
    ViolationHypothesis,
)
from trafficpulse.contracts.enums import (
    LifecycleState,
    ProducerKind,
    SignalState,
    ViolationType,
    ZoneKind,
)
from trafficpulse.rules import (
    EngineState,
    HypothesisKey,
    IllegalTransitionError,
    RuleEngine,
    UnknownHypothesisError,
    to_violation_hypothesis,
)

BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_PRODUCER = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)


def _obs(
    obs_id: str, *, secs: float = 0.0, camera: str = "cam-1", track: str | None = "track-1"
) -> InZoneObservation:
    return InZoneObservation(
        observation_id=obs_id,
        camera_id=camera,
        track_id=track,
        timestamp=BASE + timedelta(seconds=secs),
        producer=_PRODUCER,
        zone_id="zone-1",
        zone_kind=ZoneKind.LANE,
        is_inside=True,
    )


def _run_wrong_way(engine: RuleEngine) -> str:
    """A canonical NEW->CANDIDATE->ACTIVE->CLOSED run; returns the id."""

    record = engine.ingest(
        _obs("o1", secs=0), rule_id="wrong_way", violation_type=ViolationType.WRONG_WAY
    )
    engine.ingest(
        _obs("o2", secs=1), rule_id="wrong_way", violation_type=ViolationType.WRONG_WAY
    )
    engine.promote(record.hypothesis_id)
    engine.activate(record.hypothesis_id)
    engine.close(record.hypothesis_id)
    return record.hypothesis_id


# --- creation ----------------------------------------------------------------
def test_ingest_creates_new_hypothesis() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert record.state is EngineState.NEW
    assert record.camera_id == "cam-1"
    assert record.violation_type is ViolationType.WRONG_WAY
    assert record.rule_id == "r"
    assert record.track_id == "track-1"
    assert record.observation_count == 1
    assert record.generation == 0
    assert record.first_at == BASE
    assert record.last_at == BASE


def test_rule_version_captured_at_creation() -> None:
    engine = RuleEngine()
    record = engine.ingest(
        _obs("o1"), rule_id="r", violation_type=ViolationType.SPEEDING, rule_version="1.2.3"
    )
    assert record.rule_version == "1.2.3"


# --- lookup ------------------------------------------------------------------
def test_get_and_get_open() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert engine.get(record.hypothesis_id) is not None
    assert engine.get("missing") is None
    key = HypothesisKey("cam-1", ViolationType.WRONG_WAY, "r", "track-1")
    assert engine.get_open(key) is not None
    assert engine.get_open(key).hypothesis_id == record.hypothesis_id  # type: ignore[union-attr]


def test_get_open_none_after_terminal() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.abandon(record.hypothesis_id)
    key = HypothesisKey("cam-1", ViolationType.WRONG_WAY, "r", "track-1")
    assert engine.get_open(key) is None


# --- duplicate suppression / attachment --------------------------------------
def test_duplicate_observation_suppressed() -> None:
    engine = RuleEngine()
    first = engine.ingest(_obs("dup", secs=5), rule_id="r", violation_type=ViolationType.SPEEDING)
    again = engine.ingest(_obs("dup", secs=5), rule_id="r", violation_type=ViolationType.SPEEDING)
    assert first.hypothesis_id == again.hypothesis_id
    assert again.observation_count == 1


def test_second_observation_attaches_to_same_hypothesis() -> None:
    engine = RuleEngine()
    a = engine.ingest(_obs("o1", secs=0), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    b = engine.ingest(_obs("o2", secs=2), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert a.hypothesis_id == b.hypothesis_id
    assert b.observation_ids == ("o1", "o2")
    assert b.first_at == BASE
    assert b.last_at == BASE + timedelta(seconds=2)
    assert len(engine.records()) == 1


def test_track_ids_accumulate_deduped_sorted() -> None:
    engine = RuleEngine()
    engine.ingest(_obs("o1", track="track-b"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    # Same camera + rule + violation but the key includes track_id, so a second
    # track opens a distinct hypothesis; here we attach two obs of ONE track.
    rec = engine.ingest(
        _obs("o2", secs=1, track="track-b"), rule_id="r", violation_type=ViolationType.WRONG_WAY
    )
    assert rec.track_ids == ("track-b",)


def test_observation_without_track_id() -> None:
    engine = RuleEngine()
    obs = SignalStateObservation(
        observation_id="s1",
        camera_id="cam-1",
        track_id=None,
        timestamp=BASE,
        producer=_PRODUCER,
        signal_state=SignalState.RED,
    )
    record = engine.ingest(obs, rule_id="r", violation_type=ViolationType.RED_LIGHT_JUMPING)
    assert record.track_id is None
    assert record.track_ids == ()
    assert record.observation_count == 1


# --- multi-rule routing ------------------------------------------------------
def test_same_observation_feeds_multiple_rules() -> None:
    engine = RuleEngine()
    a = engine.ingest(_obs("m"), rule_id="ruleA", violation_type=ViolationType.WRONG_WAY)
    b = engine.ingest(_obs("m"), rule_id="ruleB", violation_type=ViolationType.RED_LIGHT_JUMPING)
    assert a.hypothesis_id != b.hypothesis_id
    assert len(engine.records()) == 2


# --- lifecycle transitions ---------------------------------------------------
def test_happy_path_transitions_and_lifecycle_mapping() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert engine.promote(record.hypothesis_id).state is EngineState.CANDIDATE
    assert engine.activate(record.hypothesis_id).state is EngineState.ACTIVE
    closed = engine.close(record.hypothesis_id)
    assert closed.state is EngineState.CLOSED
    assert closed.lifecycle_state is LifecycleState.CLOSED
    assert closed.is_terminal


def test_abandon_from_each_non_terminal_state() -> None:
    preps = (
        lambda e, h: None,  # abandon from NEW
        lambda e, h: e.promote(h),  # abandon from CANDIDATE
        lambda e, h: (e.promote(h), e.activate(h)),  # abandon from ACTIVE
    )
    for prep in preps:
        engine = RuleEngine()
        record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
        prep(engine, record.hypothesis_id)
        abandoned = engine.abandon(record.hypothesis_id)
        assert abandoned.state is EngineState.ABANDONED
        assert abandoned.lifecycle_state is LifecycleState.ABSTAINED


def test_transition_records_reason() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.abandon(record.hypothesis_id, reason="short track")
    assert engine.get(record.hypothesis_id).reasons == ("short track",)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "action",
    [
        lambda e, h: e.activate(h),  # NEW -> ACTIVE
        lambda e, h: e.close(h),  # NEW -> CLOSED
        lambda e, h: e.transition(h, EngineState.NEW),  # NEW -> NEW (self-loop)
    ],
)
def test_illegal_transition_from_new_raises(action) -> None:  # type: ignore[no-untyped-def]
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    with pytest.raises(IllegalTransitionError):
        action(engine, record.hypothesis_id)


def test_illegal_transition_from_candidate_and_terminal() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.promote(record.hypothesis_id)
    with pytest.raises(IllegalTransitionError):
        engine.close(record.hypothesis_id)  # CANDIDATE -> CLOSED illegal
    engine.abandon(record.hypothesis_id)
    with pytest.raises(IllegalTransitionError):
        engine.close(record.hypothesis_id)  # ABANDONED -> CLOSED illegal (terminal)


def test_unknown_hypothesis_raises() -> None:
    engine = RuleEngine()
    with pytest.raises(UnknownHypothesisError):
        engine.close("does-not-exist")


# --- reopening after terminal ------------------------------------------------
def test_reopen_after_terminal_new_generation() -> None:
    engine = RuleEngine()
    first = engine.ingest(_obs("p1", secs=0), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.abandon(first.hypothesis_id)
    second = engine.ingest(_obs("p2", secs=1), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert second.hypothesis_id != first.hypothesis_id
    assert (first.generation, second.generation) == (0, 1)
    assert len(engine.records()) == 2


# --- determinism / ordering / id stability -----------------------------------
def test_deterministic_replay() -> None:
    e1 = RuleEngine()
    e2 = RuleEngine()
    _run_wrong_way(e1)
    _run_wrong_way(e2)
    assert e1.records() == e2.records()
    assert e1.snapshot() == e2.snapshot()


def test_id_stable_across_engines() -> None:
    assert _run_wrong_way(RuleEngine()) == _run_wrong_way(RuleEngine())


def test_ingest_all_order_independent() -> None:
    e1 = RuleEngine()
    e2 = RuleEngine()
    e1.ingest_all(
        [_obs("b", secs=2), _obs("a", secs=0), _obs("c", secs=1)],
        rule_id="r",
        violation_type=ViolationType.WRONG_WAY,
    )
    e2.ingest_all(
        [_obs("a", secs=0), _obs("c", secs=1), _obs("b", secs=2)],
        rule_id="r",
        violation_type=ViolationType.WRONG_WAY,
    )
    assert e1.snapshot() == e2.snapshot()


def test_attached_observations_sorted_by_time_then_id() -> None:
    engine = RuleEngine()
    engine.ingest(_obs("z", secs=3), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.ingest(_obs("a", secs=1), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    rec = engine.ingest(_obs("m", secs=2), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert rec.observation_ids == ("a", "m", "z")
    assert rec.first_at == BASE + timedelta(seconds=1)
    assert rec.last_at == BASE + timedelta(seconds=3)


# --- serialization -----------------------------------------------------------
def test_to_violation_hypothesis_is_valid_contract() -> None:
    engine = RuleEngine()
    hid = _run_wrong_way(engine)
    vh = to_violation_hypothesis(engine.get(hid))  # type: ignore[arg-type]
    assert isinstance(vh, ViolationHypothesis)
    assert vh.hypothesis_id == hid
    assert vh.state is LifecycleState.CLOSED
    assert vh.interval.start == BASE
    assert vh.interval.end == BASE + timedelta(seconds=1)
    assert vh.track_ids == ("track-1",)


def test_violation_hypothesis_json_roundtrip() -> None:
    engine = RuleEngine()
    hid = _run_wrong_way(engine)
    vh = to_violation_hypothesis(engine.get(hid))  # type: ignore[arg-type]
    assert ViolationHypothesis.model_validate_json(vh.model_dump_json()) == vh


def test_engine_embeds_no_measurements_or_thresholds() -> None:
    engine = RuleEngine()
    hid = _run_wrong_way(engine)
    vh = to_violation_hypothesis(engine.get(hid))  # type: ignore[arg-type]
    assert vh.measurements == ()
    assert vh.thresholds == ()
    assert vh.confidence == ConfidenceBreakdown()


# --- no mutation -------------------------------------------------------------
def test_records_are_frozen() -> None:
    engine = RuleEngine()
    record = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    with pytest.raises(FrozenInstanceError):
        record.state = EngineState.CANDIDATE  # type: ignore[misc]


def test_prior_record_reference_not_mutated_by_transition() -> None:
    engine = RuleEngine()
    before = engine.ingest(_obs("o1"), rule_id="r", violation_type=ViolationType.WRONG_WAY)
    engine.promote(before.hypothesis_id)
    # The earlier reference is unchanged; the engine replaced it with a new one.
    assert before.state is EngineState.NEW
    assert engine.get(before.hypothesis_id).state is EngineState.CANDIDATE  # type: ignore[union-attr]


def test_input_observation_not_mutated() -> None:
    obs = _obs("o1", secs=4)
    snapshot = obs.model_dump()
    engine = RuleEngine()
    engine.ingest(obs, rule_id="r", violation_type=ViolationType.WRONG_WAY)
    assert obs.model_dump() == snapshot

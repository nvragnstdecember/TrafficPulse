"""Tests for wrong-way temporal reasoning and confirmation (P1-U4).

Covers configuration loading, temporal reasoning (legal/sustained/short/recovery/
boundary/duplicate/order/dropped/replay), integration with the generic P1-U3
engine, deterministic ConfirmedEvent creation and identity, and P1-U2 synthetic
scenario integration. Synthetic observations/trajectories only -- no detector,
tracker, or video.
"""

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import (
    ConfirmedEvent,
    HeadingVsLaneObservation,
    ParameterStatus,
    Producer,
    SceneConfig,
    ViolationType,
    scene_config_hash,
)
from trafficpulse.contracts.enums import LifecycleState, ProducerKind
from trafficpulse.observations.heading import (
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
from trafficpulse.rules.engine import RuleEngine, to_violation_hypothesis
from trafficpulse.rules.states import EngineState, IllegalTransitionError
from trafficpulse.rules.wrong_way import WrongWayReasoner, wrong_way_parameters
from trafficpulse.synth import (
    generate_abrupt_turn,
    generate_legal,
    generate_noisy,
    generate_short_track,
    generate_slight_drift,
    generate_stationary,
    generate_track,
    generate_truncated,
    generate_wrong_way,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
SCENE = SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))
PARAMS = wrong_way_parameters(SCENE)
SCH = scene_config_hash(SCENE)
NORTH = next(d for d in SCENE.legal_directions if d.direction_id == "dir-north")
LV = NORTH.vector
LANE = NORTH.zone_ids[0]

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)

# A wrong-way trajectory long enough to exceed min_persistence=1.0s at 30 fps
# (the default 30-frame scenario spans only ~0.93s, which correctly does NOT
# confirm -- see test_default_wrong_way_scenario_too_short_to_confirm).
WRONG_WAY_LONG = dict(frame_count=45)


def _cobs(i: int, *, contradiction: bool = True, track: str = "tk", spacing: float = 0.1,
          obs_id: str | None = None) -> HeadingVsLaneObservation:
    return HeadingVsLaneObservation(
        observation_id=obs_id if obs_id is not None else f"o-{track}-{i:03d}",
        camera_id="cam-1",
        track_id=track,
        timestamp=BASE + timedelta(seconds=i * spacing),
        producer=PRODUCER,
        lane_id="lane",
        heading_degrees=90.0 if contradiction else 270.0,
        legal_heading_degrees=270.0,
        deviation_degrees=180.0 if contradiction else 0.0,
        is_contradiction=contradiction,
    )


def _derived(track):  # type: ignore[no-untyped-def]
    return derive_heading_observations(
        track, legal_direction=LV, lane_id=LANE, deviation_max_degrees=PARAMS.deviation_max_degrees
    )


def _pipeline(track):  # type: ignore[no-untyped-def]
    derivation = derive_heading_observations_with_taint(
        track, legal_direction=LV, lane_id=LANE, deviation_max_degrees=PARAMS.deviation_max_degrees
    )
    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    events = reasoner.run_derivation(derivation)
    return reasoner, events


def _taint(track, start, stop):  # type: ignore[no-untyped-def]
    """Return a copy of ``track`` with samples in ``[start, stop)`` marked tainted."""

    return [
        ts.model_copy(update={"tainted": True}) if start <= i < stop else ts
        for i, ts in enumerate(track)
    ]


# --- configuration -----------------------------------------------------------
def test_parameters_loaded_from_scene() -> None:
    assert PARAMS.deviation_max_degrees == 120.0
    assert PARAMS.min_persistence_seconds == 1.0
    assert PARAMS.min_speed == 1.5


def test_parameters_are_provisional() -> None:
    assert PARAMS.deviation_status is ParameterStatus.PROVISIONAL
    assert PARAMS.persistence_status is ParameterStatus.PROVISIONAL
    assert PARAMS.min_speed_status is ParameterStatus.PROVISIONAL


def test_missing_wrong_way_block_raises() -> None:
    raw = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    raw["rule_parameters"] = [
        b for b in raw["rule_parameters"] if b["violation_type"] != "wrong_way"
    ]
    with pytest.raises(ValueError, match="wrong_way"):
        wrong_way_parameters(SceneConfig.model_validate(raw))


def test_unset_persistence_raises() -> None:
    raw = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    for block in raw["rule_parameters"]:
        if block["violation_type"] == "wrong_way":
            for param in block["parameters"]:
                if param["id"] == "min_persistence":
                    param["value"] = None
                    param["status"] = "unset"
    with pytest.raises(ValueError, match="min_persistence"):
        wrong_way_parameters(SceneConfig.model_validate(raw))


# --- temporal reasoning ------------------------------------------------------
def test_legal_observations_do_not_confirm() -> None:
    _, events = _pipeline(generate_legal())
    assert events == ()


def test_sustained_wrong_way_confirms() -> None:
    _, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    assert len(events) == 1


def test_short_burst_does_not_confirm() -> None:
    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    events = reasoner.run([_cobs(i) for i in range(5)])  # 0.4 s span < 1.0 s
    assert events == ()


def test_recovery_before_persistence_does_not_confirm() -> None:
    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    seq = [_cobs(0), _cobs(1), _cobs(2), _cobs(3, contradiction=False), _cobs(4), _cobs(5)]
    assert reasoner.run(seq) == ()


def test_persistence_boundary_inclusive() -> None:
    at = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    assert len(at.run([_cobs(i) for i in range(11)])) == 1  # t: 0.0 .. 1.0 -> confirm
    under = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    assert under.run([_cobs(i) for i in range(10)]) == ()  # t: 0.0 .. 0.9 -> no


def test_duplicate_observations_deterministic() -> None:
    base = [_cobs(i) for i in range(11)]
    dup = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    events_dup = dup.run(base + [_cobs(10), _cobs(10)])
    plain = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    events_plain = plain.run(base)
    assert len(events_dup) == 1
    assert events_dup[0].event_id == events_plain[0].event_id


def test_out_of_order_matches_in_order() -> None:
    seq = [_cobs(i) for i in range(15)]  # spans 1.4 s
    shuffled = list(seq)
    random.Random(1).shuffle(shuffled)
    ordered = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(seq)
    unordered = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(shuffled)
    assert len(ordered) == len(unordered) == 1
    assert ordered[0].event_id == unordered[0].event_id


def test_dropped_observations_are_timestamp_driven() -> None:
    # A wrong-way track with a dropped block still confirms: persistence is
    # measured from observation timestamps, not from frame counts, and a gap is
    # not a legal recovery.
    track = generate_track(
        start=(960.0, 660.0), direction=(0.0, 1.0), step_size=10.0, frame_count=45,
        dropped_frames=range(15, 25), track_id="ww-gap",
    )
    _, events = _pipeline(track)
    assert len(events) == 1


def test_default_wrong_way_scenario_too_short_to_confirm() -> None:
    # Documents the timing reality: 30 frames at 30 fps span ~0.93 s < 1.0 s.
    _, events = _pipeline(generate_wrong_way())
    assert events == ()


def test_replay_is_deterministic() -> None:
    track = generate_wrong_way(**WRONG_WAY_LONG)
    e1 = _pipeline(track)[1]
    e2 = _pipeline(track)[1]
    assert e1[0].event_id == e2[0].event_id
    assert e1[0].model_dump_json() == e2[0].model_dump_json()


# --- hypothesis integration (P1-U3 engine) -----------------------------------
def test_recovery_abandons_hypothesis_via_engine() -> None:
    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    reasoner.run([_cobs(0), _cobs(1), _cobs(2), _cobs(3, contradiction=False)])
    records = reasoner.engine.records()
    assert len(records) == 1
    assert records[0].state is EngineState.ABANDONED


def test_confirmed_hypothesis_is_active_and_maps_to_candidate() -> None:
    reasoner, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    record = reasoner.engine.get(events[0].source_hypothesis_id)
    assert record is not None
    assert record.state is EngineState.ACTIVE
    # Engine ACTIVE maps to the frozen contract CANDIDATE; confirmation lives in
    # the separate ConfirmedEvent, not in a CONFIRMED hypothesis state.
    assert to_violation_hypothesis(record).state is LifecycleState.CANDIDATE


def test_generic_engine_is_violation_agnostic() -> None:
    # The same generic engine drives a NON-wrong-way violation unchanged.
    engine = RuleEngine()
    record = engine.ingest(_cobs(0), rule_id="speeding", violation_type=ViolationType.SPEEDING)
    engine.promote(record.hypothesis_id)
    engine.activate(record.hypothesis_id)
    closed = engine.close(record.hypothesis_id)
    assert closed.violation_type is ViolationType.SPEEDING
    assert closed.state is EngineState.CLOSED


def test_generic_engine_source_has_no_wrong_way_identifiers() -> None:
    import trafficpulse.rules.engine as engine_mod
    import trafficpulse.rules.states as states_mod

    # Wrong-way-specific behavioral identifiers must not leak into the generic
    # engine (the disclaimer word "wrong-way" in a docstring is not one of these).
    for module in (engine_mod, states_mod):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for token in ("heading_deviation", "min_persistence", "is_contradiction",
                      "legal_direction", "heading_vs_lane"):
            assert token not in source, f"{module.__name__} leaked {token!r}"


def test_confirmed_hypothesis_has_attached_observations() -> None:
    track = generate_wrong_way(**WRONG_WAY_LONG)
    reasoner, events = _pipeline(track)
    record = reasoner.engine.get(events[0].source_hypothesis_id)
    assert record is not None
    assert record.observation_count == len(_derived(track))


def test_illegal_generic_transition_still_rejected() -> None:
    reasoner, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    hid = events[0].source_hypothesis_id
    assert hid is not None
    with pytest.raises(IllegalTransitionError):
        reasoner.engine.transition(hid, EngineState.NEW)


def test_replay_same_reasoner_no_duplicate() -> None:
    obs = _derived(generate_wrong_way(**WRONG_WAY_LONG))
    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    first = reasoner.run(obs)
    second = reasoner.run(obs)
    assert len(first) == 1
    assert second == ()
    assert len(reasoner.engine.records()) == 1


# --- ConfirmedEvent ----------------------------------------------------------
def test_event_is_wrong_way_and_valid_contract() -> None:
    _, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    event = events[0]
    assert isinstance(event, ConfirmedEvent)
    assert event.violation_type is ViolationType.WRONG_WAY
    assert event.camera_id == "cam-synthetic-01"
    assert event.track_ids == ("wrong-way-track",)
    assert ConfirmedEvent.model_validate_json(event.model_dump_json()) == event


def test_event_timestamps_from_observations() -> None:
    _, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    event = events[0]
    assert event.trigger_at > event.start_at
    assert event.created_at == event.trigger_at  # deterministic, not wall-clock
    assert (event.trigger_at - event.start_at).total_seconds() >= PARAMS.min_persistence_seconds


def test_event_scene_config_hash_matches_u5() -> None:
    _, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    assert events[0].scene_config_hash == SCH
    assert events[0].scene_config_hash == scene_config_hash(SCENE)


def test_event_id_is_deterministic() -> None:
    a = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))[1][0]
    b = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))[1][0]
    assert a.event_id == b.event_id
    assert a == b


def test_event_identity_changes_with_camera() -> None:
    common = dict(direction=(0.0, 1.0), step_size=10.0, frame_count=45, start=(960.0, 660.0))
    t1 = generate_track(camera_id="cam-A", track_id="tk", **common)
    t2 = generate_track(camera_id="cam-B", track_id="tk", **common)
    e1 = _pipeline(t1)[1][0]
    e2 = _pipeline(t2)[1][0]
    assert e1.event_id != e2.event_id


def test_event_thresholds_recorded_from_config() -> None:
    _, events = _pipeline(generate_wrong_way(**WRONG_WAY_LONG))
    thresholds = {m.name: m.value for m in events[0].thresholds}
    assert thresholds["heading_deviation_max"] == PARAMS.deviation_max_degrees
    assert thresholds["min_persistence"] == PARAMS.min_persistence_seconds
    # No fabricated confidence.
    assert events[0].confidence.aggregate is None


def test_only_one_event_per_episode() -> None:
    _, events = _pipeline(generate_wrong_way(frame_count=60))
    assert len(events) == 1


# --- P1-U2 synthetic scenario integration ------------------------------------
def test_scenario_legal_no_confirm() -> None:
    assert _pipeline(generate_legal())[1] == ()


def test_scenario_wrong_way_confirms() -> None:
    assert len(_pipeline(generate_wrong_way(**WRONG_WAY_LONG))[1]) == 1


def test_scenario_stationary_no_confirm() -> None:
    assert _pipeline(generate_stationary())[1] == ()


def test_scenario_short_no_confirm() -> None:
    assert _pipeline(generate_short_track())[1] == ()  # legal, 3 frames
    assert _pipeline(generate_wrong_way(frame_count=10))[1] == ()  # short wrong-way burst


def test_scenario_noisy_legal_no_false_confirm() -> None:
    assert _pipeline(generate_noisy(seed=3))[1] == ()


def test_scenario_noisy_wrong_way_confirms() -> None:
    track = generate_track(
        start=(960.0, 660.0), direction=(0.0, 1.0), step_size=10.0, frame_count=45,
        jitter_sigma=1.0, seed=9, track_id="noisy-ww",
    )
    assert len(_pipeline(track)[1]) == 1


def test_scenario_abrupt_turn_no_confirm() -> None:
    # A 90-degree turn never exceeds the 120-degree threshold -> no sustained
    # contradiction -> no confirmation (persistence respected).
    assert _pipeline(generate_abrupt_turn())[1] == ()


def test_scenario_truncated_deterministic_non_confirm() -> None:
    assert _pipeline(generate_truncated())[1] == ()
    assert _pipeline(generate_truncated())[1] == ()


def test_scenario_slight_drift_no_false_confirm() -> None:
    assert _pipeline(generate_slight_drift())[1] == ()


# --- taint discontinuity (regression: taint must not bridge to confirmation) --
def _ww_base(frame_count: int = 45):  # type: ignore[no-untyped-def]
    return generate_track(
        start=(960.0, 660.0), direction=(0.0, 1.0), step_size=10.0,
        frame_count=frame_count, track_id="tk",
    )


def test_taint_clean_baseline_confirms() -> None:
    # 1. Clean sustained wrong-way still confirms.
    assert len(_pipeline(_ww_base())[1]) == 1


def test_ordinary_gap_bridges_but_taint_does_not() -> None:
    # 2 + 3. An ordinary dropped-observation gap keeps timestamp-driven bridging
    # (pre- and post-gap support combine to confirm); the SAME interval marked
    # tainted resets the run, so support cannot bridge it and it does not confirm.
    gapped = generate_track(
        start=(960.0, 660.0), direction=(0.0, 1.0), step_size=10.0,
        frame_count=45, dropped_frames=range(15, 25), track_id="tk",
    )
    assert len(_pipeline(gapped)[1]) == 1  # ordinary gap -> bridges -> confirms
    assert _pipeline(_taint(_ww_base(), 15, 25))[1] == ()  # taint -> no bridge


def test_taint_interval_does_not_silently_confirm() -> None:
    # 3. The exact bridging scenario from the audit now abstains.
    assert _pipeline(_taint(_ww_base(), 15, 25))[1] == ()


def test_taint_pre_support_plus_insufficient_post_does_not_confirm() -> None:
    # 4. Insufficient clean support after taint does not confirm, and the
    # pre-taint hypothesis is abandoned at the discontinuity (via the engine).
    reasoner, events = _pipeline(_taint(_ww_base(), 15, 25))
    assert events == ()
    assert any(r.state is EngineState.ABANDONED for r in reasoner.engine.records())


def _resume_track():  # type: ignore[no-untyped-def]
    # Short early taint, then a long clean sustained wrong-way segment.
    base = generate_track(
        start=(960.0, 600.0), direction=(0.0, 1.0), step_size=8.0,
        frame_count=60, track_id="tk",
    )
    return _taint(base, 3, 8)


def test_resume_after_taint_confirms_fresh_episode() -> None:
    # 6. A genuinely sustained clean segment after the taint confirms on its own,
    # with persistence measured from the POST-taint restart (not the pre-taint
    # start), proving explicit resume semantics.
    track = _resume_track()
    reasoner, events = _pipeline(track)
    assert len(events) == 1
    assert events[0].start_at >= track[8].timestamp  # at/after first clean post-taint sample
    assert events[0].start_at > track[7].timestamp  # strictly after the tainted interval


def test_taint_resume_is_deterministic() -> None:
    # 5. Behavior after taint is deterministic (including the confirming case).
    track = _resume_track()
    e1 = _pipeline(track)[1]
    e2 = _pipeline(track)[1]
    assert len(e1) == 1
    assert e1[0].event_id == e2[0].event_id
    assert e1[0].model_dump_json() == e2[0].model_dump_json()

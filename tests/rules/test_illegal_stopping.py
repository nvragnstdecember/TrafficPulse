"""Tests for the illegal-stopping temporal reasoner (P2-U4).

Deterministic, model-free tests over the frozen ``ConfirmedEvent`` contract: the
two-stream ``(camera, track, timestamp)`` join of in-zone + stationary facts;
dwell accumulation, threshold-boundary confirmation, recovery/exit/movement
resets, taint-restart and gap breaks; separate events for separate episodes;
event content/identity/provenance; multi-track and multi-zone isolation; and
determinism, ordering, duplicate, immutability, and import-boundary audits.

Every test builds synthetic ``TrackState`` sequences, derives the real P2-U2
in-zone and P2-U3 stationary observation streams (no geometry/stationarity is
recomputed here), and reasons over them -- proving compatibility with both
derivations. No detector, tracker, video, persistence, wall-clock, or model.

The reasoner tests derive stationarity with ``window=2`` (crisp per-step
stationarity) so episode boundaries are timing-predictable; the P2-U3 default
window/jitter-robustness is covered by ``tests/observations/test_stationary.py``.
"""

import copy
import random
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import (
    BoundingBox,
    ConfirmedEvent,
    InZoneObservation,
    ModelRef,
    ObjectClass,
    ParameterStatus,
    Producer,
    ProducerKind,
    SceneConfig,
    StationaryObservation,
    TrackState,
    TrackStatus,
    ViolationType,
    ZoneKind,
    scene_config_hash,
)
from trafficpulse.contracts.scene import Zone, ZoneType
from trafficpulse.observations.stationary import (
    StationaryDerivation,
    derive_stationary_observations_with_taint,
)
from trafficpulse.observations.zones import (
    InZoneDerivation,
    derive_in_zone_observations_with_taint,
)
from trafficpulse.rules import illegal_stopping as illegal_stopping_module
from trafficpulse.rules.engine import RuleEngine, to_violation_hypothesis
from trafficpulse.rules.illegal_stopping import (
    IllegalStoppingParameters,
    IllegalStoppingReasoner,
    StoppedInZoneStep,
    illegal_stopping_parameters,
    join_stopped_in_zone,
)
from trafficpulse.rules.states import EngineState

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
SCENE = SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))
PARAMS = illegal_stopping_parameters(SCENE)
PARAMS_NO_GAP = replace(PARAMS, max_observation_gap_seconds=None)
SCH = scene_config_hash(SCENE)
ZONES = SCENE.zones

# The example scene's no-stopping zone (configs/scenes/example-scene.yaml).
NO_STOP_POLY: tuple[tuple[float, float], ...] = (
    (1260.0, 1060.0),
    (1520.0, 1060.0),
    (1470.0, 660.0),
    (1310.0, 660.0),
)
INSIDE = (1390.0, 900.0)  # bottom-center inside zone-no-stop
OUTSIDE = (400.0, 900.0)  # bottom-center clearly outside every zone

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_HALF_W = 20.0
_HEIGHT = 40.0
_PROD = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)


# --- track / derivation helpers ----------------------------------------------
def _track(
    bottoms: list[tuple[float, float]],
    *,
    camera_id: str = "cam-x",
    track_id: str = "trk-x",
    tainted_indices: tuple[int, ...] = (),
    interval_s: float = 1.0,
    timestamps: list[datetime] | None = None,
) -> list[TrackState]:
    """Build a track whose bbox bottom-center equals each point in ``bottoms``."""

    states: list[TrackState] = []
    for i, (x, y) in enumerate(bottoms):
        ts = timestamps[i] if timestamps is not None else _BASE + timedelta(seconds=interval_s * i)
        states.append(
            TrackState(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=ts,
                frame_index=i,
                object_class=ObjectClass.CAR,
                bbox=BoundingBox(x1=x - _HALF_W, y1=y - _HEIGHT, x2=x + _HALF_W, y2=y),
                status=TrackStatus.ACTIVE,
                tainted=(i in tainted_indices),
            )
        )
    return states


def _still_inside(n: int, **kw) -> list[TrackState]:  # type: ignore[no-untyped-def]
    return _track([INSIDE] * n, **kw)


def _derivations(
    track: list[TrackState], *, zones=ZONES, motion_threshold: float | None = None, window: int = 2
) -> tuple[InZoneDerivation, StationaryDerivation]:
    in_zone = derive_in_zone_observations_with_taint(track, zones=zones)
    stationary = derive_stationary_observations_with_taint(
        track, window=window, motion_threshold=motion_threshold
    )
    return in_zone, stationary


def _reason(
    track: list[TrackState],
    *,
    params: IllegalStoppingParameters = PARAMS,
    zones=ZONES,
    models: tuple[ModelRef, ...] = (),
    window: int = 2,
) -> tuple[IllegalStoppingReasoner, tuple[ConfirmedEvent, ...]]:
    in_zone, stationary = _derivations(
        track, zones=zones, motion_threshold=params.motion_threshold, window=window
    )
    reasoner = IllegalStoppingReasoner(
        RuleEngine(), params, scene_config_hash=SCH, models=models
    )
    return reasoner, reasoner.run_join(in_zone, stationary)


# --- configuration -----------------------------------------------------------
def test_parameters_loaded_from_scene() -> None:
    assert PARAMS.stationary_duration_seconds == 10.0
    assert PARAMS.motion_threshold == 0.5
    assert PARAMS.max_observation_gap_seconds == 2.0


def test_parameters_are_provisional() -> None:
    assert PARAMS.duration_status is ParameterStatus.PROVISIONAL
    assert PARAMS.motion_threshold_status is ParameterStatus.PROVISIONAL
    assert PARAMS.max_observation_gap_status is ParameterStatus.PROVISIONAL


def test_missing_illegal_stopping_block_raises() -> None:
    raw = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    raw["rule_parameters"] = [
        b for b in raw["rule_parameters"] if b["violation_type"] != "illegal_stopping"
    ]
    with pytest.raises(ValueError, match="illegal_stopping"):
        illegal_stopping_parameters(SceneConfig.model_validate(raw))


def test_unset_stationary_duration_raises() -> None:
    raw = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    for block in raw["rule_parameters"]:
        if block["violation_type"] == "illegal_stopping":
            for param in block["parameters"]:
                if param["id"] == "stationary_duration":
                    param["value"] = None
                    param["status"] = "unset"
    with pytest.raises(ValueError, match="stationary_duration"):
        illegal_stopping_parameters(SceneConfig.model_validate(raw))


def test_max_observation_gap_optional() -> None:
    raw = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
    for block in raw["rule_parameters"]:
        if block["violation_type"] == "illegal_stopping":
            block["parameters"] = [
                p for p in block["parameters"] if p["id"] != "max_observation_gap"
            ]
    params = illegal_stopping_parameters(SceneConfig.model_validate(raw))
    assert params.max_observation_gap_seconds is None
    assert params.max_observation_gap_status is ParameterStatus.UNSET


# --- two-stream join ---------------------------------------------------------
def test_join_requires_stationary_and_inside() -> None:
    # stationary + inside -> stopped; every other combination -> not stopped.
    def _steps(bottoms: list[tuple[float, float]]) -> list[StoppedInZoneStep]:
        in_zone, stationary = _derivations(_track(bottoms))
        steps, _ = join_stopped_in_zone(in_zone, stationary)
        return steps

    stationary_inside = _steps([INSIDE, INSIDE, INSIDE])
    assert stationary_inside and all(s.stopped_in_zone for s in stationary_inside)
    # stationary + outside
    assert all(not s.stopped_in_zone for s in _steps([OUTSIDE, OUTSIDE, OUTSIDE]))
    # moving + inside (large per-step jump keeps net > epsilon at window=2)
    moving_inside = _steps([(1300.0, 900.0), (1400.0, 900.0), (1300.0, 900.0)])
    assert all(not s.stopped_in_zone for s in moving_inside)
    # moving + outside
    moving_outside = _steps([(400.0, 900.0), (500.0, 900.0), (400.0, 900.0)])
    assert all(not s.stopped_in_zone for s in moving_outside)


def test_join_missing_in_zone_folds_to_not_stopped() -> None:
    # A stationary carrier with no matching in-zone fact is never a stop.
    track = _still_inside(3)
    _, stationary = _derivations(track)
    empty_in_zone = InZoneDerivation(observations=(), taint_restart_ids=frozenset())
    steps, restarts = join_stopped_in_zone(empty_in_zone, stationary)
    assert steps and all(not s.stopped_in_zone for s in steps)
    assert restarts == frozenset()


def test_join_ignores_non_no_stopping_zone_kind() -> None:
    # A non-no-stopping in-zone fact (e.g. LANE) is filtered out of the join.
    track = _still_inside(3)
    _, stationary = _derivations(track)
    lane_obs = tuple(
        InZoneObservation(
            observation_id=f"lane-{i}",
            camera_id=o.camera_id,
            track_id=o.track_id,
            timestamp=o.timestamp,
            producer=_PROD,
            zone_id="zone-lane",
            zone_kind=ZoneKind.LANE,
            is_inside=True,
        )
        for i, o in enumerate(stationary.observations)
    )
    lane_derivation = InZoneDerivation(observations=lane_obs, taint_restart_ids=frozenset())
    steps, _ = join_stopped_in_zone(lane_derivation, stationary)
    assert all(not s.stopped_in_zone for s in steps)  # LANE membership does not stop


def test_join_multi_zone_inside_any() -> None:
    # Inside any eligible no-stopping zone counts (OR across zones).
    other = Zone(
        zone_id="zone-no-stop-2",
        zone_type=ZoneType.NO_STOPPING,
        enabled=True,
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),  # far away
    )
    real = next(z for z in ZONES if z.zone_id == "zone-no-stop")
    in_zone, stationary = _derivations(_still_inside(3), zones=(other, real))
    steps, _ = join_stopped_in_zone(in_zone, stationary)
    assert steps and all(s.stopped_in_zone for s in steps)  # inside `real`, not `other`


# --- core temporal reasoning -------------------------------------------------
def test_stationary_in_zone_confirms() -> None:
    _, events = _reason(_still_inside(12))  # obs t=1..11; dwell reaches 10.0 at t=11
    assert len(events) == 1
    event = events[0]
    assert event.violation_type is ViolationType.ILLEGAL_STOPPING
    assert event.end_at is None
    assert event.camera_id == "cam-x"
    assert event.track_ids == ("trk-x",)
    assert event.rule_id == "illegal_stopping"
    assert event.rule_version == "0.1.0-provisional"
    assert event.scene_config_hash == SCH


def test_short_dwell_does_not_confirm() -> None:
    _, events = _reason(_still_inside(11))  # obs t=1..10; max dwell 9.0 < 10.0
    assert events == ()


def test_dwell_boundary_inclusive() -> None:
    # dwell exactly == stationary_duration confirms (>=), one step under does not.
    _, at = _reason(_still_inside(12))  # dwell hits exactly 10.0 at t=11
    _, under = _reason(_still_inside(11))  # only reaches 9.0
    assert len(at) == 1
    assert under == ()


def test_moving_in_zone_does_not_confirm() -> None:
    bottoms = [(1300.0 + (i % 2) * 100.0, 900.0) for i in range(14)]  # oscillate inside
    _, events = _reason(_track(bottoms))
    assert events == ()


def test_stationary_outside_does_not_confirm() -> None:
    _, events = _reason(_track([OUTSIDE] * 14))
    assert events == ()


def test_exit_zone_after_confirm_closes_run() -> None:
    # Confirm, then leave the zone: the hypothesis closes (not abandoned).
    track = _still_inside(12) + _track([OUTSIDE] * 2, timestamps=[
        _BASE + timedelta(seconds=12), _BASE + timedelta(seconds=13)
    ])
    reasoner, events = _reason(track)
    assert len(events) == 1
    record = reasoner.engine.get(events[0].source_hypothesis_id)
    assert record is not None and record.state is EngineState.CLOSED


def test_recovery_before_threshold_abandons_via_engine() -> None:
    # Stationary-in-zone briefly, then move out before the threshold -> abandon.
    track = _still_inside(4) + _track(
        [OUTSIDE] * 3,
        timestamps=[_BASE + timedelta(seconds=4 + i) for i in range(3)],
    )
    reasoner, events = _reason(track)
    assert events == ()
    records = reasoner.engine.records()
    assert len(records) == 1
    assert records[0].state is EngineState.ABANDONED


def test_min_two_observations_required() -> None:
    # A single-state track yields no observations at all; nothing to confirm.
    _, events = _reason(_still_inside(1))
    assert events == ()


# --- event content, identity, provenance -------------------------------------
def test_event_timing_fields() -> None:
    _, events = _reason(_still_inside(12))
    event = events[0]
    assert event.start_at == _BASE + timedelta(seconds=1)  # first stopped obs (t=1)
    assert event.trigger_at == _BASE + timedelta(seconds=11)  # dwell first >= 10.0
    assert event.created_at == event.trigger_at  # deterministic data timestamp
    assert event.start_at.tzinfo is not None and event.trigger_at.tzinfo is not None


def test_measurements_and_thresholds() -> None:
    _, events = _reason(_still_inside(12))
    event = events[0]
    dwell = {m.name: m for m in event.measurements}
    assert dwell["dwell_seconds"].value == 10.0
    assert dwell["dwell_seconds"].unit == "seconds"
    thresholds = {t.name: t for t in event.thresholds}
    assert thresholds["stationary_duration"].value == 10.0
    assert thresholds["stationary_duration"].unit == "seconds"
    # motion_threshold recorded (provenance) but never applied.
    assert thresholds["motion_threshold"].value == 0.5
    assert thresholds["motion_threshold"].unit == "m_per_s"


def test_event_id_deterministic_and_provenance_invariant() -> None:
    m1 = (ModelRef(name="rtdetr", version="v1"), ModelRef(name="iou-tracker", version="v0"))
    _, plain = _reason(_still_inside(12))
    _, with_models = _reason(_still_inside(12), models=m1)
    assert plain[0].event_id == with_models[0].event_id  # models never enter the id
    assert plain[0].models == ()
    assert with_models[0].models == m1
    # every other decision-bearing field is identical too
    assert plain[0].start_at == with_models[0].start_at
    assert plain[0].trigger_at == with_models[0].trigger_at


def test_motion_threshold_does_not_affect_decision() -> None:
    hot = replace(PARAMS, motion_threshold=99.0)
    _, base = _reason(_still_inside(12))
    _, other = _reason(_still_inside(12), params=hot)
    assert base[0].event_id == other[0].event_id  # motion_threshold is recorded, not applied


def test_weights_hash_stays_none() -> None:
    models = (ModelRef(name="rtdetr", version="v1"), ModelRef(name="iou-tracker", version="v0"))
    _, events = _reason(_still_inside(12), models=models)
    assert all(ref.weights_hash is None for ref in events[0].models)


def test_confirmed_hypothesis_is_active() -> None:
    reasoner, events = _reason(_still_inside(12))
    record = reasoner.engine.get(events[0].source_hypothesis_id)
    assert record is not None
    assert record.state is EngineState.ACTIVE
    # engine ACTIVE maps to the frozen contract CANDIDATE (confirmation lives in
    # the ConfirmedEvent, not a CONFIRMED hypothesis state).
    assert to_violation_hypothesis(record).state.value == "candidate"


# --- repeated confirmation / separate episodes -------------------------------
def test_no_repeated_event_for_continuous_stop() -> None:
    _, events = _reason(_still_inside(30))  # long continuous stop
    assert len(events) == 1


def test_separate_episodes_yield_separate_events() -> None:
    # stop -> leave -> stop again: two distinct episodes, two events.
    ep1 = _still_inside(12)  # t 0..11 -> confirm at t=11
    leave = _track(
        [OUTSIDE, OUTSIDE],
        timestamps=[_BASE + timedelta(seconds=12), _BASE + timedelta(seconds=13)],
    )
    ep2 = _track(
        [INSIDE] * 12,
        timestamps=[_BASE + timedelta(seconds=14 + i) for i in range(12)],
    )
    reasoner, events = _reason(ep1 + leave + ep2)
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id
    assert events[0].start_at != events[1].start_at


# --- taint / gap -------------------------------------------------------------
def test_taint_restart_prevents_confirmation() -> None:
    # A long stationary-in-zone track split by a tainted block: neither clean
    # segment reaches the dwell threshold, so support never bridges the ID switch.
    track = _still_inside(14, tainted_indices=(6, 7))
    _, events = _reason(track)
    assert events == ()


def test_gap_within_tolerance_bridges() -> None:
    ts = [_BASE + timedelta(seconds=t) for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.5]]
    _, events = _reason(_track([INSIDE] * 12, timestamps=ts))
    assert len(events) == 1  # gap 1.5 <= 2.0 bridges; dwell 10.5 >= 10.0


def test_gap_exceeding_tolerance_breaks_run() -> None:
    ts = [_BASE + timedelta(seconds=t) for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]]
    _, events = _reason(_track([INSIDE] * 12, timestamps=ts))
    assert events == ()  # gap 3.0 > 2.0 ends the run; the fresh run never dwells


def test_no_gap_break_when_unset() -> None:
    ts = [_BASE + timedelta(seconds=t) for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]]
    _, events = _reason(_track([INSIDE] * 12, timestamps=ts), params=PARAMS_NO_GAP)
    assert len(events) == 1  # no tolerance -> the gap bridges; dwell 12.0 >= 10.0


# --- determinism / ordering / duplicates -------------------------------------
def test_out_of_order_matches_in_order() -> None:
    in_zone, stationary = _derivations(_still_inside(12))
    steps, restarts = join_stopped_in_zone(in_zone, stationary)
    ordered = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(
        steps, taint_restart_ids=restarts
    )
    shuffled = list(steps)
    random.Random(7).shuffle(shuffled)
    unordered = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(
        shuffled, taint_restart_ids=restarts
    )
    assert len(ordered) == len(unordered) == 1
    assert ordered[0].event_id == unordered[0].event_id


def test_duplicate_steps_deterministic() -> None:
    in_zone, stationary = _derivations(_still_inside(12))
    steps, restarts = join_stopped_in_zone(in_zone, stationary)
    plain = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(
        steps, taint_restart_ids=restarts
    )
    dup = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(
        steps + steps[-2:], taint_restart_ids=restarts
    )
    assert len(dup) == 1
    assert dup[0].event_id == plain[0].event_id


def test_replay_is_deterministic() -> None:
    track = _still_inside(12)
    e1 = _reason(track)[1]
    e2 = _reason(track)[1]
    assert e1[0].event_id == e2[0].event_id
    assert e1[0].model_dump_json() == e2[0].model_dump_json()


# --- multi-track / multi-zone isolation --------------------------------------
def test_multi_track_isolation() -> None:
    a = _still_inside(12, track_id="A")
    b_moving = _track([(1300.0 + (i % 2) * 100.0, 900.0) for i in range(12)], track_id="B")
    in_zone_a, stat_a = _derivations(a)
    in_zone_b, stat_b = _derivations(b_moving)
    steps_a, restarts_a = join_stopped_in_zone(in_zone_a, stat_a)
    steps_b, restarts_b = join_stopped_in_zone(in_zone_b, stat_b)
    events = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run(
        steps_a + steps_b, taint_restart_ids=restarts_a | restarts_b
    )
    assert len(events) == 1  # only track A stops
    assert events[0].track_ids == ("A",)


def test_overlapping_zone_independence() -> None:
    # Two overlapping no-stopping zones both cover the point; still one run/event.
    z1 = next(z for z in ZONES if z.zone_id == "zone-no-stop")
    z2 = Zone(
        zone_id="zone-no-stop-dup",
        zone_type=ZoneType.NO_STOPPING,
        enabled=True,
        polygon=NO_STOP_POLY,  # identical polygon -> overlapping
    )
    _, events = _reason(_still_inside(12), zones=(z1, z2))
    assert len(events) == 1


# --- edges / immutability / boundary audit -----------------------------------
def test_empty_input() -> None:
    reasoner = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    assert reasoner.run([]) == ()
    assert reasoner.events == ()


def test_input_immutability() -> None:
    track = _still_inside(12)
    before = copy.deepcopy(track)
    in_zone, stationary = _derivations(track)
    obs_before = (in_zone.observations, stationary.observations)
    IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH).run_join(
        in_zone, stationary
    )
    assert track == before  # the TrackState sequence is untouched
    assert (in_zone.observations, stationary.observations) == obs_before  # derivations untouched


def test_stopped_step_is_frozen() -> None:
    step = StoppedInZoneStep(
        observation=StationaryObservation(
            observation_id="s1",
            camera_id="cam",
            track_id="t",
            timestamp=_BASE,
            producer=_PROD,
            is_stationary=True,
        ),
        stopped_in_zone=True,
    )
    with pytest.raises(FrozenInstanceError):
        step.stopped_in_zone = False  # type: ignore[misc]


def test_no_backend_or_wallclock_imports() -> None:
    source = Path(illegal_stopping_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "torch",
        "transformers",
        "cv2",
        "numpy",
        "import time",
        "perf_counter",
        "datetime.now",
        "time.time",
        "uuid",
        "random",
        "persistence",
        "evidence_stub",
        "..detector",
        "..tracking",
        "..pipeline",
    )
    for token in forbidden:
        assert token not in source, f"unexpected reference to {token!r}"


def test_reasoner_module_geometry_free() -> None:
    # The reasoner consumes observation facts only: it must not import geometry
    # (no membership/stationarity recomputation happens in the rule layer).
    source = Path(illegal_stopping_module.__file__).read_text(encoding="utf-8")
    assert "geometry" not in source


# --- synthetic enter-stop-wait integration -----------------------------------
def test_synthetic_enter_stop_wait_confirms() -> None:
    # Enter the zone from outside, then hold position inside long enough.
    approach = [(1390.0, 640.0 + i * 20.0) for i in range(3)]  # moving toward the zone
    hold = [INSIDE] * 14
    _, events = _reason(_track(approach + hold))
    assert len(events) == 1


def test_synthetic_stop_too_short_does_not_confirm() -> None:
    approach = [(1390.0, 640.0 + i * 20.0) for i in range(3)]
    hold = [INSIDE] * 5  # far too short to reach the 10 s dwell
    _, events = _reason(_track(approach + hold))
    assert events == ()

"""Tests for stationary observation derivation (P2-U3).

Deterministic, model-free tests over the frozen ``StationaryObservation``
contract: zero-displacement/jitter/steady-motion/slow-drift stationarity of the
bbox bottom-center over a trailing pixel-space window; one-frame spike behaviour;
two-state minimum; tainted-step skip + restart marking; ordinary-gap, sparse,
duplicate, and non-monotonic timestamp robustness; ``motion_threshold`` recorded
but not applied; provenance/identity propagation; timezone-aware timestamp
preservation; serialization round-trip; determinism; input immutability; and
boundary/import-boundary audits. Uses synthetic TrackStates and the P1-U2 synth
scenarios only -- no zone, no model, no wall-clock, no backend.
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trafficpulse.contracts import (
    BoundingBox,
    ObjectClass,
    ObservationAdapter,
    Producer,
    ProducerKind,
    StationaryObservation,
    TrackState,
    TrackStatus,
)
from trafficpulse.observations import stationary as stationary_module
from trafficpulse.observations.stationary import (
    DEFAULT_STATIONARY_PRODUCER,
    STATIONARY_EPSILON_PX,
    STATIONARY_WINDOW,
    StationaryDerivation,
    derive_stationary_observations,
    derive_stationary_observations_with_taint,
)
from trafficpulse.synth import generate_enter_then_stop, generate_legal, generate_stationary

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_HALF_W = 20.0
_HEIGHT = 40.0


def _track_from_bottoms(
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
        bbox = BoundingBox(x1=x - _HALF_W, y1=y - _HEIGHT, x2=x + _HALF_W, y2=y)
        states.append(
            TrackState(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=ts,
                frame_index=i,
                object_class=ObjectClass.CAR,
                bbox=bbox,
                status=TrackStatus.ACTIVE,
                tainted=(i in tainted_indices),
            )
        )
    return states


def _still(
    x: float,
    y: float,
    n: int,
    *,
    camera_id: str = "cam-x",
    track_id: str = "trk-x",
    tainted_indices: tuple[int, ...] = (),
    interval_s: float = 1.0,
    timestamps: list[datetime] | None = None,
) -> list[TrackState]:
    return _track_from_bottoms(
        [(x, y)] * n,
        camera_id=camera_id,
        track_id=track_id,
        tainted_indices=tainted_indices,
        interval_s=interval_s,
        timestamps=timestamps,
    )


def _flags(obs: list[StationaryObservation]) -> list[bool]:
    return [o.is_stationary for o in obs]


# --- core stationarity vs motion ---------------------------------------------
def test_never_moving_track_all_stationary() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 6))
    assert obs
    assert all(o.is_stationary for o in obs)


def test_small_jitter_below_epsilon_is_stationary() -> None:
    # In-place oscillation within +/-0.5 px: net window displacement <= 1 px < 2.
    bottoms = [(500.0, 500.0), (500.5, 500.0), (499.5, 500.0), (500.5, 500.0), (499.5, 500.0)]
    obs = derive_stationary_observations(_track_from_bottoms(bottoms))
    assert all(o.is_stationary for o in obs)


def test_steady_motion_above_epsilon_is_moving() -> None:
    # 10 px/step is far above the 2 px epsilon; every emitted step is moving.
    bottoms = [(500.0 + 10.0 * i, 500.0) for i in range(6)]
    obs = derive_stationary_observations(_track_from_bottoms(bottoms))
    assert obs
    assert all(not o.is_stationary for o in obs)


def test_slow_continuous_drift_becomes_moving_once_window_accumulates() -> None:
    # 1 px/step: net stays <= epsilon for the first samples, then the trailing
    # window accumulates past epsilon and the step reads moving.
    bottoms = [(500.0 + 1.0 * i, 500.0) for i in range(10)]
    obs = derive_stationary_observations(_track_from_bottoms(bottoms))
    assert _flags(obs) == [True, True, False, False, False, False, False, False, False]


def test_net_displacement_at_epsilon_is_stationary() -> None:
    # Boundary: net window displacement exactly == epsilon reads as stationary.
    obs = derive_stationary_observations(
        _track_from_bottoms([(500.0, 500.0), (500.0 + STATIONARY_EPSILON_PX, 500.0)])
    )
    assert len(obs) == 1
    assert obs[0].is_stationary


def test_one_frame_spike_is_isolated_and_recovers() -> None:
    # A single frame jumps well past epsilon then returns: exactly the spike
    # frame reads moving; the following frame recovers to stationary.
    bottoms = [(500.0, 500.0)] * 4 + [(530.0, 500.0), (500.0, 500.0)]
    obs = derive_stationary_observations(_track_from_bottoms(bottoms))
    assert _flags(obs) == [True, True, True, False, True]


# --- emission semantics / two-state minimum ----------------------------------
def test_two_identical_states_emit_one_stationary() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))
    assert len(obs) == 1 and obs[0].is_stationary


def test_single_state_yields_nothing() -> None:
    assert derive_stationary_observations(_still(500.0, 500.0, 1)) == []


def test_empty_track_yields_nothing() -> None:
    assert derive_stationary_observations([]) == []


def test_observation_emitted_for_current_state_only() -> None:
    # N states -> N-1 observations; the first state never emits.
    track = _still(500.0, 500.0, 4)
    obs = derive_stationary_observations(track)
    assert [o.timestamp for o in obs] == [s.timestamp for s in track[1:]]


# --- gaps / sparse / duplicate / non-monotonic timestamps --------------------
def test_large_temporal_gap_is_bridged() -> None:
    # A huge time gap between two stationary samples is bridged: the decision is
    # sample/pixel based, so it still reads stationary regardless of the gap.
    ts = [_BASE, _BASE + timedelta(hours=3)]
    obs = derive_stationary_observations(
        _track_from_bottoms([(500.0, 500.0), (500.0, 500.0)], timestamps=ts)
    )
    assert len(obs) == 1 and obs[0].is_stationary


def test_sparse_sampling_stationary_and_moving() -> None:
    ts = [_BASE + timedelta(minutes=5 * i) for i in range(4)]
    still = derive_stationary_observations(
        _track_from_bottoms([(500.0, 500.0)] * 4, timestamps=ts)
    )
    assert all(o.is_stationary for o in still)
    moving = derive_stationary_observations(
        _track_from_bottoms([(500.0 + 20.0 * i, 500.0) for i in range(4)], timestamps=ts)
    )
    assert all(not o.is_stationary for o in moving)


def test_duplicate_timestamps_do_not_change_decision() -> None:
    # Two consecutive samples share a timestamp: the pixel decision is unaffected
    # (position-driven); timestamps flow through only as recorded metadata.
    ts = [_BASE, _BASE + timedelta(seconds=1), _BASE + timedelta(seconds=1)]
    obs = derive_stationary_observations(_track_from_bottoms([(500.0, 500.0)] * 3, timestamps=ts))
    assert len(obs) == 2
    assert all(o.is_stationary for o in obs)
    assert [o.timestamp for o in obs] == ts[1:]


def test_non_monotonic_timestamps_do_not_change_decision() -> None:
    # A backwards timestamp does not affect the geometric stationarity decision;
    # the recorded timestamps are preserved verbatim (input order, not sorted).
    ts = [_BASE + timedelta(seconds=5), _BASE, _BASE + timedelta(seconds=2)]
    obs = derive_stationary_observations(_track_from_bottoms([(500.0, 500.0)] * 3, timestamps=ts))
    assert all(o.is_stationary for o in obs)
    assert [o.timestamp for o in obs] == ts[1:]


# --- taint handling ----------------------------------------------------------
def test_tainted_steps_skipped_and_restart_marked() -> None:
    # 5 states; index 2 tainted -> emit for states 1 and 4; state 4 is a restart.
    track = _still(500.0, 500.0, 5, tainted_indices=(2,))
    result = derive_stationary_observations_with_taint(track)
    assert [o.timestamp for o in result.observations] == [track[1].timestamp, track[4].timestamp]
    restart = [o for o in result.observations if o.observation_id in result.taint_restart_ids]
    assert len(restart) == 1 and restart[0].timestamp == track[4].timestamp


def test_first_clean_step_after_leading_taint_is_restart() -> None:
    track = _still(500.0, 500.0, 3, tainted_indices=(0,))
    result = derive_stationary_observations_with_taint(track)
    assert len(result.observations) == 1
    assert result.observations[0].observation_id in result.taint_restart_ids


def test_taint_resets_the_window() -> None:
    # Moving samples, then a taint, then a fresh clean run: the post-taint run's
    # window must not include pre-taint samples. Two identical post-taint samples
    # net to zero -> stationary, proving the window reset (no bridging).
    bottoms = [(500.0, 500.0), (600.0, 500.0), (700.0, 500.0), (800.0, 500.0), (800.0, 500.0)]
    track = _track_from_bottoms(bottoms, tainted_indices=(2,))
    result = derive_stationary_observations_with_taint(track)
    # Clean runs: [0,1] (moving) and [3,4] (identical -> stationary, restart).
    assert [o.is_stationary for o in result.observations] == [False, True]
    assert result.observations[1].observation_id in result.taint_restart_ids


def test_all_tainted_yields_nothing() -> None:
    track = _still(500.0, 500.0, 4, tainted_indices=(0, 1, 2, 3))
    result = derive_stationary_observations_with_taint(track)
    assert result.observations == ()
    assert result.taint_restart_ids == frozenset()


def test_clean_track_has_no_restarts() -> None:
    result = derive_stationary_observations_with_taint(_still(500.0, 500.0, 4))
    assert result.taint_restart_ids == frozenset()
    assert len(result.observations) == 3


# --- motion_threshold: recorded, never applied -------------------------------
def test_motion_threshold_does_not_change_decision() -> None:
    track = _track_from_bottoms([(500.0 + 1.0 * i, 500.0) for i in range(10)])
    a = derive_stationary_observations_with_taint(track, motion_threshold=0.0)
    b = derive_stationary_observations_with_taint(track, motion_threshold=1e9)
    assert a.observations == b.observations  # decision identical
    assert a.recorded_motion_threshold == 0.0  # but the value is recorded inertly
    assert b.recorded_motion_threshold == 1e9


def test_motion_threshold_default_none_and_ignored_by_list_helper() -> None:
    track = _still(500.0, 500.0, 4)
    with_threshold = derive_stationary_observations(track, motion_threshold=0.5)
    assert with_threshold == derive_stationary_observations(track)
    assert derive_stationary_observations_with_taint(track).recorded_motion_threshold is None


# --- parameter validation ----------------------------------------------------
def test_window_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        derive_stationary_observations(_still(500.0, 500.0, 4), window=1)


def test_negative_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        derive_stationary_observations(_still(500.0, 500.0, 4), epsilon_px=-1.0)


def test_window_and_epsilon_are_configurable() -> None:
    # A tiny epsilon makes even sub-pixel drift read as moving; a large one makes
    # clear motion read as stationary -- proving both are honoured.
    drift = _track_from_bottoms([(500.0 + 1.0 * i, 500.0) for i in range(5)])
    tight = derive_stationary_observations(drift, epsilon_px=0.5, window=2)
    loose = derive_stationary_observations(drift, epsilon_px=100.0)
    assert all(not o.is_stationary for o in tight)
    assert all(o.is_stationary for o in loose)


# --- provenance / identity ---------------------------------------------------
def test_contract_fields_are_stationarity_only() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))[0]
    assert obs.obs_type == "stationary"
    assert obs.is_stationary is True
    assert obs.speed_estimate is None  # no calibrated speed claimed
    assert obs.dwell_seconds is None  # dwell is a reasoning-layer accumulation


def test_track_and_camera_identity_preserved() -> None:
    track = _still(500.0, 500.0, 3, camera_id="cam-42", track_id="trk-7")
    obs = derive_stationary_observations(track)
    assert all(o.camera_id == "cam-42" for o in obs)
    assert all(o.track_id == "trk-7" for o in obs)


def test_timestamps_are_timezone_aware_and_from_trackstate() -> None:
    track = _still(500.0, 500.0, 4)
    obs = derive_stationary_observations(track)
    assert [o.timestamp for o in obs] == [s.timestamp for s in track[1:]]
    assert all(o.timestamp.tzinfo is not None for o in obs)


def test_default_producer_is_heuristic() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))
    assert obs[0].producer == DEFAULT_STATIONARY_PRODUCER
    assert obs[0].producer.kind is ProducerKind.HEURISTIC
    assert obs[0].producer.name == "stationary"


def test_custom_producer_is_used() -> None:
    prod = Producer(name="custom", version="9.9", kind=ProducerKind.HEURISTIC)
    obs = derive_stationary_observations(_still(500.0, 500.0, 2), producer=prod)
    assert all(o.producer == prod for o in obs)


def test_observation_ids_deterministic_and_unique() -> None:
    track = _track_from_bottoms([(500.0 + i, 500.0) for i in range(5)])
    ids_a = [o.observation_id for o in derive_stationary_observations(track)]
    ids_b = [o.observation_id for o in derive_stationary_observations(track)]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)
    assert all(i.startswith("sta-") for i in ids_a)


# --- serialization -----------------------------------------------------------
def test_serialization_round_trip() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))[0]
    assert ObservationAdapter.validate_python(obs.model_dump()) == obs
    assert ObservationAdapter.validate_json(obs.model_dump_json()) == obs


# --- determinism / immutability ----------------------------------------------
def test_repeated_calls_are_identical() -> None:
    track = _track_from_bottoms([(500.0 + i, 500.0) for i in range(6)], tainted_indices=(3,))
    a = derive_stationary_observations_with_taint(track)
    b = derive_stationary_observations_with_taint(track)
    assert a == b


def test_inputs_not_mutated() -> None:
    track = _still(500.0, 500.0, 4, tainted_indices=(1,))
    before = [s.model_dump() for s in track]
    derive_stationary_observations_with_taint(track)
    assert [s.model_dump() for s in track] == before


def test_result_is_frozen_dataclass() -> None:
    result = derive_stationary_observations_with_taint(_still(500.0, 500.0, 2))
    assert isinstance(result, StationaryDerivation)
    with pytest.raises(FrozenInstanceError):
        result.observations = ()  # type: ignore[misc]


# --- no zone / no model-provenance coupling ----------------------------------
def test_no_zone_dependency() -> None:
    # Stationarity is derived without any scene/zone input; the observation
    # carries no zone fields.
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))[0]
    assert not hasattr(obs, "zone_id")
    assert not hasattr(obs, "is_inside")


def test_no_dependency_on_model_provenance() -> None:
    obs = derive_stationary_observations(_still(500.0, 500.0, 2))[0]
    assert not hasattr(obs, "models")
    assert isinstance(obs.producer, Producer)


# --- P2-U2 compatibility -----------------------------------------------------
def test_result_shape_mirrors_taint_derivation_pattern() -> None:
    # Same taint-restart contract as HeadingDerivation / InZoneDerivation.
    result = derive_stationary_observations_with_taint(_still(500.0, 500.0, 3))
    assert isinstance(result.observations, tuple)
    assert isinstance(result.taint_restart_ids, frozenset)


# --- synthetic + scenario integration ----------------------------------------
def test_synthetic_stationary_trajectory_all_stationary() -> None:
    obs = derive_stationary_observations(generate_stationary(frame_count=20))
    assert obs
    assert all(o.is_stationary for o in obs)


def test_synthetic_moving_trajectory_all_moving() -> None:
    # generate_legal moves 12 px/frame -- clearly above epsilon at 30 fps.
    obs = derive_stationary_observations(generate_legal(frame_count=20))
    assert obs
    assert all(not o.is_stationary for o in obs)


def test_stop_then_go_trajectory_is_stationary_only_while_stopped() -> None:
    # enter_then_stop: 10 moving frames, then 20 held frames. Deep inside the
    # stopped segment must read stationary; deep inside motion must read moving.
    track = generate_enter_then_stop(moving_frames=10, stopped_frames=20)
    obs = derive_stationary_observations(track)
    assert not obs[2].is_stationary  # well inside the moving leg
    assert obs[-1].is_stationary  # well inside the stopped leg
    assert any(o.is_stationary for o in obs) and any(not o.is_stationary for o in obs)


# --- source / import boundary audit ------------------------------------------
def test_module_imports_no_backend_reasoning_or_wall_clock() -> None:
    source = Path(stationary_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "torch", "transformers", "cv2", "import av", "numpy", "sqlite",
        "requests", "urllib", "socket",
        "..rules", "..persistence", "..detector", "..tracking", "..pipeline",
        "import random", "import time", "from datetime", "datetime.now",
        "perf_counter", "time.time",
    )
    for token in forbidden:
        assert token not in source, f"unexpected token in stationary.py: {token!r}"


def test_window_and_epsilon_defaults_are_provisional_constants() -> None:
    assert STATIONARY_WINDOW >= 2
    assert STATIONARY_EPSILON_PX >= 0.0

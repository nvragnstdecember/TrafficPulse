"""Tests for the scene-level signal-state derivation (P3-U3).

Deterministic, model-free tests over the frozen ``SignalStateObservation``
contract: sampling a declared signal schedule at query timestamps, scene-level
identity (``track_id=None``), inclusive phase-start resolution, the honest
``UNKNOWN`` before the schedule starts, equal-start override, roi/producer
provenance, and byte-identical replay. No detector, classifier, tracker, or video.
"""

from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import Producer, SignalStateObservation
from trafficpulse.contracts.enums import ProducerKind, SignalState
from trafficpulse.observations.signal import (
    DEFAULT_SIGNAL_PRODUCER,
    SignalPhase,
    derive_signal_state_observations,
    signal_state_at,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
CAMERA = "cam-1"


def _ts(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


# A red -> green -> amber -> red schedule.
SCHEDULE = (
    SignalPhase(start=_ts(0.0), state=SignalState.RED),
    SignalPhase(start=_ts(10.0), state=SignalState.GREEN),
    SignalPhase(start=_ts(13.0), state=SignalState.AMBER),
    SignalPhase(start=_ts(15.0), state=SignalState.RED),
)


# --- signal_state_at resolution ----------------------------------------------
def test_state_before_schedule_is_unknown() -> None:
    assert signal_state_at(SCHEDULE, _ts(-1.0)) is SignalState.UNKNOWN


def test_empty_schedule_is_unknown() -> None:
    assert signal_state_at((), _ts(5.0)) is SignalState.UNKNOWN


def test_phase_start_is_inclusive() -> None:
    assert signal_state_at(SCHEDULE, _ts(10.0)) is SignalState.GREEN  # exactly at a start


def test_state_holds_until_next_phase() -> None:
    assert signal_state_at(SCHEDULE, _ts(0.0)) is SignalState.RED
    assert signal_state_at(SCHEDULE, _ts(9.999)) is SignalState.RED
    assert signal_state_at(SCHEDULE, _ts(14.0)) is SignalState.AMBER
    assert signal_state_at(SCHEDULE, _ts(999.0)) is SignalState.RED  # last phase persists


def test_unordered_schedule_resolves_deterministically() -> None:
    shuffled = (SCHEDULE[2], SCHEDULE[0], SCHEDULE[3], SCHEDULE[1])
    for offset in (0.0, 9.9, 10.0, 13.5, 20.0):
        assert signal_state_at(shuffled, _ts(offset)) is signal_state_at(SCHEDULE, _ts(offset))


def test_equal_start_later_declaration_wins() -> None:
    schedule = (
        SignalPhase(start=_ts(0.0), state=SignalState.RED),
        SignalPhase(start=_ts(0.0), state=SignalState.GREEN),  # overrides at the same instant
    )
    assert signal_state_at(schedule, _ts(0.0)) is SignalState.GREEN


# --- derivation --------------------------------------------------------------
def test_emits_one_observation_per_timestamp_in_order() -> None:
    timestamps = [_ts(t) for t in (0.0, 11.0, 14.0, 16.0)]
    obs = derive_signal_state_observations(SCHEDULE, timestamps=timestamps, camera_id=CAMERA)
    assert [o.timestamp for o in obs] == timestamps
    assert [o.signal_state for o in obs] == [
        SignalState.RED,
        SignalState.GREEN,
        SignalState.AMBER,
        SignalState.RED,
    ]


def test_observations_are_scene_level() -> None:
    obs = derive_signal_state_observations(SCHEDULE, timestamps=[_ts(0.0)], camera_id=CAMERA)
    assert all(isinstance(o, SignalStateObservation) for o in obs)
    assert obs[0].track_id is None  # scene-level: not track-bound
    assert obs[0].camera_id == CAMERA


def test_before_schedule_emits_unknown() -> None:
    obs = derive_signal_state_observations(SCHEDULE, timestamps=[_ts(-5.0)], camera_id=CAMERA)
    assert obs[0].signal_state is SignalState.UNKNOWN


def test_empty_timestamps_emits_nothing() -> None:
    assert derive_signal_state_observations(SCHEDULE, timestamps=[], camera_id=CAMERA) == ()


def test_roi_id_recorded_when_supplied() -> None:
    obs = derive_signal_state_observations(
        SCHEDULE, timestamps=[_ts(0.0)], camera_id=CAMERA, roi_id="roi-sg-001"
    )
    assert obs[0].roi_id == "roi-sg-001"


def test_default_producer_is_declared_heuristic_not_a_model() -> None:
    obs = derive_signal_state_observations(SCHEDULE, timestamps=[_ts(0.0)], camera_id=CAMERA)
    assert obs[0].producer == DEFAULT_SIGNAL_PRODUCER
    assert obs[0].producer.kind is ProducerKind.HEURISTIC  # a declared log, not a classifier


def test_custom_producer_is_honoured() -> None:
    prod = Producer(name="manual-annotation", version="1", kind=ProducerKind.HEURISTIC)
    obs = derive_signal_state_observations(
        SCHEDULE, timestamps=[_ts(0.0)], camera_id=CAMERA, producer=prod
    )
    assert obs[0].producer == prod


# --- determinism -------------------------------------------------------------
def test_replay_is_byte_identical() -> None:
    timestamps = [_ts(t) for t in (0.0, 11.0, 14.0)]
    a = derive_signal_state_observations(SCHEDULE, timestamps=timestamps, camera_id=CAMERA)
    b = derive_signal_state_observations(SCHEDULE, timestamps=timestamps, camera_id=CAMERA)
    assert [o.observation_id for o in a] == [o.observation_id for o in b]
    assert [o.model_dump_json() for o in a] == [o.model_dump_json() for o in b]


def test_distinct_state_yields_distinct_id() -> None:
    # Same (camera, timestamp) but a different resolved state -> different id
    # (content-derived), so replays with different declared logs stay distinct.
    red = derive_signal_state_observations(
        (SignalPhase(start=_ts(0.0), state=SignalState.RED),),
        timestamps=[_ts(0.0)],
        camera_id=CAMERA,
    )[0]
    green = derive_signal_state_observations(
        (SignalPhase(start=_ts(0.0), state=SignalState.GREEN),),
        timestamps=[_ts(0.0)],
        camera_id=CAMERA,
    )[0]
    assert red.observation_id != green.observation_id

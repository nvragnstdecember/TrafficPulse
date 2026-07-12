"""Tests for the generalized observation join (P3-U3).

Deterministic, model-free tests of :func:`trafficpulse.rules.joins.join_streams`:
context x per-track pairing on ``(camera, timestamp)`` / ``(camera, track,
timestamp)``, the conservative missing-side fold (never fabricate evidence), the
taint-restart union, carrier-order determinism, and -- the acceptance bar --
byte-exact reproduction of ``rules.illegal_stopping.join_stopped_in_zone`` on
hand-built illegal-stopping fixtures. Observations are constructed directly (no
tracks/geometry), wrapped in the real derivation dataclasses.
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import (
    InZoneObservation,
    Producer,
    SignalStateObservation,
    StationaryObservation,
)
from trafficpulse.contracts.enums import ProducerKind, SignalState, ZoneKind
from trafficpulse.observations.stationary import StationaryDerivation
from trafficpulse.observations.zones import InZoneDerivation
from trafficpulse.rules.illegal_stopping import StoppedInZoneStep, join_stopped_in_zone
from trafficpulse.rules.joins import JoinedStep, join_streams

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PROD = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)
CAMERA = "cam-1"
TRACK = "t1"


def _ts(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def _sta(i: int, *, stationary: bool = True, track: str = TRACK) -> StationaryObservation:
    return StationaryObservation(
        observation_id=f"sta-{track}-{i}",
        camera_id=CAMERA,
        track_id=track,
        timestamp=_ts(i),
        producer=PROD,
        is_stationary=stationary,
    )


def _inz(
    i: int, *, inside: bool, kind: ZoneKind = ZoneKind.NO_STOPPING, zone: str = "z1",
    track: str = TRACK,
) -> InZoneObservation:
    return InZoneObservation(
        observation_id=f"inz-{track}-{zone}-{i}",
        camera_id=CAMERA,
        track_id=track,
        timestamp=_ts(i),
        producer=PROD,
        zone_id=zone,
        zone_kind=kind,
        is_inside=inside,
    )


def _sig(i: int, state: SignalState) -> SignalStateObservation:
    return SignalStateObservation(
        observation_id=f"sig-{i}",
        camera_id=CAMERA,
        track_id=None,
        timestamp=_ts(i),
        producer=PROD,
        signal_state=state,
        roi_id="roi-1",
    )


def _sta_stream(
    *obs: StationaryObservation, restarts: frozenset[str] = frozenset()
) -> StationaryDerivation:
    return StationaryDerivation(obs, restarts)


def _inz_stream(
    *obs: InZoneObservation, restarts: frozenset[str] = frozenset()
) -> InZoneDerivation:
    return InZoneDerivation(obs, restarts)


# --- basic pairing -----------------------------------------------------------
def test_carrier_paired_with_track_facts_and_context() -> None:
    carrier = _sta_stream(_sta(0), _sta(1))
    inzone = _inz_stream(_inz(0, inside=True), _inz(1, inside=False))
    context = [_sig(0, SignalState.RED), _sig(1, SignalState.GREEN)]
    result = join_streams(carrier, track_streams=[inzone], context=context)

    assert [js.carrier.observation_id for js in result.steps] == ["sta-t1-0", "sta-t1-1"]
    assert result.steps[0].track_facts == (_inz(0, inside=True),)
    assert result.steps[0].context == _sig(0, SignalState.RED)
    assert result.steps[1].context == _sig(1, SignalState.GREEN)


def test_context_pairs_to_every_carrier_at_same_timestamp() -> None:
    # Two tracks at the same timestamp both pair to the one scene-level context.
    carrier = _sta_stream(_sta(0, track="t1"), _sta(0, track="t2"))
    context = [_sig(0, SignalState.RED)]
    result = join_streams(carrier, context=context)
    assert all(js.context == _sig(0, SignalState.RED) for js in result.steps)


# --- conservative missing-side fold ------------------------------------------
def test_missing_context_folds_to_none() -> None:
    carrier = _sta_stream(_sta(0), _sta(5))
    context = [_sig(0, SignalState.RED)]  # no context at t=5
    result = join_streams(carrier, context=context)
    assert result.steps[0].context == _sig(0, SignalState.RED)
    assert result.steps[1].context is None  # never fabricated


def test_missing_track_fact_folds_to_empty() -> None:
    carrier = _sta_stream(_sta(0), _sta(1))
    inzone = _inz_stream(_inz(0, inside=True))  # no in-zone fact at t=1
    result = join_streams(carrier, track_streams=[inzone])
    assert result.steps[0].track_facts == (_inz(0, inside=True),)
    assert result.steps[1].track_facts == ()  # never fabricated


def test_track_fact_or_context_without_carrier_produces_no_step() -> None:
    carrier = _sta_stream(_sta(0))
    inzone = _inz_stream(_inz(0, inside=True), _inz(9, inside=True))  # t=9 has no carrier
    context = [_sig(0, SignalState.RED), _sig(9, SignalState.RED)]  # t=9 has no carrier
    result = join_streams(carrier, track_streams=[inzone], context=context)
    assert len(result.steps) == 1  # only the carrier's timestamp yields a step
    assert result.steps[0].carrier.timestamp == _ts(0)


def test_multiple_track_facts_at_one_key_are_all_collected() -> None:
    carrier = _sta_stream(_sta(0))
    inzone = _inz_stream(_inz(0, inside=True, zone="z1"), _inz(0, inside=False, zone="z2"))
    result = join_streams(carrier, track_streams=[inzone])
    assert len(result.steps[0].track_facts) == 2


# --- taint-restart union -----------------------------------------------------
def test_carrier_taint_restart_is_carried() -> None:
    carrier = _sta_stream(_sta(0), _sta(1), restarts=frozenset({"sta-t1-1"}))
    result = join_streams(carrier)
    assert result.taint_restart_ids == frozenset({"sta-t1-1"})


def test_track_stream_taint_restart_unions_onto_carrier() -> None:
    carrier = _sta_stream(_sta(0), _sta(1))
    inzone = _inz_stream(
        _inz(0, inside=True), _inz(1, inside=True), restarts=frozenset({"inz-t1-z1-1"})
    )
    result = join_streams(carrier, track_streams=[inzone])
    # The in-zone restart at t=1 maps onto the carrier id at the same key.
    assert result.taint_restart_ids == frozenset({"sta-t1-1"})


def test_context_carries_no_taint() -> None:
    carrier = _sta_stream(_sta(0))
    result = join_streams(carrier, context=[_sig(0, SignalState.RED)])
    assert result.taint_restart_ids == frozenset()


# --- determinism -------------------------------------------------------------
def test_steps_preserve_carrier_input_order() -> None:
    carrier = _sta_stream(_sta(2), _sta(0), _sta(1))  # deliberately unordered
    result = join_streams(carrier)
    assert [js.carrier.timestamp for js in result.steps] == [_ts(2), _ts(0), _ts(1)]


def test_join_is_a_pure_function() -> None:
    carrier = _sta_stream(_sta(0), _sta(1))
    inzone = _inz_stream(_inz(0, inside=True), _inz(1, inside=False))
    context = [_sig(0, SignalState.RED)]
    a = join_streams(carrier, track_streams=[inzone], context=context)
    b = join_streams(carrier, track_streams=[inzone], context=context)
    assert a == b


def test_joined_step_is_immutable() -> None:
    js: JoinedStep[StationaryObservation] = join_streams(_sta_stream(_sta(0))).steps[0]
    with pytest.raises(FrozenInstanceError):
        js.context = _sig(0, SignalState.RED)  # type: ignore[misc]


# --- equivalence to join_stopped_in_zone (acceptance bar) --------------------
def _fold_stopped(js: JoinedStep[StationaryObservation]) -> bool:
    """The illegal-stopping fold, applied caller-side to the generic join result."""

    return js.carrier.is_stationary and any(
        o.is_inside for o in js.track_facts
        if isinstance(o, InZoneObservation) and o.zone_kind is ZoneKind.NO_STOPPING
    )


def _assert_equivalent(inzone: InZoneDerivation, stationary: StationaryDerivation) -> None:
    expected_steps, expected_restarts = join_stopped_in_zone(inzone, stationary)
    result = join_streams(stationary, track_streams=[inzone])
    got_steps = [
        StoppedInZoneStep(observation=js.carrier, stopped_in_zone=_fold_stopped(js))
        for js in result.steps
    ]
    assert got_steps == expected_steps
    assert result.taint_restart_ids == expected_restarts


def test_generalized_join_reproduces_join_stopped_in_zone_mixed() -> None:
    # stationary+inside, stationary+outside, moving+inside, moving+outside.
    stationary = _sta_stream(
        _sta(0, stationary=True), _sta(1, stationary=True),
        _sta(2, stationary=False), _sta(3, stationary=False),
    )
    inzone = _inz_stream(
        _inz(0, inside=True), _inz(1, inside=False),
        _inz(2, inside=True), _inz(3, inside=False),
    )
    _assert_equivalent(inzone, stationary)


def test_generalized_join_reproduces_join_stopped_in_zone_missing_inzone() -> None:
    # A carrier with no in-zone fact at its key must fold to not-stopped in both.
    stationary = _sta_stream(_sta(0, stationary=True), _sta(1, stationary=True))
    inzone = _inz_stream(_inz(0, inside=True))  # nothing at t=1
    _assert_equivalent(inzone, stationary)


def test_generalized_join_reproduces_join_stopped_in_zone_taint() -> None:
    stationary = _sta_stream(
        _sta(0, stationary=True), _sta(1, stationary=True),
        restarts=frozenset({"sta-t1-1"}),
    )
    inzone = _inz_stream(
        _inz(0, inside=True), _inz(1, inside=True), restarts=frozenset({"inz-t1-z1-1"})
    )
    _assert_equivalent(inzone, stationary)


def test_generalized_join_reproduces_join_stopped_in_zone_non_no_stopping_ignored() -> None:
    # A non-no-stopping in-zone fact must not count toward stopped-in-zone in either.
    stationary = _sta_stream(_sta(0, stationary=True))
    inzone = _inz_stream(_inz(0, inside=True, kind=ZoneKind.JUNCTION_CONFLICT))
    _assert_equivalent(inzone, stationary)

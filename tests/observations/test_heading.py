"""Tests for heading-vs-lane observation derivation (P1-U4, concern 1).

Deterministic, model-free tests: angular deviation for legal/opposite/
perpendicular/diagonal movement, zero-displacement and insufficient-length
handling, timestamp/identity provenance, legal-direction consumption and scaling
invariance, tainted-data abstention, threshold-comparison semantics, and
immutability. Uses synthetic TrackStates only.
"""

import pytest

from trafficpulse.contracts.scene import DirectionVector
from trafficpulse.observations.heading import derive_heading_observations
from trafficpulse.synth import (
    build_track,
    generate_diagonal,
    generate_legal,
    generate_stationary,
    generate_track,
    generate_wrong_way,
    linear_positions,
)

UP = DirectionVector(dx=0.0, dy=-1.0)  # legal "north": decreasing y


def _derive(track, *, legal=UP, deviation_max=120.0, lane="lane"):  # type: ignore[no-untyped-def]
    return derive_heading_observations(
        track, legal_direction=legal, lane_id=lane, deviation_max_degrees=deviation_max
    )


# --- angular deviation -------------------------------------------------------
def test_legal_movement_low_deviation() -> None:
    obs = _derive(generate_legal())
    assert obs
    assert all(o.deviation_degrees == pytest.approx(0.0) for o in obs)
    assert all(not o.is_contradiction for o in obs)


def test_opposite_movement_180() -> None:
    obs = _derive(generate_wrong_way())
    assert all(o.deviation_degrees == pytest.approx(180.0) for o in obs)
    assert all(o.is_contradiction for o in obs)


def test_perpendicular_movement_90() -> None:
    track = generate_track(
        start=(500.0, 500.0), direction=(1.0, 0.0), step_size=10.0, frame_count=5
    )
    obs = _derive(track)
    assert all(o.deviation_degrees == pytest.approx(90.0) for o in obs)


def test_diagonal_movement_45() -> None:
    track = generate_track(
        start=(500.0, 900.0), direction=(1.0, -1.0), step_size=10.0, frame_count=5
    )
    obs = _derive(track)
    assert all(o.deviation_degrees == pytest.approx(45.0) for o in obs)


# --- edge behavior -----------------------------------------------------------
def test_zero_displacement_skipped() -> None:
    assert _derive(generate_stationary()) == []


def test_insufficient_length_yields_nothing() -> None:
    assert _derive(generate_track(frame_count=1)) == []
    assert _derive([]) == []


def test_tainted_positions_yield_no_observations() -> None:
    positions = linear_positions((960.0, 700.0), (0.0, 1.0), 12.0, 6)  # wrong-way motion
    track = build_track(positions, track_id="tainted", tainted=True)
    assert _derive(track) == []


# --- provenance --------------------------------------------------------------
def test_timestamps_come_from_trackstate() -> None:
    track = generate_wrong_way(frame_count=5)
    obs = _derive(track)
    assert [o.timestamp for o in obs] == [ts.timestamp for ts in track[1:]]


def test_track_and_camera_identity_preserved() -> None:
    track = generate_wrong_way(frame_count=5)
    obs = _derive(track)
    assert all(o.track_id == "wrong-way-track" for o in obs)
    assert all(o.camera_id == track[0].camera_id for o in obs)
    assert all(o.lane_id == "lane" for o in obs)


def test_observation_ids_deterministic_and_unique() -> None:
    track = generate_wrong_way(frame_count=6)
    ids_a = [o.observation_id for o in _derive(track)]
    ids_b = [o.observation_id for o in _derive(track)]
    assert ids_a == ids_b  # deterministic
    assert len(set(ids_a)) == len(ids_a)  # unique per step


# --- configured legal direction ---------------------------------------------
def test_legal_direction_scaling_invariant() -> None:
    track = generate_wrong_way(frame_count=6)
    a = _derive(track, legal=DirectionVector(dx=0.0, dy=-1.0))
    b = _derive(track, legal=DirectionVector(dx=0.0, dy=-5.0))
    assert [o.deviation_degrees for o in a] == [o.deviation_degrees for o in b]


def test_is_contradiction_uses_strict_threshold() -> None:
    # Perpendicular movement: deviation is exactly 90 degrees.
    track = generate_track(
        start=(500.0, 500.0), direction=(1.0, 0.0), step_size=10.0, frame_count=4
    )
    assert all(not o.is_contradiction for o in _derive(track, deviation_max=90.0))
    assert all(o.is_contradiction for o in _derive(track, deviation_max=89.9))


# --- invariants / immutability ----------------------------------------------
def test_deviation_within_zero_to_180() -> None:
    for track in (generate_legal(), generate_wrong_way(), generate_diagonal()):
        assert all(0.0 <= o.deviation_degrees <= 180.0 for o in _derive(track))


def test_heading_within_zero_to_360() -> None:
    obs = _derive(generate_wrong_way())
    assert all(0.0 <= o.heading_degrees <= 360.0 for o in obs)
    assert all(o.legal_heading_degrees == pytest.approx(270.0) for o in obs)  # up = 270 deg


def test_inputs_not_mutated() -> None:
    track = generate_wrong_way(frame_count=5)
    before = [ts.model_dump() for ts in track]
    legal = DirectionVector(dx=0.0, dy=-1.0)
    _derive(track, legal=legal)
    assert [ts.model_dump() for ts in track] == before
    assert legal == DirectionVector(dx=0.0, dy=-1.0)

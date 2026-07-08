"""Unit tests for the synthetic trajectory core (P1-U2).

Deterministic, model-free tests of position builders and ``TrackState``
assembly: reproducibility, seed behavior, timestamp/frame ordering, movement and
stationary correctness, bounded jitter, dropped-frame behavior, ``TrackState``
and geometry compatibility, absence of mutation, and error handling.
"""

from datetime import timedelta

import pytest

from trafficpulse.contracts import BoundingBox, ObjectClass, TrackState, TrackStatus
from trafficpulse.geometry import (
    ZeroVectorError,
    angle_between_degrees,
    direction,
    point_in_polygon,
)
from trafficpulse.synth.trajectories import (
    DEFAULT_START_TIME,
    build_track,
    curved_positions,
    generate_track,
    linear_positions,
    segmented_positions,
)


def _center(ts: TrackState) -> tuple[float, float]:
    b = ts.bbox
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


# --- reproducibility / determinism -------------------------------------------
def test_reproducible_same_params() -> None:
    assert generate_track(frame_count=20) == generate_track(frame_count=20)


def test_reproducible_noisy_same_seed() -> None:
    a = generate_track(frame_count=20, jitter_sigma=1.5, seed=42)
    b = generate_track(frame_count=20, jitter_sigma=1.5, seed=42)
    assert a == b


def test_repeated_execution_stable() -> None:
    results = [generate_track(frame_count=15, jitter_sigma=1.0, seed=7) for _ in range(4)]
    assert all(r == results[0] for r in results)


# --- seed behavior -----------------------------------------------------------
def test_different_seed_different_trajectory() -> None:
    a = generate_track(frame_count=20, jitter_sigma=1.5, seed=1)
    b = generate_track(frame_count=20, jitter_sigma=1.5, seed=2)
    assert a != b


def test_zero_jitter_is_seed_independent() -> None:
    # With no jitter the RNG is never consulted, so seed cannot matter.
    a = generate_track(frame_count=20, jitter_sigma=0.0, seed=1)
    b = generate_track(frame_count=20, jitter_sigma=0.0, seed=999)
    assert a == b


# --- timestamp / frame ordering ----------------------------------------------
def test_timestamps_ordered_spaced_and_aware() -> None:
    states = generate_track(frame_count=10, frame_interval_s=0.5)
    for i in range(len(states) - 1):
        assert states[i + 1].timestamp > states[i].timestamp
    for i, s in enumerate(states):
        assert s.timestamp == DEFAULT_START_TIME + timedelta(seconds=0.5 * i)
        assert s.timestamp.tzinfo is not None


def test_frame_indices_sequential_with_offset() -> None:
    states = generate_track(frame_count=10, start_frame_index=5)
    assert [s.frame_index for s in states] == list(range(5, 15))


# --- movement / stationary correctness ---------------------------------------
def test_linear_movement_advances_along_direction() -> None:
    states = generate_track(
        start=(100.0, 500.0), direction=(0.0, -1.0), step_size=10.0, frame_count=5
    )
    for i, s in enumerate(states):
        assert _center(s) == pytest.approx((100.0, 500.0 - 10.0 * i))


def test_stationary_positions_constant_zero_velocity() -> None:
    states = generate_track(start=(300.0, 300.0), step_size=0.0, frame_count=8)
    assert {_center(s) for s in states} == {(300.0, 300.0)}
    for s in states:
        assert s.velocity is not None
        assert s.velocity.vx == 0.0
        assert s.velocity.vy == 0.0


def test_velocity_matches_configured_step() -> None:
    step, interval = 9.0, 1.0 / 30.0
    states = generate_track(
        start=(500.0, 900.0),
        direction=(0.0, -1.0),
        step_size=step,
        frame_count=5,
        frame_interval_s=interval,
    )
    for s in states[:-1]:
        assert s.velocity is not None
        assert s.velocity.vx == pytest.approx(0.0)
        assert s.velocity.vy == pytest.approx(-step / interval)


def test_include_velocity_false_leaves_none() -> None:
    states = generate_track(frame_count=3, include_velocity=False)
    assert all(s.velocity is None for s in states)


# --- jitter bounds -----------------------------------------------------------
def test_jitter_bounded_by_clamp_times_sigma() -> None:
    sigma, clamp = 3.0, 2.5
    start, dvec, step, n = (500.0, 500.0), (1.0, -1.0), 8.0, 40
    clean = linear_positions(start, dvec, step, n)
    states = generate_track(
        start=start,
        direction=dvec,
        step_size=step,
        frame_count=n,
        jitter_sigma=sigma,
        jitter_clamp_sigmas=clamp,
        seed=3,
    )
    bound = clamp * sigma + 1e-9
    for i, s in enumerate(states):
        cx, cy = _center(s)
        assert abs(cx - clean[i][0]) <= bound
        assert abs(cy - clean[i][1]) <= bound
    # Jitter actually perturbed at least one frame off the clean line.
    assert any(_center(s) != clean[i] for i, s in enumerate(states))


# --- dropped-frame behavior --------------------------------------------------
def test_dropped_frames_omitted() -> None:
    states = generate_track(frame_count=10, dropped_frames={2, 3, 7})
    assert [s.frame_index for s in states] == [0, 1, 4, 5, 6, 8, 9]


def test_all_frames_dropped_yields_empty() -> None:
    assert generate_track(frame_count=5, dropped_frames=range(5)) == []


def test_drop_does_not_shift_surviving_frames() -> None:
    full = generate_track(frame_count=10, jitter_sigma=2.0, seed=11)
    dropped = generate_track(frame_count=10, jitter_sigma=2.0, seed=11, dropped_frames={4})
    full_by_idx = {s.frame_index: s for s in full}
    for s in dropped:
        assert s == full_by_idx[s.frame_index]


# --- TrackState compatibility ------------------------------------------------
def test_states_are_valid_trackstates() -> None:
    states = generate_track(frame_count=6)
    for s in states:
        assert isinstance(s, TrackState)
        assert isinstance(s.bbox, BoundingBox)
        assert s.bbox.x2 > s.bbox.x1
        assert s.bbox.y2 > s.bbox.y1
        assert isinstance(s.object_class, ObjectClass)
        assert s.status is TrackStatus.ACTIVE


def test_trackstate_json_roundtrip() -> None:
    s = generate_track(frame_count=3, jitter_sigma=1.0, seed=5)[0]
    assert TrackState.model_validate_json(s.model_dump_json()) == s


def test_bbox_center_equals_point_away_from_origin() -> None:
    states = generate_track(start=(500.0, 500.0), step_size=0.0, frame_count=1)
    b = states[0].bbox
    assert (b.x1, b.y1, b.x2, b.y2) == (480.0, 480.0, 520.0, 520.0)


def test_bbox_clamped_non_negative_near_origin() -> None:
    b = build_track([(5.0, 5.0)], bbox_size=(40.0, 40.0))[0].bbox
    assert (b.x1, b.y1, b.x2, b.y2) == (0.0, 0.0, 40.0, 40.0)


# --- geometry compatibility --------------------------------------------------
def test_geometry_consumes_centers() -> None:
    states = generate_track(
        start=(960.0, 1000.0), direction=(0.0, -1.0), step_size=10.0, frame_count=10
    )
    centers = [_center(s) for s in states]
    poly = ((900.0, 1010.0), (1020.0, 1010.0), (1020.0, 860.0), (900.0, 860.0))
    assert all(point_in_polygon(c, poly) for c in centers)
    heading = direction(centers[0], centers[-1])
    assert angle_between_degrees(heading, (0.0, -1.0)) == pytest.approx(0.0)


# --- no mutation -------------------------------------------------------------
def test_input_positions_not_mutated() -> None:
    positions = [(10.0, 10.0), (10.0, 20.0), (10.0, 30.0)]
    snapshot = list(positions)
    build_track(positions, jitter_sigma=1.0, seed=1)
    assert positions == snapshot


# --- position builders -------------------------------------------------------
def test_curved_positions_count_and_heading_change() -> None:
    pos = curved_positions((100.0, 100.0), (0.0, -1.0), 5.0, 10, 0.1)
    assert len(pos) == 10
    d0 = direction(pos[0], pos[1])
    d1 = direction(pos[-2], pos[-1])
    assert angle_between_degrees(d0, d1) > 1.0


def test_segmented_positions_length_and_shape() -> None:
    pos = segmented_positions((100.0, 100.0), [((0.0, -1.0), 5.0, 4), ((1.0, 0.0), 5.0, 3)])
    assert len(pos) == 1 + 4 + 3
    assert pos[4][1] < pos[0][1]  # first leg moved up (y decreased)
    assert pos[-1][0] > pos[4][0]  # second leg moved right (x increased)


def test_segmented_zero_step_holds_position() -> None:
    pos = segmented_positions((10.0, 10.0), [((0.0, -1.0), 0.0, 3)])
    assert pos == [(10.0, 10.0)] * 4


# --- error handling ----------------------------------------------------------
def test_zero_direction_moving_raises() -> None:
    with pytest.raises(ZeroVectorError):
        linear_positions((0.0, 0.0), (0.0, 0.0), 5.0, 3)


def test_zero_direction_stationary_ok() -> None:
    assert linear_positions((1.0, 1.0), (0.0, 0.0), 0.0, 3) == [(1.0, 1.0)] * 3


def test_build_track_empty_positions_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_track([])


def test_invalid_frame_count_raises() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        linear_positions((0.0, 0.0), (0.0, -1.0), 5.0, 0)


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError, match="frame_interval_s"):
        build_track([(0.0, 0.0)], frame_interval_s=0.0)


def test_invalid_bbox_size_raises() -> None:
    with pytest.raises(ValueError, match="bbox_size"):
        build_track([(100.0, 100.0)], bbox_size=(0.0, 40.0))


def test_negative_leg_count_raises() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        segmented_positions((0.0, 0.0), [((0.0, -1.0), 5.0, -1)])

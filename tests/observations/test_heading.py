"""Tests for heading-vs-lane observation derivation (P1-U4, concern 1).

Deterministic, model-free tests: angular deviation for legal/opposite/
perpendicular/diagonal movement, zero-displacement and insufficient-length
handling, timestamp/identity provenance, legal-direction consumption and scaling
invariance, tainted-data abstention, threshold-comparison semantics, and
immutability, and lane containment / boundary abstain. Uses synthetic
TrackStates only.
"""

import pytest

from trafficpulse.contracts.scene import DirectionVector
from trafficpulse.geometry import point_in_polygon
from trafficpulse.observations.heading import (
    derive_heading_observations,
    derive_heading_observations_with_taint,
)
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


# --- lane containment + boundary abstain (architecture-review 5a) -------------
# Heading is only meaningful relative to the lane whose legal direction it is
# compared against. Without containment the direction is applied to every track in
# frame, so lawful traffic on an opposing carriageway -- which moves at ~180 deg to
# the declared direction by construction -- confirms as a violation.
#
# A two-carriageway camera, in one frame's pixel space: the governed lane on the
# left, an opposing carriageway on the right, nothing shared between them.
GOVERNED_LANE = [(100.0, 0.0), (200.0, 0.0), (200.0, 400.0), (100.0, 400.0)]
OPPOSING_CARRIAGEWAY_X = 500.0
DOWN = DirectionVector(dx=0.0, dy=1.0)  # legal flow for these cases: increasing y


def _straight_track(x: float, *, direction: int, states: int = 10, step: float = 20.0):  # type: ignore[no-untyped-def]
    """A constant-x track stepping ``direction * step`` in y, as 40x40 boxes."""

    return build_track(
        linear_positions((x, 200.0), (0.0, float(direction)), step, states),
        track_id=f"track-x{int(x)}",
    )


def _contradictions(track, **kwargs):  # type: ignore[no-untyped-def]
    obs = derive_heading_observations(
        track, legal_direction=DOWN, lane_id="lane-governed", deviation_max_degrees=120.0, **kwargs
    )
    return obs, [o for o in obs if o.is_contradiction]


def test_inside_lane_wrong_way_still_contradicts() -> None:
    # The case containment must NOT break: a track opposing the legal direction
    # inside the governed lane is exactly what the rule is for.
    track = _straight_track(150.0, direction=-1)
    obs, contradicting = _contradictions(track, lane_polygon=GOVERNED_LANE)
    assert obs, "an in-lane track must still produce observations"
    assert len(contradicting) == len(obs)
    assert {o.lane_id for o in obs} == {"lane-governed"}


def test_inside_lane_legal_travel_never_contradicts() -> None:
    _, contradicting = _contradictions(
        _straight_track(150.0, direction=1), lane_polygon=GOVERNED_LANE
    )
    assert contradicting == []


def test_opposite_carriageway_traffic_produces_no_observations_at_all() -> None:
    # The reported defect. This track opposes the governed lane's direction, but it
    # is lawful traffic on the other carriageway: the rule must have nothing to say.
    track = _straight_track(OPPOSING_CARRIAGEWAY_X, direction=-1)
    centers = [
        ((s.bbox.x1 + s.bbox.x2) / 2, (s.bbox.y1 + s.bbox.y2) / 2) for s in track
    ]
    assert all(not point_in_polygon(c, GOVERNED_LANE) for c in centers)
    obs, _ = _contradictions(track, lane_polygon=GOVERNED_LANE)
    assert obs == []


def test_without_a_polygon_the_opposite_carriageway_would_contradict() -> None:
    # Characterises precisely what containment buys: the same track, ungated, votes
    # against the lane it is not in. This is the pre-fix behaviour, kept explicit so
    # the gate cannot be quietly removed without this test failing.
    obs, contradicting = _contradictions(_straight_track(OPPOSING_CARRIAGEWAY_X, direction=-1))
    assert obs and len(contradicting) == len(obs)


def test_a_step_leaving_the_lane_is_dropped_but_the_in_lane_steps_survive() -> None:
    # Both endpoints must be inside: a displacement with one end outside is not a
    # movement "inside the lane". The track starts in-lane and exits to the right.
    positions = [(150.0, 200.0), (150.0, 180.0), (150.0, 160.0), (600.0, 160.0), (600.0, 140.0)]
    track = build_track(positions, track_id="exiting")
    obs, _ = _contradictions(track, lane_polygon=GOVERNED_LANE)
    # Steps 1->2 and 2->3 are in-lane; 3->4 (leaving) and 4->5 (outside) are not.
    assert len(obs) == 2


def test_boundary_band_abstains_while_the_lane_interior_still_votes() -> None:
    # A track hugging the inside of the lane edge: contained, but not confidently
    # so. With a margin wider than its clearance it abstains; with none it votes.
    just_inside = _straight_track(102.0, direction=-1)  # center x = 122, 22 px clear
    voting, _ = _contradictions(just_inside, lane_polygon=GOVERNED_LANE)
    assert voting, "with no abstain band a contained track votes"
    abstaining, _ = _contradictions(
        just_inside, lane_polygon=GOVERNED_LANE, boundary_abstain_margin=30.0
    )
    assert abstaining == [], "within the abstain band the track must not vote"


def test_the_abstain_band_does_not_silence_the_lane_interior() -> None:
    # The same margin that silences the edge-hugging track leaves the mid-lane one
    # untouched -- the band abstains near the boundary, it does not shrink the lane.
    obs, contradicting = _contradictions(
        _straight_track(130.0, direction=-1),  # center x = 150, 50 px clear
        lane_polygon=GOVERNED_LANE,
        boundary_abstain_margin=30.0,
    )
    assert obs and len(contradicting) == len(obs)


def test_a_zero_margin_keeps_containment_and_disables_only_the_band() -> None:
    inside, _ = _contradictions(
        _straight_track(102.0, direction=-1),
        lane_polygon=GOVERNED_LANE,
        boundary_abstain_margin=0.0,
    )
    outside, _ = _contradictions(
        _straight_track(OPPOSING_CARRIAGEWAY_X, direction=-1),
        lane_polygon=GOVERNED_LANE,
        boundary_abstain_margin=0.0,
    )
    assert inside and outside == []


def test_containment_skips_are_ordinary_gaps_not_taint_restarts() -> None:
    # An out-of-lane excursion drops observations; it does not mark the resuming
    # observation as a taint discontinuity (that marker means ID-switch only).
    positions = [(150.0, 300.0), (150.0, 280.0), (600.0, 280.0), (150.0, 260.0), (150.0, 240.0)]
    derivation = derive_heading_observations_with_taint(
        build_track(positions, track_id="excursion"),
        legal_direction=DOWN,
        lane_id="lane-governed",
        deviation_max_degrees=120.0,
        lane_polygon=GOVERNED_LANE,
    )
    assert derivation.observations
    assert derivation.taint_restart_ids == frozenset()

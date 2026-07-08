"""Scenario-builder tests for the synthetic trajectory generator (P1-U2).

Verifies each named ``generate_*`` builder produces the intended motion shape,
that motion-relevant scenarios integrate with the real U5 example scene geometry
(lane membership, heading vs the configured legal direction), and that every
builder is deterministic. No violation decision is made -- only geometric shape
facts are asserted.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import SceneConfig, TrackState
from trafficpulse.geometry import angle_between_degrees, direction, point_in_polygon
from trafficpulse.synth import (
    generate_abrupt_turn,
    generate_curved,
    generate_diagonal,
    generate_disappearing,
    generate_enter_then_stop,
    generate_legal,
    generate_noisy,
    generate_short_track,
    generate_slight_drift,
    generate_stationary,
    generate_truncated,
    generate_wrong_way,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = SceneConfig.model_validate(
    yaml.safe_load((REPO_ROOT / "configs" / "scenes" / "example-scene.yaml").read_text("utf-8"))
)
LANE = next(z for z in SCENE.zones if z.zone_id == "zone-lane-north")
NORTH = next(d for d in SCENE.legal_directions if d.direction_id == "dir-north")
LEGAL_VEC = (NORTH.vector.dx, NORTH.vector.dy)

ALL_BUILDERS: tuple[Callable[[], list[TrackState]], ...] = (
    generate_legal,
    generate_wrong_way,
    generate_stationary,
    generate_enter_then_stop,
    generate_short_track,
    generate_noisy,
    generate_slight_drift,
    generate_diagonal,
    generate_curved,
    generate_abrupt_turn,
    generate_disappearing,
    generate_truncated,
)


def _center(ts: TrackState) -> tuple[float, float]:
    b = ts.bbox
    return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)


def _centers(states: list[TrackState]) -> list[tuple[float, float]]:
    return [_center(s) for s in states]


def _heading(states: list[TrackState]) -> tuple[float, float]:
    centers = _centers(states)
    return direction(centers[0], centers[-1])


# --- shared shape invariants -------------------------------------------------
def test_all_builders_return_nonempty_valid_trackstates() -> None:
    for builder in ALL_BUILDERS:
        states = builder()
        assert states, f"{builder.__name__} produced no states"
        for s in states:
            assert isinstance(s, TrackState)
            assert s.bbox.x2 > s.bbox.x1
            assert s.bbox.y2 > s.bbox.y1


def test_all_builders_deterministic() -> None:
    for builder in ALL_BUILDERS:
        assert builder() == builder(), f"{builder.__name__} not deterministic"


# --- legal / wrong-way against real scene geometry ---------------------------
def test_legal_inside_lane_and_aligned() -> None:
    states = generate_legal()
    assert all(point_in_polygon(_center(s), LANE.polygon) for s in states)
    assert angle_between_degrees(_heading(states), LEGAL_VEC) == pytest.approx(0.0)


def test_wrong_way_inside_lane_and_opposed() -> None:
    states = generate_wrong_way()
    assert all(point_in_polygon(_center(s), LANE.polygon) for s in states)
    assert angle_between_degrees(_heading(states), LEGAL_VEC) == pytest.approx(180.0)


# --- individual scenario shapes ----------------------------------------------
def test_stationary_constant_and_zero_velocity() -> None:
    states = generate_stationary()
    assert len({_center(s) for s in states}) == 1
    for s in states:
        assert s.velocity is not None
        assert s.velocity.vx == 0.0
        assert s.velocity.vy == 0.0


def test_enter_then_stop_moves_then_holds() -> None:
    centers = _centers(generate_enter_then_stop())
    assert centers[0] != centers[-1]  # it moved at some point
    assert len(set(centers[-10:])) == 1  # and holds at the end


def test_short_track_is_short() -> None:
    assert len(generate_short_track()) == 3


def test_noisy_reproducible_and_perturbed() -> None:
    assert generate_noisy(seed=3) == generate_noisy(seed=3)
    assert generate_noisy(seed=1) != generate_noisy(seed=2)
    # Jitter introduces horizontal variation absent from the clean legal track.
    xs = {round(_center(s)[0], 6) for s in generate_noisy(seed=3)}
    assert len(xs) > 1


def test_slight_drift_is_small_and_sideways() -> None:
    centers = _centers(generate_slight_drift())
    dx = centers[-1][0] - centers[0][0]
    dy = centers[-1][1] - centers[0][1]
    assert dx > 0.0  # drifts to the right
    assert abs(dx) < abs(dy) * 0.2  # but only slightly, relative to travel


def test_diagonal_monotonic() -> None:
    centers = _centers(generate_diagonal())
    assert all(centers[i + 1][0] > centers[i][0] for i in range(len(centers) - 1))
    assert all(centers[i + 1][1] < centers[i][1] for i in range(len(centers) - 1))


def test_curved_heading_changes() -> None:
    centers = _centers(generate_curved())
    first = direction(centers[0], centers[1])
    last = direction(centers[-2], centers[-1])
    assert angle_between_degrees(first, last) > 5.0


def test_abrupt_turn_is_sharp() -> None:
    centers = _centers(generate_abrupt_turn())
    first = direction(centers[0], centers[1])
    last = direction(centers[-2], centers[-1])
    assert angle_between_degrees(first, last) == pytest.approx(90.0)


def test_disappearing_has_gap() -> None:
    states = generate_disappearing()
    idx = [s.frame_index for s in states]
    assert set(range(12, 18)).isdisjoint(idx)  # the gap is missing
    assert 11 in idx and 18 in idx  # but frames around it are present
    assert len(states) == 24


def test_truncated_is_cut_short() -> None:
    assert len(generate_truncated()) == 10

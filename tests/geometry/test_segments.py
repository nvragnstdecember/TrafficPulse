"""Unit tests for segment orientation, intersection, and crossing facts (P1-U1).

Covers orientation sign, on-segment membership (including degenerate segments),
segment intersection (ordinary, none, endpoint contact, parallel, collinear
overlap/disjoint, degenerate, and axis-aligned crossings), and the directional
``stop_line_crossing`` facts. Coordinates are image-space (top-left origin,
+y down). These are geometric facts only -- no legality decision is made.
"""

from trafficpulse.geometry.segments import (
    CrossingFact,
    orientation,
    point_on_segment,
    segments_intersect,
    side_of_line,
    stop_line_crossing,
)


# --- orientation -------------------------------------------------------------
def test_orientation_collinear_is_zero() -> None:
    assert orientation((0.0, 0.0), (2.0, 2.0), (4.0, 4.0)) == 0


def test_orientation_two_sides() -> None:
    left = orientation((0.0, 0.0), (4.0, 0.0), (2.0, 1.0))
    right = orientation((0.0, 0.0), (4.0, 0.0), (2.0, -1.0))
    assert left != 0
    assert right != 0
    assert left == -right


def test_orientation_sign_flips_with_argument_order() -> None:
    a, b, c = (0.0, 0.0), (4.0, 0.0), (2.0, 3.0)
    assert orientation(a, b, c) == -orientation(b, a, c)


# --- point_on_segment --------------------------------------------------------
def test_on_segment_interior_and_endpoints() -> None:
    a, b = (0.0, 0.0), (4.0, 4.0)
    assert point_on_segment((2.0, 2.0), a, b) is True
    assert point_on_segment(a, a, b) is True
    assert point_on_segment(b, a, b) is True


def test_on_segment_collinear_but_outside() -> None:
    assert point_on_segment((5.0, 5.0), (0.0, 0.0), (4.0, 4.0)) is False


def test_on_segment_not_collinear() -> None:
    assert point_on_segment((2.0, 3.0), (0.0, 0.0), (4.0, 4.0)) is False


def test_on_segment_degenerate() -> None:
    assert point_on_segment((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)) is True
    assert point_on_segment((2.0, 2.0), (1.0, 1.0), (1.0, 1.0)) is False


# --- segments_intersect ------------------------------------------------------
def test_ordinary_intersection() -> None:
    assert segments_intersect((0.0, 0.0), (4.0, 4.0), (0.0, 4.0), (4.0, 0.0)) is True


def test_no_intersection() -> None:
    assert segments_intersect((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 4.0)) is False


def test_endpoint_contact() -> None:
    # Shared endpoint (2,2) counts as an intersection.
    assert segments_intersect((0.0, 0.0), (2.0, 2.0), (2.0, 2.0), (4.0, 0.0)) is True


def test_touch_endpoint_on_interior() -> None:
    # "T" contact: an endpoint lies on the interior of the other segment.
    assert segments_intersect((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 3.0)) is True


def test_parallel_non_intersection() -> None:
    assert segments_intersect((0.0, 0.0), (4.0, 0.0), (0.0, 2.0), (4.0, 2.0)) is False


def test_collinear_overlap() -> None:
    assert segments_intersect((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (6.0, 0.0)) is True


def test_collinear_disjoint() -> None:
    assert segments_intersect((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)) is False


def test_degenerate_movement_on_segment() -> None:
    # A zero-length movement that lands on the line segment intersects it.
    assert segments_intersect((2.0, 0.0), (2.0, 0.0), (0.0, 0.0), (4.0, 0.0)) is True


def test_degenerate_movement_off_segment() -> None:
    assert segments_intersect((2.0, 5.0), (2.0, 5.0), (0.0, 0.0), (4.0, 0.0)) is False


def test_vertical_horizontal_crossing() -> None:
    assert segments_intersect((2.0, 0.0), (2.0, 4.0), (0.0, 2.0), (4.0, 2.0)) is True


# --- directional crossing facts ---------------------------------------------
LINE_A = (0.0, 0.0)
LINE_B = (4.0, 0.0)  # a horizontal stop line at y = 0


def test_crossing_side_change_one_direction() -> None:
    fact = stop_line_crossing((2.0, -1.0), (2.0, 1.0), LINE_A, LINE_B)
    assert isinstance(fact, CrossingFact)
    assert fact.previous_side != 0
    assert fact.current_side != 0
    assert fact.previous_side != fact.current_side
    assert fact.side_changed is True
    assert fact.intersects_segment is True


def test_crossing_side_change_opposite_direction() -> None:
    forward = stop_line_crossing((2.0, -1.0), (2.0, 1.0), LINE_A, LINE_B)
    backward = stop_line_crossing((2.0, 1.0), (2.0, -1.0), LINE_A, LINE_B)
    assert backward.side_changed is True
    # The sides are the mirror of the forward movement.
    assert backward.previous_side == forward.current_side
    assert backward.current_side == forward.previous_side


def test_crossing_no_side_change() -> None:
    fact = stop_line_crossing((1.0, 1.0), (3.0, 1.0), LINE_A, LINE_B)
    assert fact.previous_side == fact.current_side
    assert fact.side_changed is False
    assert fact.intersects_segment is False


def test_crossing_point_on_line() -> None:
    # Current point lies exactly on the line: side 0, so no side change even
    # though the movement segment touches the stop-line segment.
    fact = stop_line_crossing((2.0, -1.0), (2.0, 0.0), LINE_A, LINE_B)
    assert fact.current_side == 0
    assert fact.side_changed is False
    assert fact.intersects_segment is True


def test_side_of_line_matches_orientation() -> None:
    p = (2.0, -1.0)
    assert side_of_line(p, LINE_A, LINE_B) == orientation(LINE_A, LINE_B, p)


# --- property-style invariants ----------------------------------------------
_SEG_PAIRS = [
    (((0.0, 0.0), (4.0, 4.0)), ((0.0, 4.0), (4.0, 0.0))),  # crossing
    (((0.0, 0.0), (4.0, 0.0)), ((0.0, 2.0), (4.0, 2.0))),  # parallel
    (((0.0, 0.0), (4.0, 0.0)), ((2.0, 0.0), (6.0, 0.0))),  # collinear overlap
    (((0.0, 0.0), (2.0, 0.0)), ((4.0, 0.0), (6.0, 0.0))),  # collinear disjoint
    (((0.0, 0.0), (2.0, 2.0)), ((2.0, 2.0), (4.0, 0.0))),  # endpoint contact
]


def test_invariant_intersection_symmetric_segment_order() -> None:
    for s1, s2 in _SEG_PAIRS:
        forward = segments_intersect(s1[0], s1[1], s2[0], s2[1])
        swapped = segments_intersect(s2[0], s2[1], s1[0], s1[1])
        assert forward == swapped


def test_invariant_intersection_symmetric_endpoint_order() -> None:
    for s1, s2 in _SEG_PAIRS:
        base = segments_intersect(s1[0], s1[1], s2[0], s2[1])
        rev1 = segments_intersect(s1[1], s1[0], s2[0], s2[1])
        rev2 = segments_intersect(s1[0], s1[1], s2[1], s2[0])
        assert base == rev1 == rev2


def test_invariant_deterministic_repeated_calls() -> None:
    for _ in range(5):
        assert segments_intersect((0.0, 0.0), (4.0, 4.0), (0.0, 4.0), (4.0, 0.0)) is True

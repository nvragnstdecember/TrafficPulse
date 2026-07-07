"""Unit tests for point-in-polygon membership (P1-U1).

Covers convex and concave polygons; points inside, outside, on edges, and on
vertices; horizontal and vertical edges; the "boundary counts as inside" policy;
and determinism. Coordinates are image-space (top-left origin, +y down).
"""

import pytest

from trafficpulse.geometry.polygons import point_in_polygon

# A convex axis-aligned square (ordered ring; closing edge implicit).
SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))

# A concave L-shape. The bottom-right square [2,4] x [2,4] is the notch and lies
# OUTSIDE the polygon.
L_SHAPE = (
    (0.0, 0.0),
    (4.0, 0.0),
    (4.0, 2.0),
    (2.0, 2.0),
    (2.0, 4.0),
    (0.0, 4.0),
)

# A convex non-axis-aligned triangle.
TRIANGLE = ((0.0, 0.0), (4.0, 1.0), (1.0, 4.0))


# --- convex: inside / outside ------------------------------------------------
def test_convex_point_inside() -> None:
    assert point_in_polygon((2.0, 2.0), SQUARE) is True


def test_convex_point_outside() -> None:
    assert point_in_polygon((5.0, 5.0), SQUARE) is False
    assert point_in_polygon((-1.0, 2.0), SQUARE) is False


def test_triangle_inside_outside() -> None:
    assert point_in_polygon((1.5, 1.5), TRIANGLE) is True
    assert point_in_polygon((3.5, 3.5), TRIANGLE) is False


# --- boundary policy: on edge / on vertex count as inside --------------------
def test_point_on_edge_is_inside() -> None:
    assert point_in_polygon((2.0, 0.0), SQUARE) is True  # top edge
    assert point_in_polygon((4.0, 2.0), SQUARE) is True  # right edge


def test_point_on_vertex_is_inside() -> None:
    assert point_in_polygon((0.0, 0.0), SQUARE) is True
    assert point_in_polygon((4.0, 4.0), SQUARE) is True


def test_point_on_reflex_vertex_is_inside() -> None:
    # The concave (reflex) vertex of the L-shape.
    assert point_in_polygon((2.0, 2.0), L_SHAPE) is True


# --- concave: inside and outside-in-notch ------------------------------------
def test_concave_inside_top_band() -> None:
    assert point_in_polygon((1.0, 1.0), L_SHAPE) is True


def test_concave_inside_left_column() -> None:
    assert point_in_polygon((1.0, 3.0), L_SHAPE) is True


def test_concave_outside_in_notch() -> None:
    # (3, 3) sits in the notch of the L -> outside the polygon.
    assert point_in_polygon((3.0, 3.0), L_SHAPE) is False


# --- horizontal / vertical edges ---------------------------------------------
def test_point_on_horizontal_edge() -> None:
    # Horizontal edge (4,2)-(2,2) of the L-shape.
    assert point_in_polygon((3.0, 2.0), L_SHAPE) is True


def test_point_on_vertical_edge() -> None:
    # Vertical edge (2,2)-(2,4) of the L-shape.
    assert point_in_polygon((2.0, 3.0), L_SHAPE) is True


def test_interior_near_horizontal_edge() -> None:
    # y coincides with two horizontal edges; interior parity must be correct.
    assert point_in_polygon((1.0, 2.0), L_SHAPE) is True


def test_interior_near_vertical_edge() -> None:
    assert point_in_polygon((3.0, 1.0), L_SHAPE) is True


# --- degenerate input --------------------------------------------------------
def test_too_few_vertices_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        point_in_polygon((0.0, 0.0), ((0.0, 0.0), (1.0, 1.0)))


# --- property-style invariants ----------------------------------------------
def test_invariant_deterministic_repeated_calls() -> None:
    for _ in range(5):
        assert point_in_polygon((2.0, 2.0), SQUARE) is True
        assert point_in_polygon((3.0, 3.0), L_SHAPE) is False


def test_invariant_polygon_input_not_mutated() -> None:
    poly = tuple(SQUARE)
    point_in_polygon((2.0, 2.0), poly)
    assert poly == SQUARE

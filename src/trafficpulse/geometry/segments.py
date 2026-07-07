"""Line-segment orientation, intersection, and directional crossing facts.

Deterministic, dependency-free geometry over the image-space convention
documented in :mod:`trafficpulse.geometry.vectors` (top-left origin, +y down).

This module answers *geometric* questions only:

* how three points are oriented (the sign of a cross-product determinant);
* whether a point lies on a closed segment;
* whether two closed segments intersect;
* which side of an infinite line a point lies on;
* whether a movement from one point to another changed sides of a line and/or
  crossed a finite segment.

It decides nothing about red-light violations, legal vs illegal crossing,
signal state, temporal grace, or persistence -- those belong to later
observation/rule/temporal units. In particular ``stop_line_crossing`` reports
side facts about a configured stop line; it does not interpret them.
"""

from typing import NamedTuple

from .vectors import NUMERIC_EPSILON, Point, cross, displacement


def orientation(a: Point, b: Point, c: Point) -> int:
    """Return the orientation of the ordered triple ``(a, b, c)``.

    Computed as the sign of ``cross(b - a, c - a)``:

    * ``0``  -- ``a``, ``b``, ``c`` are collinear (within ``NUMERIC_EPSILON``);
    * ``+1`` -- ``c`` lies on one side of the directed line ``a -> b``;
    * ``-1`` -- ``c`` lies on the other side.

    Under the image-space convention (``y`` increasing downward), a positive
    value corresponds to what appears as a clockwise turn on screen; the sign is
    used here only to distinguish the two sides consistently, never to assert a
    visual turn direction.
    """

    val = cross(displacement(a, b), displacement(a, c))
    if val > NUMERIC_EPSILON:
        return 1
    if val < -NUMERIC_EPSILON:
        return -1
    return 0


def point_on_segment(p: Point, a: Point, b: Point) -> bool:
    """Return ``True`` if ``p`` lies on the closed segment ``[a, b]``.

    "Closed" means the endpoints count: ``p == a`` or ``p == b`` returns
    ``True``. A degenerate segment (``a == b``) is treated as the single point
    ``a``; then the result is ``True`` iff ``p`` coincides with ``a``.
    """

    if orientation(a, b, p) != 0:
        return False
    within_x = (
        min(a[0], b[0]) - NUMERIC_EPSILON <= p[0] <= max(a[0], b[0]) + NUMERIC_EPSILON
    )
    within_y = (
        min(a[1], b[1]) - NUMERIC_EPSILON <= p[1] <= max(a[1], b[1]) + NUMERIC_EPSILON
    )
    return within_x and within_y


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Return ``True`` if closed segments ``[p1, p2]`` and ``[p3, p4]`` meet.

    Boundary contact counts as intersection: a shared endpoint, a "T" touch
    where an endpoint lies on the other segment, and any collinear overlap all
    return ``True``. Collinear-but-disjoint and parallel-non-touching segments
    return ``False``.

    The result is symmetric under swapping the two segments and under swapping
    the endpoints within a segment. Either or both segments may be degenerate
    (a single point); a degenerate segment intersects iff its point lies on the
    other segment.
    """

    d1 = orientation(p3, p4, p1)
    d2 = orientation(p3, p4, p2)
    d3 = orientation(p1, p2, p3)
    d4 = orientation(p1, p2, p4)

    # General case: each segment straddles the line of the other.
    if d1 != d2 and d3 != d4:
        return True

    # Collinear / touching cases: an endpoint lies on the opposing segment.
    if d1 == 0 and point_on_segment(p1, p3, p4):
        return True
    if d2 == 0 and point_on_segment(p2, p3, p4):
        return True
    if d3 == 0 and point_on_segment(p3, p1, p2):
        return True
    return d4 == 0 and point_on_segment(p4, p1, p2)


def side_of_line(p: Point, a: Point, b: Point) -> int:
    """Return which side of the infinite line through ``a``, ``b`` holds ``p``.

    Alias for ``orientation(a, b, p)``: ``0`` on the line, ``+1`` / ``-1`` for
    the two sides (see :func:`orientation` for the sign convention).
    """

    return orientation(a, b, p)


class CrossingFact(NamedTuple):
    """Purely geometric facts about a movement relative to a line/segment.

    Attributes:
        previous_side: Side of the infinite line for the previous point
            (``+1`` / ``-1``; ``0`` if the point lies on the line).
        current_side: Side of the infinite line for the current point.
        side_changed: ``True`` iff both sides are non-zero and differ -- i.e.
            the movement passed from one side of the infinite line to the other.
            A point lying exactly on the line (side ``0``) never counts as a
            side change.
        intersects_segment: ``True`` iff the movement segment actually crosses
            or touches the *finite* line segment (not merely its infinite line).
    """

    previous_side: int
    current_side: int
    side_changed: bool
    intersects_segment: bool


def stop_line_crossing(
    previous: Point, current: Point, line_a: Point, line_b: Point
) -> CrossingFact:
    """Report geometric crossing facts for a movement against a line segment.

    Given a movement ``previous -> current`` and a stop-line segment
    ``[line_a, line_b]``, this returns which side of the line's infinite
    extension each point is on, whether the side changed, and whether the
    movement segment intersects the finite stop-line segment.

    It makes **no** legality, signal-state, temporal, or persistence decision.
    Comparing the movement direction against a configured crossing direction is
    left to the caller (e.g. via ``vectors.angle_between_degrees``).
    """

    prev_side = side_of_line(previous, line_a, line_b)
    curr_side = side_of_line(current, line_a, line_b)
    side_changed = prev_side != 0 and curr_side != 0 and prev_side != curr_side
    intersects = segments_intersect(previous, current, line_a, line_b)
    return CrossingFact(prev_side, curr_side, side_changed, intersects)

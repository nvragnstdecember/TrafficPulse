"""Point-in-polygon membership for TrafficPulse geometry (P1-U1).

Deterministic, dependency-free membership test over the image-space convention
documented in :mod:`trafficpulse.geometry.vectors` (top-left origin, +y down).
It answers a single geometric question -- "is this point inside this polygon?"
-- and decides nothing about zones, violations, or rules.

Scope: simple polygons as produced by U5 ``SceneConfig`` zones/ROIs (an ordered
ring of at least three vertices, closed implicitly). Self-intersecting polygons
and arbitrary GIS geometry are out of scope and not supported.
"""

from collections.abc import Sequence

from .segments import point_on_segment
from .vectors import Point


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return ``True`` if ``point`` lies inside ``polygon``.

    Boundary policy: **the boundary counts as inside.** A point lying exactly on
    any edge or vertex of the polygon returns ``True``.

    The polygon is an ordered ring of vertices; the closing edge from the last
    vertex back to the first is implicit (matching the U5 ``ordered_ring``
    convention), so the caller must not repeat the first vertex at the end.

    Interior membership uses the even-odd (ray-casting) rule with a horizontal
    ray toward increasing ``x``. Horizontal edges are skipped by the half-open
    ``y`` comparison and cannot cause a division by zero; vertical edges are
    handled naturally. Because boundary points are resolved first, the classic
    vertex/edge ray-casting ambiguities do not affect the result.

    Raises:
        ValueError: if ``polygon`` has fewer than three vertices.
    """

    n = len(polygon)
    if n < 3:
        raise ValueError("polygon requires at least 3 vertices")

    # Boundary counts as inside: resolve on-edge/on-vertex points up front.
    for i in range(n):
        if point_on_segment(point, polygon[i], polygon[(i + 1) % n]):
            return True

    # Even-odd rule: count edge crossings of the horizontal ray at y = point.y
    # going toward +x. An edge crosses the ray line iff its endpoints are on
    # strictly opposite sides of y (half-open test), which also excludes
    # horizontal edges and guarantees ``by != ay`` below.
    x, y = point
    inside = False
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if (ay > y) != (by > y):
            x_cross = ax + (y - ay) / (by - ay) * (bx - ax)
            if x < x_cross:
                inside = not inside
    return inside

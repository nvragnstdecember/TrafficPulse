"""Deterministic geometry primitives for TrafficPulse (Phase 1, unit P1-U1).

A small, well-tested geometry layer that converts typed spatial inputs into
reusable *geometric facts* for later observation generation (zone membership,
stop-line crossing, movement direction, heading-vs-legal-direction deviation,
and trajectory displacement).

This layer answers geometric questions only. It is independent of detectors,
trackers, video ingestion, ML models, violation rules, temporal FSMs, event
confirmation, and evidence generation, and it embeds no behavioral threshold.
It depends only on the Python standard library (``math``) and consumes plain
``tuple[float, float]`` points/vectors -- the same structure U5 scene
configuration uses -- without importing or mutating the contract models.

See the module docstrings for the coordinate convention (image space, top-left
origin, +y down), the numerical tolerance policy, angle units, zero-vector
behavior, and point-in-polygon boundary behavior.
"""

from .polygons import point_in_polygon
from .segments import (
    CrossingFact,
    orientation,
    point_on_segment,
    segments_intersect,
    side_of_line,
    stop_line_crossing,
)
from .vectors import (
    NUMERIC_EPSILON,
    Point,
    Vector,
    ZeroVectorError,
    angle_between_degrees,
    cross,
    direction,
    displacement,
    dot,
    is_zero_vector,
    magnitude,
    normalize,
)

__all__ = [
    "NUMERIC_EPSILON",
    # structural types
    "Point",
    "Vector",
    # vectors
    "ZeroVectorError",
    "displacement",
    "magnitude",
    "is_zero_vector",
    "normalize",
    "direction",
    "dot",
    "cross",
    "angle_between_degrees",
    # segments
    "orientation",
    "point_on_segment",
    "segments_intersect",
    "side_of_line",
    "CrossingFact",
    "stop_line_crossing",
    # polygons
    "point_in_polygon",
]

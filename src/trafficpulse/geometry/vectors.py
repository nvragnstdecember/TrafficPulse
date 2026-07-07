"""2D vector/scalar primitives for TrafficPulse geometry (P1-U1).

These are deterministic, dependency-free (stdlib ``math`` only) helpers that
answer *geometric* questions. They make no violation, temporal, or legality
decision and embed no behavioral threshold.

Coordinate convention
---------------------
All points and vectors live in the image-space convention frozen by U5
(``configs/scenes/schema.yaml`` and ``FrameSpec``), matching the U2
``BoundingBox``:

* pixel coordinates;
* origin at the top-left;
* ``x`` increases to the right;
* ``y`` increases **downward**.

Because ``y`` increases downward, this module deliberately avoids visual terms
such as "clockwise", "left turn", or "positive rotation". The only orientation
notion used is the sign of the 2D cross-product determinant (see
``geometry.segments.orientation``), whose meaning is defined there.

Structural types
----------------
``Point`` and ``Vector`` are both ``tuple[float, float]``. This is intentionally
the same structure as the U5 ``Point`` alias (``contracts.scene.Point``) and as
a ``DirectionVector``'s ``(dx, dy)`` pair, so scene-configuration data flows in
without conversion, yet geometry stays independent of the pydantic contract
layer (no import, no mutation, no duplication of those models).

Angle units
-----------
Angular deviation is reported in **degrees**, in the closed range
``[0, 180]`` -- compatible with the U5 provisional ``degrees`` parameter unit
and the U2 ``DeviationDegrees`` alias.

Zero-vector behavior
--------------------
A vector whose magnitude is ``<= NUMERIC_EPSILON`` is treated as the zero
vector. Its direction is undefined, so ``normalize``, ``direction``, and
``angle_between_degrees`` raise :class:`ZeroVectorError` rather than returning a
silent NaN. Callers that want to branch without catching an exception can test
``is_zero_vector`` first.

Numerical tolerance policy
--------------------------
``NUMERIC_EPSILON`` is the single named absolute tolerance for floating-point
comparisons in this package. It is a numerical implementation detail -- **not**
a behavioral threshold (e.g. a wrong-way angle, a stop duration, or a minimum
speed), none of which belong in geometry.
"""

import math

# --- Structural types --------------------------------------------------------
Point = tuple[float, float]
Vector = tuple[float, float]

# --- Numerical tolerance policy ----------------------------------------------
# Single, named, absolute tolerance for float comparisons (zero-vector
# detection, orientation sign, on-segment bounds, cosine clamping). It guards
# floating-point noise; for exact integer pixel inputs comparisons are exact.
NUMERIC_EPSILON: float = 1e-9


class ZeroVectorError(ValueError):
    """Raised when an operation needs a non-zero (directioned) vector.

    The zero vector has no defined direction, so normalization, movement
    direction, and angular deviation are undefined for it.
    """


def displacement(a: Point, b: Point) -> Vector:
    """Return the displacement vector from ``a`` to ``b`` (``b - a``)."""

    return (b[0] - a[0], b[1] - a[1])


def dot(a: Vector, b: Vector) -> float:
    """Return the dot product ``a . b``."""

    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vector, b: Vector) -> float:
    """Return the scalar 2D cross product ``ax*by - ay*bx``.

    This is the z-component of the 3D cross product of ``a`` and ``b`` embedded
    in the plane. It is a low-level helper reused by orientation tests; its sign
    convention under image coordinates is documented on
    ``geometry.segments.orientation``.
    """

    return a[0] * b[1] - a[1] * b[0]


def magnitude(v: Vector) -> float:
    """Return the Euclidean length of ``v`` (always ``>= 0``)."""

    return math.hypot(v[0], v[1])


def is_zero_vector(v: Vector) -> bool:
    """Return ``True`` if ``v`` is the zero vector within ``NUMERIC_EPSILON``.

    Equivalent to "``v`` is not structurally usable as a direction".
    """

    return magnitude(v) <= NUMERIC_EPSILON


def normalize(v: Vector) -> Vector:
    """Return the unit vector in the direction of ``v``.

    Raises:
        ZeroVectorError: if ``v`` is the zero vector (``is_zero_vector``).
    """

    m = magnitude(v)
    if m <= NUMERIC_EPSILON:
        raise ZeroVectorError("cannot normalize a zero-length vector")
    return (v[0] / m, v[1] / m)


def direction(a: Point, b: Point) -> Vector:
    """Return the unit movement direction from ``a`` to ``b``.

    This is the smallest deterministic "movement direction" helper: the
    normalized displacement between two positions. It performs no smoothing,
    interpolation, jitter filtering, or persistence.

    Raises:
        ZeroVectorError: if ``a`` and ``b`` coincide (degenerate movement has no
            defined direction).
    """

    return normalize(displacement(a, b))


def angle_between_degrees(a: Vector, b: Vector) -> float:
    """Return the angular deviation between ``a`` and ``b`` in degrees.

    The result lies in the closed range ``[0, 180]`` and is symmetric in its
    arguments (``angle_between_degrees(a, b) == angle_between_degrees(b, a)``).
    It is orientation-reference-free: it is the unsigned angle between the two
    directions, so it does not depend on the image-space rotation sense.

    The cosine is clamped to ``[-1, 1]`` before ``acos`` so that floating-point
    error on parallel/antiparallel inputs cannot raise a math-domain error.

    Raises:
        ZeroVectorError: if either argument is the zero vector.
    """

    ma = magnitude(a)
    mb = magnitude(b)
    if ma <= NUMERIC_EPSILON or mb <= NUMERIC_EPSILON:
        raise ZeroVectorError("angular deviation is undefined for a zero vector")
    cos_theta = dot(a, b) / (ma * mb)
    # Clamp to the valid domain of acos; float error can push it slightly out.
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))

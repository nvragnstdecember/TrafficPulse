"""Unit tests for geometry vector primitives (P1-U1).

Deterministic, model-free tests of displacement, magnitude, normalization,
dot/cross products, movement direction, and angular deviation, including
explicit zero-vector behavior, cosine clamping, and property-style invariants.
Angles are in image-space (top-left origin, +y down); deviation is in degrees
in ``[0, 180]``.
"""

import math

import pytest

from trafficpulse.geometry.vectors import (
    NUMERIC_EPSILON,
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


# --- displacement ------------------------------------------------------------
def test_displacement_basic() -> None:
    assert displacement((1.0, 2.0), (4.0, 6.0)) == (3.0, 4.0)


def test_displacement_negative_and_zero() -> None:
    assert displacement((5.0, 5.0), (2.0, 9.0)) == (-3.0, 4.0)
    assert displacement((7.0, 7.0), (7.0, 7.0)) == (0.0, 0.0)


# --- magnitude ---------------------------------------------------------------
def test_magnitude_3_4_5() -> None:
    assert magnitude((3.0, 4.0)) == 5.0


def test_magnitude_zero() -> None:
    assert magnitude((0.0, 0.0)) == 0.0


# --- normalization -----------------------------------------------------------
def test_normalize_unit_length() -> None:
    nx, ny = normalize((3.0, 4.0))
    assert math.isclose(nx, 0.6)
    assert math.isclose(ny, 0.8)


def test_normalize_preserves_direction() -> None:
    # A scaled copy normalizes to the same unit vector.
    assert normalize((0.0, -2.0)) == pytest.approx((0.0, -1.0))


# --- zero-vector behavior (explicit) -----------------------------------------
def test_is_zero_vector_true_false() -> None:
    assert is_zero_vector((0.0, 0.0)) is True
    assert is_zero_vector((NUMERIC_EPSILON / 2, 0.0)) is True
    assert is_zero_vector((1.0, 0.0)) is False


def test_normalize_zero_raises() -> None:
    with pytest.raises(ZeroVectorError):
        normalize((0.0, 0.0))


def test_direction_degenerate_movement_raises() -> None:
    with pytest.raises(ZeroVectorError):
        direction((10.0, 10.0), (10.0, 10.0))


def test_angle_zero_vector_raises() -> None:
    with pytest.raises(ZeroVectorError):
        angle_between_degrees((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ZeroVectorError):
        angle_between_degrees((1.0, 0.0), (0.0, 0.0))


# --- dot / cross -------------------------------------------------------------
def test_dot_product() -> None:
    assert dot((1.0, 2.0), (3.0, 4.0)) == 11.0
    assert dot((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_cross_product_sign() -> None:
    # +x cross +y is positive under this coordinate algebra.
    assert cross((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert cross((0.0, 1.0), (1.0, 0.0)) == -1.0


# --- direction (movement) ----------------------------------------------------
def test_direction_is_unit_displacement() -> None:
    # Moving up-the-image (decreasing y) yields (0, -1).
    assert direction((100.0, 500.0), (100.0, 200.0)) == pytest.approx((0.0, -1.0))


# --- angular deviation -------------------------------------------------------
def test_angle_same_direction_is_zero() -> None:
    assert angle_between_degrees((0.0, -1.0), (0.0, -3.0)) == pytest.approx(0.0)


def test_angle_opposite_direction_is_180() -> None:
    assert angle_between_degrees((0.0, -1.0), (0.0, 1.0)) == pytest.approx(180.0)


def test_angle_perpendicular_is_90() -> None:
    assert angle_between_degrees((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_angle_non_axis_aligned() -> None:
    # (1,0) vs (1,1) is 45 degrees regardless of image-space y-down convention.
    assert angle_between_degrees((1.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)
    # (1,1) vs (-1,1) is 90 degrees.
    assert angle_between_degrees((1.0, 1.0), (-1.0, 1.0)) == pytest.approx(90.0)


def test_angle_matches_legal_direction_deviation() -> None:
    # Movement up the image vs a legal "north" (up) direction => ~0 deviation;
    # a reversed movement => ~180. (No threshold applied here.)
    legal = (0.0, -1.0)
    up = direction((0.0, 100.0), (0.0, 0.0))
    down = direction((0.0, 0.0), (0.0, 100.0))
    assert angle_between_degrees(up, legal) == pytest.approx(0.0)
    assert angle_between_degrees(down, legal) == pytest.approx(180.0)


def test_angle_cosine_clamped_no_domain_error() -> None:
    # Parallel/antiparallel scaled copies must never raise a math-domain error
    # from acos even when float division nudges the cosine past +-1.
    samples = [
        ((0.1, 0.2), (0.2, 0.4)),
        ((0.1, 0.2), (0.3, 0.6)),
        ((0.7, -0.3), (7.0, -3.0)),
        ((0.1, 0.2), (-0.2, -0.4)),
        ((1e-3, 2e-3), (2e-3, 4e-3)),
    ]
    for a, b in samples:
        deg = angle_between_degrees(a, b)
        assert 0.0 <= deg <= 180.0
        assert not math.isnan(deg)


# --- property-style invariants ----------------------------------------------
_VECTORS = [
    (3.0, 4.0),
    (-2.0, 5.0),
    (0.0, -7.0),
    (1.5, -2.5),
    (100.0, 0.0),
    (-0.1, -0.2),
]


def test_invariant_magnitude_non_negative() -> None:
    for v in _VECTORS:
        assert magnitude(v) >= 0.0


def test_invariant_normalized_has_unit_magnitude() -> None:
    for v in _VECTORS:
        assert magnitude(normalize(v)) == pytest.approx(1.0)


def test_invariant_angle_symmetric() -> None:
    for a in _VECTORS:
        for b in _VECTORS:
            assert angle_between_degrees(a, b) == pytest.approx(angle_between_degrees(b, a))


def test_invariant_angle_in_range() -> None:
    for a in _VECTORS:
        for b in _VECTORS:
            deg = angle_between_degrees(a, b)
            assert 0.0 <= deg <= 180.0


def test_invariant_dot_symmetric() -> None:
    for a in _VECTORS:
        for b in _VECTORS:
            assert dot(a, b) == dot(b, a)


def test_invariant_inputs_not_mutated() -> None:
    a = (3.0, 4.0)
    b = (1.0, -2.0)
    normalize(a)
    displacement(a, b)
    angle_between_degrees(a, b)
    direction(a, b)
    assert a == (3.0, 4.0)
    assert b == (1.0, -2.0)

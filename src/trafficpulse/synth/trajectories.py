"""Deterministic synthetic trajectory generation (P1-U2).

A *reproducible source of ``TrackState`` sequences* for testing the observation
and rule pipeline before detector, tracker, or video uncertainty exists. This
is not a simulator, tracker, detector, or visualizer -- it only fabricates
ordered ``TrackState`` data.

Pipeline
--------
1. A parametric *position builder* (``linear_positions``, ``curved_positions``,
   ``segmented_positions``) returns an ordered list of clean center points in
   the U5 image-space convention (top-left origin, +y down).
2. :func:`build_track` turns those centers into a ``list[TrackState]``: it wraps
   each point in a valid ``BoundingBox`` centered on it, assigns a tz-aware
   ``timestamp`` and increasing ``frame_index``, optionally sets a ground-truth
   ``Velocity``, and optionally applies bounded jitter and dropped frames.

Determinism
-----------
Output is a pure function of ``(seed, parameters)``. A single local
``random.Random(seed)`` drives all jitter; there is no dependence on wall-clock
time and no shared/global RNG. Identical seed and parameters always yield
identical trajectories. With ``jitter_sigma == 0`` no random numbers are drawn,
so the result is independent of ``seed`` by design.

Randomness
----------
Jitter is bounded and Gaussian-like, built only from the standard library. Each
axis offset is ``z * sigma`` where ``z`` is a Bates/central-limit approximation
of a standard normal (``sum of 12 uniforms - 6``; mean 0, variance 1, support
``[-6, 6]``) clamped to ``+/- jitter_clamp_sigmas``. Hence every jitter offset
satisfies ``abs(offset) <= jitter_clamp_sigmas * sigma``.

Coordinate / contract compatibility
------------------------------------
Points and vectors are the P1-U1 ``geometry`` structural types
(``tuple[float, float]``); direction normalization reuses ``geometry.normalize``
(which raises ``ZeroVectorError`` for a zero direction). ``BoundingBox`` centers
are clamped to be non-negative so every emitted box is valid per the U2
contract; a recovered center equals the trajectory point exactly whenever that
point is at least half a box away from the frame origin.
"""

import math
import random
from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta

from ..contracts import BoundingBox, ObjectClass, TrackState, TrackStatus, Velocity
from ..geometry import Point, Vector, normalize

# --- defaults (synthetic; no real deployment meaning) ------------------------
DEFAULT_CAMERA_ID = "cam-synthetic-01"
DEFAULT_TRACK_ID = "synthetic-track"
DEFAULT_START_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
DEFAULT_FRAME_INTERVAL_S = 1.0 / 30.0  # 30 fps
DEFAULT_BBOX_SIZE: tuple[float, float] = (40.0, 40.0)
DEFAULT_JITTER_CLAMP_SIGMAS = 3.0

# Number of uniform samples averaged for the bounded Gaussian approximation.
# The sum of 12 U(0, 1) variates has variance 1, so ``sum - 6`` approximates a
# standard normal with support in [-6, 6].
_BATES_N = 12


def _bounded_gaussian(rng: random.Random, sigma: float, clamp_sigmas: float) -> float:
    """Return a deterministic, bounded, Gaussian-like offset.

    Draws nothing (and returns ``0.0``) when ``sigma <= 0`` so a zero-jitter
    trajectory never consumes the RNG. Otherwise the result lies strictly within
    ``+/- clamp_sigmas * sigma``.
    """

    if sigma <= 0.0:
        return 0.0
    standard = sum(rng.random() for _ in range(_BATES_N)) - _BATES_N / 2.0
    standard = max(-clamp_sigmas, min(clamp_sigmas, standard))
    return standard * sigma


# --- position builders -------------------------------------------------------
def linear_positions(
    start: Point, direction: Vector, step_size: float, frame_count: int
) -> list[Point]:
    """Return ``frame_count`` points advancing from ``start`` along ``direction``.

    ``direction`` need not be unit length; it is normalized. ``step_size`` is the
    per-frame distance in pixels. A ``step_size`` of ``0`` yields a stationary
    sequence (all points equal ``start``) regardless of ``direction``.

    Raises:
        ValueError: if ``frame_count < 1``.
        ZeroVectorError: if ``step_size != 0`` and ``direction`` is the zero
            vector (a moving track needs a defined direction).
    """

    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if step_size == 0.0:
        return [start for _ in range(frame_count)]
    ux, uy = normalize(direction)
    return [
        (start[0] + ux * step_size * i, start[1] + uy * step_size * i)
        for i in range(frame_count)
    ]


def curved_positions(
    start: Point,
    initial_direction: Vector,
    step_size: float,
    frame_count: int,
    turn_rate_radians: float,
) -> list[Point]:
    """Return points along a piecewise-linear arc approximation.

    The unit step direction is rotated by ``turn_rate_radians`` each frame (a
    constant-curvature approximation), then advanced by ``step_size``. Rotation
    is the standard planar rotation; under the image-space +y-down convention a
    positive ``turn_rate_radians`` curves toward what appears on screen as a
    clockwise turn (documented here, not asserted as a semantic direction).

    Raises:
        ValueError: if ``frame_count < 1``.
        ZeroVectorError: if ``initial_direction`` is the zero vector.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    dx, dy = normalize(initial_direction)
    cos_t = math.cos(turn_rate_radians)
    sin_t = math.sin(turn_rate_radians)
    x, y = start
    positions: list[Point] = [start]
    for _ in range(frame_count - 1):
        dx, dy = dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t
        x += dx * step_size
        y += dy * step_size
        positions.append((x, y))
    return positions


def segmented_positions(
    start: Point, legs: Sequence[tuple[Vector, float, int]]
) -> list[Point]:
    """Return points for a sequence of straight legs joined end to end.

    Each leg is ``(direction, step_size, frame_count)`` and contributes
    ``frame_count`` new points after the current position. The returned list
    begins with ``start`` (frame 0) and has length ``1 + sum(frame_count)``. A
    leg with ``step_size == 0`` holds position (useful for "enter then stop").

    Raises:
        ValueError: if any leg count is negative.
        ZeroVectorError: if a moving leg (``step_size != 0``) has a zero
            direction vector.
    """

    x, y = start
    positions: list[Point] = [start]
    for direction, step_size, count in legs:
        if count < 0:
            raise ValueError("leg frame_count must be >= 0")
        ux, uy = (0.0, 0.0) if step_size == 0.0 else normalize(direction)
        for _ in range(count):
            x += ux * step_size
            y += uy * step_size
            positions.append((x, y))
    return positions


# --- TrackState assembly -----------------------------------------------------
def _clean_velocities(positions: Sequence[Point], frame_interval_s: float) -> list[Vector]:
    """Ground-truth per-frame velocity (px/s) from consecutive clean positions.

    Forward difference for every frame except the last, which reuses the last
    backward difference. A single-point sequence has zero velocity.
    """

    n = len(positions)
    if n == 1:
        return [(0.0, 0.0)]
    velocities: list[Vector] = []
    for i in range(n):
        a, b = (positions[i], positions[i + 1]) if i < n - 1 else (positions[i - 1], positions[i])
        velocities.append(((b[0] - a[0]) / frame_interval_s, (b[1] - a[1]) / frame_interval_s))
    return velocities


def build_track(
    positions: Sequence[Point],
    *,
    track_id: str = DEFAULT_TRACK_ID,
    camera_id: str = DEFAULT_CAMERA_ID,
    object_class: ObjectClass = ObjectClass.MOTORCYCLE,
    status: TrackStatus = TrackStatus.ACTIVE,
    start_time: datetime = DEFAULT_START_TIME,
    frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
    start_frame_index: int = 0,
    bbox_size: tuple[float, float] = DEFAULT_BBOX_SIZE,
    jitter_sigma: float = 0.0,
    jitter_clamp_sigmas: float = DEFAULT_JITTER_CLAMP_SIGMAS,
    seed: int = 0,
    dropped_frames: Collection[int] = (),
    include_velocity: bool = True,
    confidence: float | None = None,
    tainted: bool = False,
) -> list[TrackState]:
    """Assemble a ``list[TrackState]`` from an ordered list of center positions.

    Each position becomes one ``TrackState`` unless its ``frame_index`` is in
    ``dropped_frames`` (interpreted as absolute ``frame_index`` values), in which
    case it is omitted -- producing a real gap in ``frame_index`` and
    ``timestamp`` for "disappearing/reappearing" tracks. Jitter is drawn for
    every position (dropped ones included) so surviving frames are unaffected by
    which others are dropped.

    Raises:
        ValueError: if ``positions`` is empty, ``frame_interval_s <= 0``,
            ``start_frame_index < 0``, or a ``bbox_size`` component is not
            positive.
    """

    if not positions:
        raise ValueError("positions must be non-empty")
    if frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s must be > 0")
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be >= 0")
    if bbox_size[0] <= 0.0 or bbox_size[1] <= 0.0:
        raise ValueError("bbox_size components must be > 0")

    rng = random.Random(seed)
    dropped = set(dropped_frames)
    velocities = _clean_velocities(positions, frame_interval_s) if include_velocity else None
    half_w = bbox_size[0] / 2.0
    half_h = bbox_size[1] / 2.0

    states: list[TrackState] = []
    for i, (cx, cy) in enumerate(positions):
        # Draw jitter unconditionally to keep the RNG stream drop-independent.
        jx = _bounded_gaussian(rng, jitter_sigma, jitter_clamp_sigmas)
        jy = _bounded_gaussian(rng, jitter_sigma, jitter_clamp_sigmas)
        frame_index = start_frame_index + i
        if frame_index in dropped:
            continue

        px = cx + jx
        py = cy + jy
        x1 = max(0.0, px - half_w)
        y1 = max(0.0, py - half_h)
        bbox = BoundingBox(x1=x1, y1=y1, x2=x1 + bbox_size[0], y2=y1 + bbox_size[1])

        velocity: Velocity | None = None
        if velocities is not None:
            vx, vy = velocities[i]
            velocity = Velocity(vx=vx, vy=vy)

        states.append(
            TrackState(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=start_time + timedelta(seconds=frame_interval_s * i),
                frame_index=frame_index,
                object_class=object_class,
                bbox=bbox,
                confidence=confidence,
                status=status,
                tainted=tainted,
                velocity=velocity,
            )
        )
    return states


def generate_track(
    *,
    start: Point = (960.0, 900.0),
    direction: Vector = (0.0, -1.0),
    step_size: float = 6.0,
    frame_count: int = 30,
    track_id: str = DEFAULT_TRACK_ID,
    camera_id: str = DEFAULT_CAMERA_ID,
    object_class: ObjectClass = ObjectClass.MOTORCYCLE,
    status: TrackStatus = TrackStatus.ACTIVE,
    start_time: datetime = DEFAULT_START_TIME,
    frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
    start_frame_index: int = 0,
    bbox_size: tuple[float, float] = DEFAULT_BBOX_SIZE,
    jitter_sigma: float = 0.0,
    jitter_clamp_sigmas: float = DEFAULT_JITTER_CLAMP_SIGMAS,
    seed: int = 0,
    dropped_frames: Collection[int] = (),
    include_velocity: bool = True,
    confidence: float | None = None,
    tainted: bool = False,
) -> list[TrackState]:
    """Convenience linear generator: ``linear_positions`` + :func:`build_track`.

    Exposes the configurable knobs the unit requires (start, direction, step
    size, frame count, timestamp interval, jitter, dropped frames, seed) in one
    call. For non-linear shapes use a position builder with :func:`build_track`.
    """

    positions = linear_positions(start, direction, step_size, frame_count)
    return build_track(
        positions,
        track_id=track_id,
        camera_id=camera_id,
        object_class=object_class,
        status=status,
        start_time=start_time,
        frame_interval_s=frame_interval_s,
        start_frame_index=start_frame_index,
        bbox_size=bbox_size,
        jitter_sigma=jitter_sigma,
        jitter_clamp_sigmas=jitter_clamp_sigmas,
        seed=seed,
        dropped_frames=dropped_frames,
        include_velocity=include_velocity,
        confidence=confidence,
        tainted=tainted,
    )

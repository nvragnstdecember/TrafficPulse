"""Reusable synthetic-trajectory scenario builders (P1-U2).

Thin, opinionated wrappers over :mod:`trafficpulse.synth.trajectories` that each
return a ``list[TrackState]`` for a named motion pattern used to exercise the
downstream observation/rule pipeline. Defaults place motion-relevant scenarios
inside the synthetic example scene's northbound lane (``zone-lane-north`` in
``configs/scenes/example-scene.yaml``) and align them with its legal direction
(``dir-north`` = up the image, decreasing y), so tests can feed real scene
geometry without hard-coding any rule threshold here.

These builders make no violation decision; a name like ``generate_wrong_way``
describes the synthetic *motion shape*, not a verdict. Only ``generate_noisy``
is stochastic (and takes a ``seed``); every other builder is fully deterministic
with no randomness, so it needs no seed.
"""

from ..contracts import TrackState
from .trajectories import (
    build_track,
    curved_positions,
    generate_track,
    segmented_positions,
)

# Geometry that keeps the default tracks inside the example northbound lane.
_LANE_BOTTOM = (960.0, 1040.0)  # near the bottom of zone-lane-north
_LANE_TOP = (960.0, 700.0)  # near the top of zone-lane-north
_UP: tuple[float, float] = (0.0, -1.0)  # legal "north": decreasing y
_DOWN: tuple[float, float] = (0.0, 1.0)  # against the legal direction
_STEP = 12.0
_FRAMES = 30


def generate_legal(*, frame_count: int = _FRAMES) -> list[TrackState]:
    """Straight, legal travel: up the lane, aligned with ``dir-north``."""

    return generate_track(
        start=_LANE_BOTTOM,
        direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="legal-track",
    )


def generate_wrong_way(*, frame_count: int = _FRAMES) -> list[TrackState]:
    """Sustained wrong-way travel: down the lane, opposite ``dir-north``."""

    return generate_track(
        start=_LANE_TOP,
        direction=_DOWN,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="wrong-way-track",
    )


def generate_stationary(*, frame_count: int = _FRAMES) -> list[TrackState]:
    """A vehicle that never moves (``step_size == 0``)."""

    return generate_track(
        start=(960.0, 850.0),
        direction=_UP,
        step_size=0.0,
        frame_count=frame_count,
        track_id="stationary-track",
    )


def generate_enter_then_stop(
    *, moving_frames: int = 10, stopped_frames: int = 20
) -> list[TrackState]:
    """A vehicle that travels for a while, then holds position."""

    positions = segmented_positions(
        _LANE_BOTTOM,
        legs=[(_UP, _STEP, moving_frames), (_UP, 0.0, stopped_frames)],
    )
    return build_track(positions, track_id="enter-then-stop-track")


def generate_short_track(*, frame_count: int = 3) -> list[TrackState]:
    """A very short track (too few frames for sustained reasoning)."""

    return generate_track(
        start=(960.0, 1000.0),
        direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="short-track",
    )


def generate_noisy(
    *, seed: int = 0, frame_count: int = _FRAMES, jitter_sigma: float = 2.0
) -> list[TrackState]:
    """Legal travel with bounded pseudo-Gaussian positional jitter."""

    return generate_track(
        start=_LANE_BOTTOM,
        direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="noisy-track",
        jitter_sigma=jitter_sigma,
        seed=seed,
    )


def generate_slight_drift(*, frame_count: int = _FRAMES) -> list[TrackState]:
    """Mostly-vertical travel with a small constant sideways drift."""

    return generate_track(
        start=_LANE_BOTTOM,
        direction=(0.08, -1.0),
        step_size=_STEP,
        frame_count=frame_count,
        track_id="drift-track",
    )


def generate_diagonal(*, frame_count: int = _FRAMES) -> list[TrackState]:
    """Diagonal travel (up and to the right, ~45 degrees)."""

    return generate_track(
        start=(760.0, 1000.0),
        direction=(1.0, -1.0),
        step_size=10.0,
        frame_count=frame_count,
        track_id="diagonal-track",
    )


def generate_curved(
    *, frame_count: int = _FRAMES, turn_rate_radians: float = 0.05
) -> list[TrackState]:
    """A gently curving approach (constant-curvature approximation)."""

    positions = curved_positions(
        _LANE_BOTTOM,
        initial_direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        turn_rate_radians=turn_rate_radians,
    )
    return build_track(positions, track_id="curved-track")


def generate_abrupt_turn(*, first_leg: int = 15, second_leg: int = 15) -> list[TrackState]:
    """Two straight legs meeting at a sharp (~90 degree) turn."""

    positions = segmented_positions(
        _LANE_BOTTOM,
        legs=[(_UP, _STEP, first_leg), ((1.0, 0.0), _STEP, second_leg)],
    )
    return build_track(positions, track_id="abrupt-turn-track")


def generate_disappearing(
    *, frame_count: int = _FRAMES, gap: tuple[int, int] = (12, 18)
) -> list[TrackState]:
    """A track with a contiguous block of dropped (missing) frames.

    ``gap`` is a half-open ``[start, stop)`` range of ``frame_index`` values to
    omit, leaving a real gap between the disappearing and reappearing segments.
    """

    return generate_track(
        start=_LANE_BOTTOM,
        direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="disappearing-track",
        dropped_frames=range(gap[0], gap[1]),
    )


def generate_truncated(*, frame_count: int = 10) -> list[TrackState]:
    """A track cut short mid-motion (ends before a natural completion)."""

    return generate_track(
        start=_LANE_BOTTOM,
        direction=_UP,
        step_size=_STEP,
        frame_count=frame_count,
        track_id="truncated-track",
    )

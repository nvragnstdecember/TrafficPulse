"""Deterministic synthetic trajectory generation for TrafficPulse (P1-U2).

A reproducible source of ``TrackState`` sequences for testing the observation
and rule pipeline before detector, tracker, or video uncertainty is introduced.
It is *not* a simulator, tracker, detector, or visualizer: it only fabricates
ordered ``TrackState`` data from parametric motion descriptions.

Public surface:

* core -- ``build_track`` (assemble ``TrackState``s from center positions),
  ``generate_track`` (linear convenience generator), and the position builders
  ``linear_positions`` / ``curved_positions`` / ``segmented_positions``;
* scenarios -- twelve named ``generate_*`` builders for common motion shapes.

Everything is deterministic (a pure function of seed and parameters), depends
only on the Python standard library plus the frozen U2 ``contracts`` and P1-U1
``geometry`` layers, and hard-codes no behavioral rule threshold.
"""

from .scenarios import (
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
from .trajectories import (
    DEFAULT_BBOX_SIZE,
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_INTERVAL_S,
    DEFAULT_JITTER_CLAMP_SIGMAS,
    DEFAULT_START_TIME,
    DEFAULT_TRACK_ID,
    build_track,
    curved_positions,
    generate_track,
    linear_positions,
    segmented_positions,
)

__all__ = [
    # defaults
    "DEFAULT_BBOX_SIZE",
    "DEFAULT_CAMERA_ID",
    "DEFAULT_FRAME_INTERVAL_S",
    "DEFAULT_JITTER_CLAMP_SIGMAS",
    "DEFAULT_START_TIME",
    "DEFAULT_TRACK_ID",
    # core
    "linear_positions",
    "curved_positions",
    "segmented_positions",
    "build_track",
    "generate_track",
    # scenarios
    "generate_legal",
    "generate_wrong_way",
    "generate_stationary",
    "generate_enter_then_stop",
    "generate_short_track",
    "generate_noisy",
    "generate_slight_drift",
    "generate_diagonal",
    "generate_curved",
    "generate_abrupt_turn",
    "generate_disappearing",
    "generate_truncated",
]

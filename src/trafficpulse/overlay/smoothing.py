"""Presentation-only temporal smoothing for overlay geometry and displayed values.

Detector boxes and classifier scores jitter frame to frame. That jitter is *real
data* -- it must reach the tracker, the reasoner, the event store, and the evidence
manifest exactly as measured. But drawing it verbatim makes a correct system look
unstable: boxes shiver, captions flip between layout candidates, and a confidence
readout churns through digits that carry no information.

This module smooths only what is **drawn**. It sits in the visualization layer,
consumes values the inference pass already produced, and writes nothing back: no
detection, track, observation, event, or manifest is affected, and removing it
changes only pixels. Nothing here is violation-specific -- it is arithmetic over
keyed numeric streams, so every present and future overlay provider can use it.

Two mechanisms, deliberately both
---------------------------------
* an **exponential moving average** removes high-frequency shake while following
  genuine motion (a low ``alpha`` lags more and calms more);
* a **deadband** then refuses to republish a value that has barely moved, which is
  what actually stops flicker. An EMA alone still emits a slightly different number
  every frame -- enough to redraw a caption one pixel over, or tick a percentage
  between 96% and 97% forever. The deadband holds the published value until the
  smoothed one has drifted far enough to be worth showing.

Discontinuities are respected, not smeared
-------------------------------------------
A stream that disappears for more than ``max_gap_frames`` and comes back is treated
as new: the smoother restarts from the incoming value rather than interpolating
across the gap. Otherwise a track that reappears elsewhere in the frame would be
drawn gliding across the scene through positions nothing was ever observed at --
fabricating motion, which is exactly what a visualization layer must not do.

Determinism
-----------
A smoother is a small state machine over a stream consumed in frame order, so a
caller must feed frames in ascending order (providers precompute the whole capture
up front, which keeps their per-frame lookup pure and order-independent). No
wall-clock, no randomness: the same capture always yields the same drawn geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SmoothingConfig:
    """How hard to smooth, and when to give up and restart.

    Defaults are tuned for 25-30 fps traffic footage: boxes settle within a few
    frames while still tracking a vehicle crossing the frame, and a displayed
    percentage stops churning without lagging a real change in the reading.
    """

    #: EMA weight for a new box observation (lower = calmer, more lag).
    box_alpha: float = 0.45
    #: Republish a box only once a corner has drifted this far, in pixels.
    box_deadband_px: float = 1.5
    #: EMA weight for a new scalar observation (e.g. a confidence).
    value_alpha: float = 0.25
    #: Republish a scalar only once it has drifted this far (0.02 = 2 points).
    value_deadband: float = 0.02
    #: Frames a stream may vanish for and still be considered continuous.
    max_gap_frames: int = 4


@dataclass
class _Stream:
    """One key's running state: the EMA, what was last published, and when."""

    smoothed: tuple[float, ...]
    published: tuple[float, ...]
    last_frame: int


class StreamSmoother:
    """EMA + deadband over keyed numeric streams consumed in frame order.

    Keys are opaque -- a caller uses whatever identity it already has (a track id, a
    track id plus a role suffix). Boxes and scalars share one keyspace, so use
    distinct keys for distinct quantities of the same entity.
    """

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self._config = config if config is not None else SmoothingConfig()
        self._streams: dict[str, _Stream] = {}

    @property
    def config(self) -> SmoothingConfig:
        return self._config

    def box(self, key: str, frame_index: int, bounds: Bounds) -> Bounds:
        """Return the drawable box for ``bounds``, smoothed and deadbanded."""

        out = self._advance(
            key,
            frame_index,
            tuple(float(v) for v in bounds),
            alpha=self._config.box_alpha,
            deadband=self._config.box_deadband_px,
        )
        return (out[0], out[1], out[2], out[3])

    def value(self, key: str, frame_index: int, value: float | None) -> float | None:
        """Return the drawable scalar for ``value`` (``None`` passes through).

        A ``None`` is an *absent measurement*, not a zero, so it is never smoothed
        into the stream and never invents a number: it is returned as-is and leaves
        the running state untouched, ready to resume when a real value returns.
        """

        if value is None:
            return None
        return self._advance(
            key,
            frame_index,
            (float(value),),
            alpha=self._config.value_alpha,
            deadband=self._config.value_deadband,
        )[0]

    def _advance(
        self,
        key: str,
        frame_index: int,
        observation: tuple[float, ...],
        *,
        alpha: float,
        deadband: float,
    ) -> tuple[float, ...]:
        stream = self._streams.get(key)
        if stream is None or frame_index - stream.last_frame > self._config.max_gap_frames:
            # New or resumed after a gap: start from what was actually observed
            # rather than gliding there from a stale position.
            self._streams[key] = _Stream(
                smoothed=observation, published=observation, last_frame=frame_index
            )
            return observation

        smoothed = tuple(
            alpha * new + (1.0 - alpha) * old
            for new, old in zip(observation, stream.smoothed, strict=True)
        )
        stream.smoothed = smoothed
        stream.last_frame = frame_index
        moved = max(abs(s - p) for s, p in zip(smoothed, stream.published, strict=True))
        if moved >= deadband:
            stream.published = smoothed
        return stream.published

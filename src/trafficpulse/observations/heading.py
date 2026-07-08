"""Heading-vs-lane observation derivation (P1-U4, concern 1).

Deterministically converts an ordered ``TrackState`` sequence plus a configured
legal direction into ``HeadingVsLaneObservation`` facts, using only P1-U1
geometry. This is *observation derivation*, not reasoning: it computes per-step
geometric facts (movement heading, deviation from the legal direction, and a
per-step ``is_contradiction`` flag against a configured angular threshold) and
makes no violation, persistence, or confirmation decision.

Position source
---------------
``TrackState`` carries a ``bbox`` (and an optional ``velocity``); it has no
explicit point. The representative position used here is the **bbox center**.
For the P1-U2 synthetic source (constant-size boxes) the center recovers the
generated trajectory point exactly, and because displacement direction is
invariant to any *consistent* reference point, the choice is inconsequential to
the heading. Center is chosen for exactness and simplicity; a ground-contact
(bottom-center) reference is an equivalent future option once calibrated
ground-plane reasoning is added. The prompt requires displacement between
consecutive positions, so the optional ``velocity`` field is deliberately not
used.

Heading convention
------------------
``heading_degrees`` and ``legal_heading_degrees`` are ``atan2(dy, dx)`` mapped to
``[0, 360)`` in the U5 image-space convention (origin top-left, +y down): 0 deg
is +x (right), increasing toward +y (down). They are provenance only; the
load-bearing quantity is ``deviation_degrees`` -- the reference-free unsigned
angle in ``[0, 180]`` from P1-U1 ``angle_between_degrees``.

Explicit edge behavior
----------------------
* Fewer than two TrackStates -> no observations.
* A zero-displacement step (``is_zero_vector`` via the geometry numeric epsilon,
  a numerical fact, not a behavioral threshold) -> no observation for that step.
  This is an *ordinary* gap (a genuinely missing/immobile sample of one track).
* A step whose either endpoint is a tainted ``TrackState`` -> no observation,
  and the next clean observation is flagged as a **taint restart** (see below).

Ordinary gaps vs explicit taint
-------------------------------
An ordinary gap (missing or zero-displacement samples of a single continuous
track) and an explicit taint (an ID-switch discontinuity, architecture-review
§13 -- "tainted tracks may abstain but never confirm") must not be conflated.
Both drop observations, but taint additionally marks the first clean observation
that resumes after it, via ``HeadingDerivation.taint_restart_ids``. The reasoning
layer resets its persistence run at those restarts, so wrong-way support can
never silently accumulate *across* a tainted interval, while ordinary gaps keep
their timestamp-driven bridging. ``derive_heading_observations`` (the plain
observation list) is unchanged; ``derive_heading_observations_with_taint``
additionally returns the restart markers.

No smoothing, no interpolation, no frame inference, no tracking.
"""

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..contracts import HeadingVsLaneObservation, Producer, TrackState
from ..contracts.enums import ProducerKind
from ..contracts.scene import DirectionVector
from ..geometry import Vector, angle_between_degrees, displacement, is_zero_vector

DEFAULT_PRODUCER = Producer(
    name="wrong-way-heading", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)


@dataclass(frozen=True)
class HeadingDerivation:
    """Observations plus the ids of observations that resume after taint.

    ``taint_restart_ids`` are the ``observation_id``s of clean observations that
    immediately follow one or more tainted steps; the reasoner treats them as
    explicit discontinuities and resets its persistence run there.
    """

    observations: tuple[HeadingVsLaneObservation, ...]
    taint_restart_ids: frozenset[str]


def _center(track_state: TrackState) -> Vector:
    box = track_state.bbox
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _heading_degrees(vector: Vector) -> float:
    """Absolute heading of ``vector`` in ``[0, 360)`` (image space, +y down)."""

    return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _observation_id(camera_id: str, track_id: str, lane_id: str, iso_timestamp: str) -> str:
    preimage = "\x1f".join((camera_id, track_id, lane_id, iso_timestamp))
    return "hvl-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _iter_derivation(
    track: Sequence[TrackState],
    *,
    legal_direction: DirectionVector,
    lane_id: str,
    deviation_max_degrees: float,
    producer: Producer | None,
) -> Iterator[tuple[HeadingVsLaneObservation, bool]]:
    """Yield ``(observation, is_taint_restart)`` for each usable step.

    ``is_taint_restart`` is ``True`` when the observation is the first clean one
    after one or more tainted steps. Zero-displacement (ordinary-gap) skips do
    not set it -- only taint does.
    """

    legal_vector: Vector = (legal_direction.dx, legal_direction.dy)
    legal_heading = _heading_degrees(legal_vector)
    prod = producer if producer is not None else DEFAULT_PRODUCER
    taint_since_last_emit = False

    for previous, current in zip(track, track[1:], strict=False):
        if previous.tainted or current.tainted:
            taint_since_last_emit = True  # abstain on tainted data; mark discontinuity
            continue
        step: Vector = displacement(_center(previous), _center(current))
        if is_zero_vector(step):
            continue  # ordinary gap: no usable heading, but NOT a taint discontinuity
        deviation = angle_between_degrees(step, legal_vector)
        observation = HeadingVsLaneObservation(
            observation_id=_observation_id(
                current.camera_id, current.track_id, lane_id, current.timestamp.isoformat()
            ),
            camera_id=current.camera_id,
            track_id=current.track_id,
            timestamp=current.timestamp,
            producer=prod,
            lane_id=lane_id,
            heading_degrees=_heading_degrees(step),
            legal_heading_degrees=legal_heading,
            deviation_degrees=deviation,
            is_contradiction=deviation > deviation_max_degrees,
        )
        yield observation, taint_since_last_emit
        taint_since_last_emit = False


def derive_heading_observations(
    track: Sequence[TrackState],
    *,
    legal_direction: DirectionVector,
    lane_id: str,
    deviation_max_degrees: float,
    producer: Producer | None = None,
) -> list[HeadingVsLaneObservation]:
    """Derive ``HeadingVsLaneObservation`` facts from a TrackState sequence.

    Returns one observation per usable consecutive step, in input order. Steps
    involving a tainted TrackState or with zero displacement are skipped. Use
    :func:`derive_heading_observations_with_taint` when the taint-discontinuity
    markers are needed for reasoning.

    Args:
        track: ordered TrackStates (as produced by the P1-U2 synth source).
        legal_direction: the configured lane legal direction (U5 value object);
            only its ``(dx, dy)`` are used, and its magnitude does not affect the
            deviation.
        lane_id: the configured lane/zone id recorded on each observation.
        deviation_max_degrees: the provisional configured angular threshold
            (``heading_deviation_max``); a step is a contradiction iff its
            deviation strictly exceeds it. Passed in from configuration.
        producer: observation provenance (defaults to a synthetic heuristic).
    """

    return [
        observation
        for observation, _ in _iter_derivation(
            track,
            legal_direction=legal_direction,
            lane_id=lane_id,
            deviation_max_degrees=deviation_max_degrees,
            producer=producer,
        )
    ]


def derive_heading_observations_with_taint(
    track: Sequence[TrackState],
    *,
    legal_direction: DirectionVector,
    lane_id: str,
    deviation_max_degrees: float,
    producer: Producer | None = None,
) -> HeadingDerivation:
    """Like :func:`derive_heading_observations`, but also return taint restarts.

    The returned ``taint_restart_ids`` name the observations that resume after a
    tainted interval; the reasoning layer resets its persistence run there so
    support cannot bridge the tainted (ID-switch) discontinuity.
    """

    observations: list[HeadingVsLaneObservation] = []
    restart_ids: set[str] = set()
    for observation, is_restart in _iter_derivation(
        track,
        legal_direction=legal_direction,
        lane_id=lane_id,
        deviation_max_degrees=deviation_max_degrees,
        producer=producer,
    ):
        observations.append(observation)
        if is_restart:
            restart_ids.add(observation.observation_id)
    return HeadingDerivation(tuple(observations), frozenset(restart_ids))

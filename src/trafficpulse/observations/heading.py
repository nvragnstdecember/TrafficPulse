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

Lane containment and boundary abstain
-------------------------------------
Heading is only meaningful *relative to the lane whose legal direction it is
compared against*, so a step is derived only when it happened **inside that
lane's polygon** (architecture-review §5a: "track heading/displacement inside a
lane polygon"). Without this gate the configured legal direction is applied to
every track in frame, and lawful traffic on an opposing carriageway -- which by
construction moves at ~180 deg to the declared direction -- is reported as a
contradiction. On a real divided-road camera that is a false-positive generator.

The same section names "near the polygon boundary" as an abstain condition: a
reference point within ``boundary_abstain_margin`` of the boundary is not
confidently in the lane (detection jitter alone can move it across), so its step
abstains rather than voting. ``margin = 0.0`` keeps containment but disables the
abstain band; ``lane_polygon = None`` disables both, which is the behaviour the
pure-geometry callers (and this module's own unit tests) rely on. The
wrong-way pipeline always supplies the polygon -- see
:func:`trafficpulse.pipeline.wrong_way.wrong_way_finalize_strategy`.

Containment uses the **bbox center**, deliberately the same reference point this
module already measures displacement between, so "the displacement happened
inside the lane" is a statement about one consistent point. This differs from
:mod:`trafficpulse.observations.zones`, which asks a different question ("does
this vehicle *occupy* the zone") and answers it with the ground-contact
bottom-center. Both choices are provisional in the same way and for the same
reason: neither survives contact with calibrated ground-plane reasoning.

A containment/boundary skip is an **ordinary gap**, not a taint restart: it drops
the step's observation and nothing more. Whether support should also be *reset*
when a track leaves and re-enters the lane is a persistence-semantics question
this derivation deliberately does not decide.

Explicit edge behavior
----------------------
* Fewer than two TrackStates -> no observations.
* A zero-displacement step (``is_zero_vector`` via the geometry numeric epsilon,
  a numerical fact, not a behavioral threshold) -> no observation for that step.
  This is an *ordinary* gap (a genuinely missing/immobile sample of one track).
* A step with either endpoint outside the lane polygon, or within
  ``boundary_abstain_margin`` of its boundary -> no observation for that step
  (an ordinary gap; see above).
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
from ..geometry import (
    Point,
    Vector,
    angle_between_degrees,
    displacement,
    distance_to_polygon_boundary,
    is_zero_vector,
    point_in_polygon,
)

DEFAULT_PRODUCER = Producer(
    name="wrong-way-heading", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)

#: Provisional default clearance (pixels) a reference point must keep from the
#: lane boundary before its step is allowed to vote. Small on purpose: it guards
#: against localisation jitter at the edge, it is not a policy threshold, and it
#: is stated in pixels because an uncalibrated scene has no metric scale. Sites
#: that need a different band set ``boundary_abstain_margin`` in their scene's
#: ``wrong_way`` parameter block, which is what the pipeline actually passes.
DEFAULT_BOUNDARY_ABSTAIN_MARGIN = 4.0


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


def _confidently_inside(
    track_state: TrackState,
    lane_polygon: Sequence[Point] | None,
    boundary_abstain_margin: float,
) -> bool:
    """Whether this state's reference point is inside the lane, clear of its edge.

    ``None`` polygon means "no lane gating configured" and admits everything, so
    the pure-geometry callers keep their existing behaviour. Otherwise the point
    must be inside *and* at least ``boundary_abstain_margin`` from the boundary;
    a non-positive margin keeps containment and disables the abstain band.
    """

    if lane_polygon is None:
        return True
    point = _center(track_state)
    if not point_in_polygon(point, lane_polygon):
        return False
    if boundary_abstain_margin <= 0.0:
        return True
    return distance_to_polygon_boundary(point, lane_polygon) >= boundary_abstain_margin


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
    lane_polygon: Sequence[Point] | None = None,
    boundary_abstain_margin: float = 0.0,
) -> Iterator[tuple[HeadingVsLaneObservation, bool]]:
    """Yield ``(observation, is_taint_restart)`` for each usable step.

    ``is_taint_restart`` is ``True`` when the observation is the first clean one
    after one or more tainted steps. Zero-displacement and lane-containment
    (ordinary-gap) skips do not set it -- only taint does.
    """

    legal_vector: Vector = (legal_direction.dx, legal_direction.dy)
    legal_heading = _heading_degrees(legal_vector)
    prod = producer if producer is not None else DEFAULT_PRODUCER
    taint_since_last_emit = False

    for previous, current in zip(track, track[1:], strict=False):
        if previous.tainted or current.tainted:
            taint_since_last_emit = True  # abstain on tainted data; mark discontinuity
            continue
        if not (
            _confidently_inside(previous, lane_polygon, boundary_abstain_margin)
            and _confidently_inside(current, lane_polygon, boundary_abstain_margin)
        ):
            # The step did not happen confidently inside the lane whose legal
            # direction governs it, so this derivation has nothing to say about
            # it. Both endpoints are required: a displacement with one end
            # outside is not a movement "inside the lane". An ordinary gap, not
            # a taint discontinuity.
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
    lane_polygon: Sequence[Point] | None = None,
    boundary_abstain_margin: float = 0.0,
) -> list[HeadingVsLaneObservation]:
    """Derive ``HeadingVsLaneObservation`` facts from a TrackState sequence.

    Returns one observation per usable consecutive step, in input order. Steps
    involving a tainted TrackState, with zero displacement, or not confidently
    inside ``lane_polygon`` are skipped. Use
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
        lane_polygon: the polygon of the lane ``lane_id`` names. When given, only
            steps whose endpoints lie confidently inside it are derived. ``None``
            disables lane gating entirely.
        boundary_abstain_margin: clearance a reference point must keep from the
            lane boundary before its step may vote. Ignored when ``lane_polygon``
            is ``None``; ``0.0`` keeps containment without an abstain band.
    """

    return [
        observation
        for observation, _ in _iter_derivation(
            track,
            legal_direction=legal_direction,
            lane_id=lane_id,
            deviation_max_degrees=deviation_max_degrees,
            producer=producer,
            lane_polygon=lane_polygon,
            boundary_abstain_margin=boundary_abstain_margin,
        )
    ]


def derive_heading_observations_with_taint(
    track: Sequence[TrackState],
    *,
    legal_direction: DirectionVector,
    lane_id: str,
    deviation_max_degrees: float,
    producer: Producer | None = None,
    lane_polygon: Sequence[Point] | None = None,
    boundary_abstain_margin: float = 0.0,
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
        lane_polygon=lane_polygon,
        boundary_abstain_margin=boundary_abstain_margin,
    ):
        observations.append(observation)
        if is_restart:
            restart_ids.add(observation.observation_id)
    return HeadingDerivation(tuple(observations), frozenset(restart_ids))

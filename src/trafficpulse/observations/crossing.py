"""Stop-line / junction-entry crossing observation derivation (P3-U4).

Deterministically converts an ordered ``TrackState`` sequence plus a configured
stop line and the junction / signal-controlled zone it guards into
``InZoneObservation`` facts, using only P1-U1 geometry. This is *observation
derivation*, not reasoning: it computes a per-step **validated junction-entry**
membership fact and makes no signal-state, legality, temporal, dedup, or
confirmation decision (all of which belong to the later red-light unit, P3-U5).

No new contract (plan §4a)
--------------------------
The crossing event is carried by the **existing** ``InZoneObservation`` for the
junction / signal-controlled zone: an ``is_inside`` False->True transition is the
entry. ``stop_line_crossing`` geometry is used *inside* this derivation to
validate that an entry follows a **forward** crossing of the stop line in the
configured ``crossing_direction`` (reducing false positives from reversing and
boundary jitter). No ``line_crossing`` observation variant is introduced.

Position source
---------------
Membership and crossing both use the **bbox bottom-center** ``((x1 + x2) / 2,
y2)`` -- the ground-contact reference (architecture-review §17), the same point
P2-U2 in-zone / P2-U3 stationary use, so all per-track evidence streams agree on
where the vehicle *is*.

Validated-entry semantics (why a state flag, not a same-step ``AND``)
--------------------------------------------------------------------
A real stop line and the junction polygon it guards **need not be contiguous**
(in the example scene the stop line sits ~80px in front of the junction zone), so
the forward stop-line crossing and the polygon entry happen on *different* steps.
A per-track ``crossed_forward`` flag bridges them:

* a **forward** stop-line crossing (``side_changed`` and ``intersects_segment``
  and the movement aligned with ``crossing_direction``) sets the flag;
* a **backward** crossing clears it (the track left across the line);
* the emitted ``is_inside`` is ``point_in_polygon(bottom_center, zone) and
  crossed_forward``.

So ``is_inside`` becomes True only when the track occupies the zone **having
entered via a validated forward crossing**; a track that reverses into the zone,
or whose boundary jitter never crosses the finite stop-line segment, never
registers an entry. Membership after a valid entry stays honest (sustained True
while inside). Boundary/edge membership follows the frozen
:func:`~trafficpulse.geometry.point_in_polygon` semantics exactly (boundary counts
as inside).

Two-state minimum, taint handling
----------------------------------
Steps are consecutive ``(previous, current)`` pairs and the observation is emitted
for ``current``; a track shorter than two clean states yields nothing (mirroring
the heading / in-zone / stationary ">= 2 states" shape). A step whose either
endpoint is tainted is skipped, **resets the ``crossed_forward`` flag** (so an
entry can never bridge an ID-switch discontinuity -- architecture-review §13), and
the next clean observation is flagged a **taint restart** via
``CrossingDerivation.taint_restart_ids`` -- reusing the ``HeadingDerivation``
mechanism verbatim. An ordinary missing/dropped sample is not a restart and keeps
its bridging.

Determinism
-----------
Output is a pure function of the inputs: steps are processed in input order, no
wall-clock, no randomness, no set/hash iteration in the emit path, and neither the
``TrackState`` sequence nor the scene inputs are mutated. Both ``True`` and
``False`` facts are emitted, so the reasoner sees the complete stream.
"""

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..contracts import InZoneObservation, Producer, TrackState
from ..contracts.enums import ProducerKind, ZoneKind
from ..contracts.scene import StopLine, Zone, ZoneType
from ..geometry import Point, displacement, dot, point_in_polygon, stop_line_crossing

DEFAULT_CROSSING_PRODUCER = Producer(
    name="junction-crossing", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)

# Scene ``ZoneType`` -> observation ``ZoneKind`` for the zones a stop-line crossing
# can enter: the junction/intersection conflict area and the signal-controlled
# region a signal group governs. Both map to the conflict-area kind (there is no
# distinct signal-controlled ``ZoneKind``, and none is added -- no contract change);
# the truthful zone identity is recorded on ``zone_id``.
_ENTRY_ZONE_KINDS: dict[ZoneType, ZoneKind] = {
    ZoneType.INTERSECTION: ZoneKind.JUNCTION_CONFLICT,
    ZoneType.SIGNAL_CONTROLLED_REGION: ZoneKind.JUNCTION_CONFLICT,
}


@dataclass(frozen=True)
class CrossingDerivation:
    """Observations plus the ids of observations that resume after taint.

    Same ``(observations, taint_restart_ids)`` shape as the heading / in-zone /
    stationary derivations, so it flows through the P3-U3 generalized join and the
    reasoner's taint-restart handling with no new plumbing. ``taint_restart_ids``
    are the ``observation_id``s of clean observations immediately following one or
    more tainted steps; the reasoner resets its run there.
    """

    observations: tuple[InZoneObservation, ...]
    taint_restart_ids: frozenset[str]


def _bottom_center(track_state: TrackState) -> Point:
    """Ground-contact reference point: bbox bottom-center ``((x1+x2)/2, y2)``."""

    box = track_state.bbox
    return ((box.x1 + box.x2) / 2.0, box.y2)


def _observation_id(camera_id: str, track_id: str, zone_id: str, iso_timestamp: str) -> str:
    preimage = "\x1f".join((camera_id, track_id, zone_id, iso_timestamp))
    return "crs-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _entry_zone_kind(zone: Zone) -> ZoneKind:
    """Map an entry zone's scene ``ZoneType`` to its observation ``ZoneKind``.

    Raises:
        ValueError: if ``zone`` is not a junction/intersection or signal-controlled
            region -- the only zone kinds a stop-line crossing enters.
    """

    kind = _ENTRY_ZONE_KINDS.get(zone.zone_type)
    if kind is None:
        raise ValueError(
            f"crossing derivation zone {zone.zone_id!r} must be an intersection or "
            f"signal-controlled region, not {zone.zone_type.value!r}"
        )
    return kind


def _is_forward_crossing(
    previous: Point, current: Point, line_a: Point, line_b: Point, crossing: Point
) -> bool | None:
    """Classify this step's stop-line crossing, if any, against ``crossing_direction``.

    Returns ``True`` for a forward crossing (side changed, the finite segment was
    crossed, and the movement aligns with ``crossing_direction``), ``False`` for a
    backward crossing (same but against the direction), and ``None`` for no
    crossing (the finite segment was not crossed, or no side change).
    """

    fact = stop_line_crossing(previous, current, line_a, line_b)
    if not (fact.side_changed and fact.intersects_segment):
        return None  # not a crossing of the finite stop-line segment
    return dot(displacement(previous, current), crossing) > 0.0


def _iter_derivation(
    track: Sequence[TrackState],
    *,
    stop_line: StopLine,
    zone: Zone,
    producer: Producer | None,
) -> Iterator[tuple[InZoneObservation, bool]]:
    """Yield ``(observation, is_taint_restart)`` for each usable step.

    ``is_taint_restart`` is ``True`` for the first clean observation resuming after
    one or more tainted steps. ``is_inside`` is ``point_in_polygon(bottom_center,
    zone) and crossed_forward``, where ``crossed_forward`` tracks whether the track
    has forward-crossed the stop line and not since reverse-crossed it (reset on
    taint).
    """

    zone_kind = _entry_zone_kind(zone)
    line_a: Point = stop_line.endpoints.a
    line_b: Point = stop_line.endpoints.b
    crossing: Point = (stop_line.crossing_direction.dx, stop_line.crossing_direction.dy)
    prod = producer if producer is not None else DEFAULT_CROSSING_PRODUCER
    taint_since_last_emit = False
    crossed_forward = False

    for previous, current in zip(track, track[1:], strict=False):
        if previous.tainted or current.tainted:
            taint_since_last_emit = True  # abstain on tainted data; mark discontinuity
            crossed_forward = False  # an entry cannot bridge an ID-switch discontinuity
            continue
        prev_pt = _bottom_center(previous)
        curr_pt = _bottom_center(current)
        forward = _is_forward_crossing(prev_pt, curr_pt, line_a, line_b, crossing)
        if forward is not None:
            crossed_forward = forward  # forward crossing sets, backward crossing clears
        is_inside = point_in_polygon(curr_pt, zone.polygon) and crossed_forward
        observation = InZoneObservation(
            observation_id=_observation_id(
                current.camera_id, current.track_id, zone.zone_id, current.timestamp.isoformat()
            ),
            camera_id=current.camera_id,
            track_id=current.track_id,
            timestamp=current.timestamp,
            producer=prod,
            zone_id=zone.zone_id,
            zone_kind=zone_kind,
            is_inside=is_inside,
        )
        yield observation, taint_since_last_emit
        taint_since_last_emit = False


def derive_crossing_observations(
    track: Sequence[TrackState],
    *,
    stop_line: StopLine,
    zone: Zone,
    producer: Producer | None = None,
) -> list[InZoneObservation]:
    """Derive junction-entry ``InZoneObservation`` facts from a TrackState sequence.

    Returns one observation per usable consecutive step, in input order, emitted
    for the *current* sample; a track shorter than two clean states yields nothing.
    Tainted steps are skipped and reset the validated-entry flag. Use
    :func:`derive_crossing_observations_with_taint` when the taint-discontinuity
    markers are needed for reasoning.

    Args:
        track: ordered TrackStates for a single ``(camera_id, track_id)`` (as
            produced by the P1-U2 synth source or the real tracker).
        stop_line: the configured stop line whose forward crossing (in its
            ``crossing_direction``) gates the entry.
        zone: the junction / signal-controlled ``Zone`` the crossing enters; only
            its ``polygon`` (bottom-center membership), ``zone_id``, and
            ``zone_type`` are used.
        producer: observation provenance (defaults to a synthetic heuristic).

    Raises:
        ValueError: if ``zone`` is not an intersection or signal-controlled region.
    """

    return [
        observation
        for observation, _ in _iter_derivation(
            track, stop_line=stop_line, zone=zone, producer=producer
        )
    ]


def derive_crossing_observations_with_taint(
    track: Sequence[TrackState],
    *,
    stop_line: StopLine,
    zone: Zone,
    producer: Producer | None = None,
) -> CrossingDerivation:
    """Like :func:`derive_crossing_observations`, but also return taint restarts.

    The returned ``taint_restart_ids`` name the observations that resume after a
    tainted interval; the reasoning layer resets its persistence run there so a
    junction entry cannot bridge the tainted (ID-switch) discontinuity.

    Raises:
        ValueError: if ``zone`` is not an intersection or signal-controlled region.
    """

    observations: list[InZoneObservation] = []
    restart_ids: set[str] = set()
    for observation, is_restart in _iter_derivation(
        track, stop_line=stop_line, zone=zone, producer=producer
    ):
        observations.append(observation)
        if is_restart:
            restart_ids.add(observation.observation_id)
    return CrossingDerivation(tuple(observations), frozenset(restart_ids))

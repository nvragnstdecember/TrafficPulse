"""In-zone observation derivation (P2-U2).

Deterministically converts an ordered ``TrackState`` sequence plus configured
scene zones into ``InZoneObservation`` facts, using only P1-U1 geometry. This is
*observation derivation*, not reasoning: it computes a per-step geometric
membership fact (``is_inside``) for each eligible zone and makes no violation,
dwell, stationarity, persistence, or confirmation decision.

Position source
---------------
Membership uses the **bbox bottom-center** ``((x1 + x2) / 2, y2)`` -- the
ground-contact point (architecture-review §17 names bbox bottom-center as the
ground-plane reference). For "is the vehicle *in* this zone" the wheels' contact
point is the defensible choice. This deliberately differs from the heading
derivation's bbox-*center*: heading needs a displacement-direction-invariant
reference, whereas zone membership needs ground contact. The choice is
provisional and revisitable once calibrated ground-plane reasoning exists.

Eligible zones
--------------
First slice: **no-stopping only**. A scene ``ZoneType.NO_STOPPING`` maps to the
observation ``ZoneKind.NO_STOPPING`` (both wire the value ``"no_stopping"``);
disabled zones and every other zone type are ignored. The derivation makes no
fail-fast decision when no eligible zone exists -- it simply emits nothing; the
illegal-stopping pipeline (a later unit) owns any scene-configuration guard.

Emission
--------
One ``InZoneObservation`` per (usable step, eligible zone), carrying ``zone_id``,
``zone_kind``, and ``is_inside`` (a deterministic boolean; boundary points count
as inside per :func:`trafficpulse.geometry.point_in_polygon`). Both ``True`` and
``False`` memberships are emitted -- the illegal-stopping reasoner joins in-zone
with stationarity later, so it needs the negative facts too. Unlike heading,
there is **no** zero-displacement skip: a stationary vehicle sitting inside a
zone must still produce in-zone facts.

Two-state minimum, taint handling
----------------------------------
Steps are consecutive ``(previous, current)`` pairs and the observation is
emitted for ``current``; a track shorter than two states therefore yields
nothing (mirroring the heading derivation's ">= 2 states" shape). A step whose
either endpoint is a tainted ``TrackState`` is skipped, and every observation of
the next clean step is flagged as a **taint restart** via
``InZoneDerivation.taint_restart_ids`` -- reusing the ``HeadingDerivation``
mechanism verbatim so downstream reasoning can never bridge an ID-switch
discontinuity (architecture-review §13: tainted tracks may abstain but never
confirm). An ordinary missing/dropped sample is not a restart.

Determinism
-----------
Output is a pure function of the inputs: steps are processed in input order and,
within each step, eligible zones in input (scene-declaration) order. No
wall-clock, no randomness, no set/hash iteration in the emit path, and neither
the ``TrackState`` sequence nor the ``Zone`` inputs are mutated.
"""

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..contracts import InZoneObservation, Producer, TrackState
from ..contracts.enums import ProducerKind, ZoneKind
from ..contracts.scene import Zone, ZoneType
from ..geometry import Point, point_in_polygon

DEFAULT_IN_ZONE_PRODUCER = Producer(
    name="in-zone", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)

# Scene ``ZoneType`` -> observation ``ZoneKind`` for the kinds P2-U2 derives
# membership for. First slice: no-stopping only (plan §8a B.4). Both wire the
# value ``"no_stopping"``. Adding a kind here (e.g. parking) is a scoped, later
# decision -- the scene ``ZoneType`` closed set has no ``parking`` member today.
_ELIGIBLE_ZONE_KINDS: dict[ZoneType, ZoneKind] = {
    ZoneType.NO_STOPPING: ZoneKind.NO_STOPPING,
}


@dataclass(frozen=True)
class InZoneDerivation:
    """Observations plus the ids of observations that resume after taint.

    ``taint_restart_ids`` are the ``observation_id``s of clean observations that
    immediately follow one or more tainted steps; the reasoner treats them as
    explicit discontinuities and resets its persistence run there.
    """

    observations: tuple[InZoneObservation, ...]
    taint_restart_ids: frozenset[str]


def _bottom_center(track_state: TrackState) -> Point:
    """Ground-contact reference point: bbox bottom-center ``((x1+x2)/2, y2)``."""

    box = track_state.bbox
    return ((box.x1 + box.x2) / 2.0, box.y2)


def _observation_id(camera_id: str, track_id: str, zone_id: str, iso_timestamp: str) -> str:
    preimage = "\x1f".join((camera_id, track_id, zone_id, iso_timestamp))
    return "inz-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _eligible_targets(zones: Sequence[Zone]) -> list[tuple[Zone, ZoneKind]]:
    """Select the enabled, in-scope zones with their mapped observation kind.

    Input (scene-declaration) order is preserved so multi-zone emission is
    deterministic. A disabled zone or a zone whose ``zone_type`` is not in the
    first-slice eligible set contributes nothing.
    """

    return [
        (zone, _ELIGIBLE_ZONE_KINDS[zone.zone_type])
        for zone in zones
        if zone.enabled and zone.zone_type in _ELIGIBLE_ZONE_KINDS
    ]


def _iter_derivation(
    track: Sequence[TrackState],
    *,
    zones: Sequence[Zone],
    producer: Producer | None,
) -> Iterator[tuple[InZoneObservation, bool]]:
    """Yield ``(observation, is_taint_restart)`` for each usable step x zone.

    ``is_taint_restart`` is ``True`` for every observation of the first clean
    step after one or more tainted steps. Membership is evaluated on ``current``'s
    bottom-center; there is no zero-displacement skip (a stationary in-zone track
    must still emit).
    """

    targets = _eligible_targets(zones)
    if not targets:
        return
    prod = producer if producer is not None else DEFAULT_IN_ZONE_PRODUCER
    taint_since_last_emit = False

    for previous, current in zip(track, track[1:], strict=False):
        if previous.tainted or current.tainted:
            taint_since_last_emit = True  # abstain on tainted data; mark discontinuity
            continue
        point = _bottom_center(current)
        iso_timestamp = current.timestamp.isoformat()
        for zone, zone_kind in targets:
            observation = InZoneObservation(
                observation_id=_observation_id(
                    current.camera_id, current.track_id, zone.zone_id, iso_timestamp
                ),
                camera_id=current.camera_id,
                track_id=current.track_id,
                timestamp=current.timestamp,
                producer=prod,
                zone_id=zone.zone_id,
                zone_kind=zone_kind,
                is_inside=point_in_polygon(point, zone.polygon),
            )
            yield observation, taint_since_last_emit
        taint_since_last_emit = False


def derive_in_zone_observations(
    track: Sequence[TrackState],
    *,
    zones: Sequence[Zone],
    producer: Producer | None = None,
) -> list[InZoneObservation]:
    """Derive ``InZoneObservation`` facts from a TrackState sequence.

    Returns one observation per usable consecutive step and per eligible
    (enabled, no-stopping) zone, in input order. Steps involving a tainted
    TrackState are skipped. Use :func:`derive_in_zone_observations_with_taint`
    when the taint-discontinuity markers are needed for reasoning.

    Args:
        track: ordered TrackStates (as produced by the P1-U2 synth source or the
            real tracker), grouped to a single ``(camera_id, track_id)``.
        zones: configured scene zones (e.g. ``scene.zones``); only enabled
            no-stopping zones contribute observations.
        producer: observation provenance (defaults to a synthetic heuristic).
    """

    return [
        observation
        for observation, _ in _iter_derivation(track, zones=zones, producer=producer)
    ]


def derive_in_zone_observations_with_taint(
    track: Sequence[TrackState],
    *,
    zones: Sequence[Zone],
    producer: Producer | None = None,
) -> InZoneDerivation:
    """Like :func:`derive_in_zone_observations`, but also return taint restarts.

    The returned ``taint_restart_ids`` name the observations that resume after a
    tainted interval; the reasoning layer resets its persistence run there so
    support cannot bridge the tainted (ID-switch) discontinuity.
    """

    observations: list[InZoneObservation] = []
    restart_ids: set[str] = set()
    for observation, is_restart in _iter_derivation(track, zones=zones, producer=producer):
        observations.append(observation)
        if is_restart:
            restart_ids.add(observation.observation_id)
    return InZoneDerivation(tuple(observations), frozenset(restart_ids))

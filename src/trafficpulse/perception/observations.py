"""Motorcycle-perception observation contracts (v1.1 U1).

The reusable perception-layer aggregates that future motorcycle rule engines
(no-helmet, triple-riding) consume. They sit *between* the frozen
``Detection -> TrackState -> Association`` flow and the reasoning-boundary
``Observation`` union: they package the existing per-frame tracks + rider
associations into a stable, explicit, per-motorcycle view, without making any
rule decision.

They are **not** part of the frozen Phase 0-F reasoning ``Observation`` union
(that set stays fixed at its seven variants); they are a distinct perception
family, colocated with the derivation that produces them. Each is a frozen,
strict :class:`~trafficpulse.contracts.primitives.ContractModel`, so it
serializes, round-trips, and persists exactly like every other contract — which
is what keeps it evidence- and workspace-compatible.

Scope note: this unit establishes *perception* (which motorcycle, where, with
which riders). It deliberately introduces **no** helmet-, count-, or
speed-specific fields — those belong to the reasoning observations already in the
frozen union and to the future units that populate them.
"""

from pydantic import AwareDatetime

from ..contracts import Producer
from ..contracts.primitives import (
    BoundingBox,
    Confidence,
    ContractModel,
    NonEmptyStr,
    NonNegativeInt,
)


class MotorcycleObservation(ContractModel):
    """One tracked motorcycle in one frame, with its associated riders.

    This is the per-frame answer to "which motorcycle is here, and who is on it".
    ``rider_track_ids`` are the person track ids the association layer linked to
    this motorcycle in this frame (empty when none) — the riders a future rule
    engine reasons about. ``bbox`` / ``confidence`` come from the motorcycle's own
    ``TrackState``.
    """

    observation_id: NonEmptyStr
    camera_id: NonEmptyStr
    motorcycle_track_id: NonEmptyStr
    timestamp: AwareDatetime
    frame_index: NonNegativeInt | None = None
    bbox: BoundingBox
    confidence: Confidence | None = None
    rider_track_ids: tuple[NonEmptyStr, ...] = ()
    producer: Producer

    @property
    def rider_count(self) -> int:
        """Number of riders associated with this motorcycle in this frame."""

        return len(self.rider_track_ids)


class RiderObservation(ContractModel):
    """One rider associated with one motorcycle in one frame.

    Emitted only for a *person* track the association layer linked to a
    motorcycle. ``association_confidence`` is the geometric overlap plausibility
    from the association layer (``IoMin``) — never a calibrated probability.
    ``rider_index`` is a **stable ordinal** within the motorcycle's rider set for
    this frame (ordered by rider track id); it is *not* a driver/pillion claim —
    front-vs-back attribution needs geometry this unit does not yet compute, so
    slot semantics are left to a later, calibrated unit.
    """

    observation_id: NonEmptyStr
    camera_id: NonEmptyStr
    rider_track_id: NonEmptyStr
    motorcycle_track_id: NonEmptyStr
    timestamp: AwareDatetime
    frame_index: NonNegativeInt | None = None
    bbox: BoundingBox
    confidence: Confidence | None = None
    association_confidence: Confidence
    rider_index: NonNegativeInt
    producer: Producer


class MotorcycleTrackObservation(ContractModel):
    """A temporal summary of one stable motorcycle track across frames.

    Aggregates a motorcycle's per-frame :class:`MotorcycleObservation`s into the
    track-level facts a future rule engine needs: how long the (stable-id)
    motorcycle was seen, the peak rider count over its life, and the union of
    every rider ever associated with it. It makes no violation decision — a
    sustained-count or helmet conclusion is reasoning-layer work.
    """

    observation_id: NonEmptyStr
    camera_id: NonEmptyStr
    motorcycle_track_id: NonEmptyStr
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    first_frame_index: NonNegativeInt | None = None
    last_frame_index: NonNegativeInt | None = None
    frame_count: NonNegativeInt
    max_rider_count: NonNegativeInt
    associated_rider_track_ids: tuple[NonEmptyStr, ...] = ()
    producer: Producer

"""Rider-count observation derivation (v1.1 U3).

Turns one frame's frozen ``TrackState``s into ``RiderCountObservation`` facts --
one per tracked motorcycle, carrying how many riders were associated with it that
frame -- for the triple-riding reasoner to reason over temporally.

Reuse, not a second counting path
---------------------------------
The count is **not** recomputed here. It is read from the v1.1 U1 motorcycle
perception layer: :func:`~trafficpulse.perception.derive_perception_frame` already
associates riders to motorcycles (reusing the P4-U4
:func:`~trafficpulse.association.associate_riders` derivation) and exposes
``MotorcycleObservation.rider_count``. This module only *restamps* that count as
the frozen ``RiderCountObservation`` the reasoning layer consumes, and carries the
frame's rider↔motorcycle ``Association``s so the reasoner can name the riders on a
confirmed event. It performs no detection, tracking, or counting of its own and
reads no pixels.

Keyed by the motorcycle
-----------------------
Triple riding is a property of the *vehicle*, so each ``RiderCountObservation`` is
keyed by the **motorcycle** track (``track_id`` = ``motorcycle_track_id``); the
reasoner groups by that track. A motorcycle with no riders still emits (count 0),
so the reasoner sees the true count trajectory and a run ends when the count drops
below the threshold.

Determinism
-----------
A pure function of the input states (perception is deterministic and
order-independent); ids are content-derived (sha256 of camera/motorcycle/
timestamp); no wall-clock, no randomness. Tainted motorcycles abstain (the
perception layer skips them), so a count never bridges an ID switch.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from ..association import RiderAssociationConfig
from ..contracts import Association, Producer, RiderCountObservation, TrackState
from ..contracts.enums import ProducerKind
from ..perception import derive_perception_frame

DEFAULT_RIDER_COUNT_PRODUCER = Producer(
    name="rider-count", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)


@dataclass(frozen=True)
class RiderCountDerivation:
    """Rider-count observations plus the rider↔motorcycle links (and taint restarts).

    ``associations`` is carried (not folded into the observations) so the reasoner
    can attribute the specific rider tracks to a confirmed triple-riding event
    without the count observation having to duplicate rider identity.
    ``taint_restart_ids`` is empty from the per-frame derivation (taint is a
    cross-frame notion) and populated by the accumulating frame observer, so the
    reasoner never bridges an ID switch.
    """

    observations: tuple[RiderCountObservation, ...]
    associations: tuple[Association, ...]
    taint_restart_ids: frozenset[str] = frozenset()


def _observation_id(camera_id: str, motorcycle_track_id: str, iso_timestamp: str) -> str:
    preimage = "\x1f".join((camera_id, motorcycle_track_id, iso_timestamp))
    return "rct-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def derive_rider_count_observations(
    states: Sequence[TrackState],
    *,
    association_config: RiderAssociationConfig | None = None,
    producer: Producer | None = None,
) -> RiderCountDerivation:
    """Derive one frame's ``RiderCountObservation`` facts from its track states.

    Args:
        states: the ``TrackState``s of a single frame (as the tracker emits them).
            Non-motorcycle / non-person classes are ignored; tainted tracks
            abstain (both handled by the reused perception layer).
        association_config: rider↔motorcycle association policy, forwarded to the
            perception layer (defaults apply when omitted).
        producer: observation provenance (defaults to the provisional heuristic).

    Returns:
        A :class:`RiderCountDerivation`: one observation per tracked motorcycle
        (keyed by the motorcycle track, count = associated riders) plus the
        frame's rider↔motorcycle associations, in a deterministic order.
    """

    prod = producer if producer is not None else DEFAULT_RIDER_COUNT_PRODUCER
    frame = derive_perception_frame(states, association_config=association_config)
    observations = tuple(
        RiderCountObservation(
            observation_id=_observation_id(
                motorcycle.camera_id,
                motorcycle.motorcycle_track_id,
                motorcycle.timestamp.isoformat(),
            ),
            camera_id=motorcycle.camera_id,
            track_id=motorcycle.motorcycle_track_id,
            timestamp=motorcycle.timestamp,
            producer=prod,
            rider_count=motorcycle.rider_count,
            motorcycle_track_id=motorcycle.motorcycle_track_id,
        )
        for motorcycle in frame.motorcycles
    )
    return RiderCountDerivation(observations=observations, associations=frame.associations)

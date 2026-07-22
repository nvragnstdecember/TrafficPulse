"""Motorcycle perception derivation (v1.1 U1).

Turns one frame's frozen ``TrackState``s into the perception aggregates future
motorcycle rule engines consume — :class:`MotorcycleObservation`,
:class:`RiderObservation` — and folds a track's per-frame observations into a
:class:`MotorcycleTrackObservation`.

Reuse, not a new inference path
-------------------------------
This layer performs **no** detection or tracking and reads **no** pixels. Motorcycle
and rider *detection* are the existing ``Detector``'s output (``ObjectClass.MOTORCYCLE``
and ``ObjectClass.PERSON``); stable track ids are the existing IoU ``Tracker``'s
output; the rider↔motorcycle link is the existing
:func:`~trafficpulse.association.associate_riders` derivation (P4-U4), reused
verbatim. This module only *aggregates* those frozen outputs, exactly as the
``observations`` and ``association`` derivations do — one shared inference path,
no duplication.

Determinism
-----------
Output is a pure function of the input states: motorcycles are emitted in
``motorcycle_track_id`` order, riders in ``rider_track_id`` order within each
motorcycle, ids are content-derived (no wall-clock, no counter), and nothing is
mutated. Tainted tracks abstain on both sides (via the association layer and an
explicit motorcycle filter), so perception never bridges an ID-switch.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from ..association import RiderAssociationConfig, associate_riders
from ..contracts import Association, Producer, TrackState
from ..contracts.enums import ObjectClass, ProducerKind
from .observations import (
    MotorcycleObservation,
    MotorcycleTrackObservation,
    RiderObservation,
)

DEFAULT_PERCEPTION_PRODUCER = Producer(
    name="motorcycle-perception", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)


@dataclass(frozen=True)
class PerceptionFrame:
    """The perception aggregates derived from a single frame's track states.

    ``motorcycles`` has one entry per tracked motorcycle in the frame (with or
    without riders); ``riders`` has one entry per associated rider; and
    ``associations`` is the reused frozen ``Association`` output — carried so the
    evidence layer can reference the link explicitly.
    """

    motorcycles: tuple[MotorcycleObservation, ...]
    riders: tuple[RiderObservation, ...]
    associations: tuple[Association, ...]


def _digest(prefix: str, *parts: str) -> str:
    preimage = "\x1f".join(parts)
    return prefix + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def derive_perception_frame(
    states: Sequence[TrackState],
    *,
    association_config: RiderAssociationConfig | None = None,
    producer: Producer | None = None,
) -> PerceptionFrame:
    """Derive the per-frame motorcycle/rider observations for one frame.

    Args:
        states: the ``TrackState``s of a single frame (as the tracker emits them).
            Non-motorcycle / non-person classes are ignored; tainted tracks
            abstain.
        association_config: rider↔motorcycle association policy, forwarded to
            :func:`~trafficpulse.association.associate_riders` (defaults apply
            when omitted).
        producer: provenance for the emitted observations (defaults to the
            provisional heuristic producer).

    Returns:
        A :class:`PerceptionFrame`: one :class:`MotorcycleObservation` per tracked
        motorcycle (riders attached), one :class:`RiderObservation` per associated
        rider, and the reused ``Association`` tuple — all in a deterministic order.
    """

    prod = producer if producer is not None else DEFAULT_PERCEPTION_PRODUCER
    associations = associate_riders(states, config=association_config)

    # Riders linked to each motorcycle: (rider_track_id, overlap_confidence).
    riders_by_motorcycle: dict[str, list[tuple[str, float]]] = {}
    for association in associations:
        riders_by_motorcycle.setdefault(association.object_track_id, []).append(
            (association.subject_track_id, association.confidence)
        )

    state_by_id = {state.track_id: state for state in states}
    motorcycles = sorted(
        (
            state
            for state in states
            if state.object_class is ObjectClass.MOTORCYCLE and not state.tainted
        ),
        key=lambda state: state.track_id,
    )

    motorcycle_obs: list[MotorcycleObservation] = []
    rider_obs: list[RiderObservation] = []
    for motorcycle in motorcycles:
        linked = sorted(riders_by_motorcycle.get(motorcycle.track_id, ()), key=lambda pair: pair[0])
        motorcycle_obs.append(
            MotorcycleObservation(
                observation_id=_digest(
                    "mot-",
                    motorcycle.camera_id,
                    motorcycle.track_id,
                    motorcycle.timestamp.isoformat(),
                ),
                camera_id=motorcycle.camera_id,
                motorcycle_track_id=motorcycle.track_id,
                timestamp=motorcycle.timestamp,
                frame_index=motorcycle.frame_index,
                bbox=motorcycle.bbox,
                confidence=motorcycle.confidence,
                rider_track_ids=tuple(rider_id for rider_id, _ in linked),
                producer=prod,
            )
        )
        for rider_index, (rider_id, overlap) in enumerate(linked):
            rider_state = state_by_id.get(rider_id)
            if rider_state is None:  # pragma: no cover - association guarantees membership
                continue
            rider_obs.append(
                RiderObservation(
                    observation_id=_digest(
                        "rid-",
                        rider_state.camera_id,
                        rider_id,
                        motorcycle.track_id,
                        rider_state.timestamp.isoformat(),
                    ),
                    camera_id=rider_state.camera_id,
                    rider_track_id=rider_id,
                    motorcycle_track_id=motorcycle.track_id,
                    timestamp=rider_state.timestamp,
                    frame_index=rider_state.frame_index,
                    bbox=rider_state.bbox,
                    confidence=rider_state.confidence,
                    association_confidence=overlap,
                    rider_index=rider_index,
                    producer=prod,
                )
            )

    return PerceptionFrame(tuple(motorcycle_obs), tuple(rider_obs), associations)


def summarize_motorcycle_track(
    observations: Sequence[MotorcycleObservation],
    *,
    producer: Producer | None = None,
) -> MotorcycleTrackObservation:
    """Fold one motorcycle's per-frame observations into a track-level summary.

    Args:
        observations: the :class:`MotorcycleObservation`s of a **single**
            ``motorcycle_track_id`` (in any order — they are sorted by time here).
        producer: provenance (defaults to the provisional heuristic producer).

    Returns:
        A :class:`MotorcycleTrackObservation` with the track's lifespan, frame
        count, peak rider count, and the union of every associated rider.

    Raises:
        ValueError: if ``observations`` is empty or mixes multiple motorcycle or
            camera ids (the summary is defined for one stable track).
    """

    if not observations:
        raise ValueError("summarize_motorcycle_track requires at least one observation")
    motorcycle_ids = {obs.motorcycle_track_id for obs in observations}
    camera_ids = {obs.camera_id for obs in observations}
    if len(motorcycle_ids) != 1 or len(camera_ids) != 1:
        raise ValueError("all observations must share one motorcycle_track_id and camera_id")

    prod = producer if producer is not None else DEFAULT_PERCEPTION_PRODUCER
    ordered = sorted(observations, key=lambda obs: (obs.timestamp, obs.frame_index or 0))
    first, last = ordered[0], ordered[-1]
    rider_union = sorted({rider for obs in ordered for rider in obs.rider_track_ids})

    return MotorcycleTrackObservation(
        observation_id=_digest(
            "mtk-",
            first.camera_id,
            first.motorcycle_track_id,
            first.timestamp.isoformat(),
            last.timestamp.isoformat(),
        ),
        camera_id=first.camera_id,
        motorcycle_track_id=first.motorcycle_track_id,
        first_seen=first.timestamp,
        last_seen=last.timestamp,
        first_frame_index=first.frame_index,
        last_frame_index=last.frame_index,
        frame_count=len(ordered),
        max_rider_count=max(obs.rider_count for obs in ordered),
        associated_rider_track_ids=tuple(rider_union),
        producer=prod,
    )


def summarize_motorcycle_tracks(
    frames: Sequence[PerceptionFrame],
    *,
    producer: Producer | None = None,
) -> tuple[MotorcycleTrackObservation, ...]:
    """Summarize every motorcycle track across a run's per-frame perception.

    Groups each frame's :class:`MotorcycleObservation`s by
    ``motorcycle_track_id`` and folds each group with
    :func:`summarize_motorcycle_track`. Returns the summaries sorted by
    ``motorcycle_track_id`` (deterministic, order-independent).
    """

    by_track: dict[str, list[MotorcycleObservation]] = {}
    for frame in frames:
        for obs in frame.motorcycles:
            by_track.setdefault(obs.motorcycle_track_id, []).append(obs)
    return tuple(
        summarize_motorcycle_track(group, producer=producer)
        for _, group in sorted(by_track.items())
    )

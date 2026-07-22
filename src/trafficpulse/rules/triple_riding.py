"""Triple-riding temporal reasoning and confirmation (v1.1 U3).

Consumes ``RiderCountObservation`` facts (from ``observations.rider_count``, which
reuses the v1.1 U1 motorcycle perception + P4-U4 association) keyed by the
**motorcycle** track, drives the generic P1-U3 ``RuleEngine`` for hypothesis
lifecycle mechanics through the shared P3-U1 ``TemporalRunReasoner``, and mints
frozen U2 ``ConfirmedEvent`` objects when a motorcycle carries at least the
threshold number of riders for long enough.

**Reasoning begins here; perception ended at the observation stream.** This module
imports no detector, tracker, or perception backend, touches no pixels, and holds
no model. It reads frozen contracts only, so the decision replays deterministically
from the observation log (architecture-review §14/§15).

Separation of concerns
----------------------
This module owns only triple-riding *semantics*: an observation supports the
violation iff its ``rider_count`` meets the configured threshold, and support must
persist. Everything else -- ids, transition validation, run tracking, taint reset,
content-derived ``event_id`` -- is delegated to the ``TemporalRunReasoner`` this
reasoner *holds*, exactly as the wrong-way, illegal-stopping, and no-helmet
reasoners do. There is no second rule engine.

Temporal consistency (the point of the rule; see the plan's anti-flicker goal)
------------------------------------------------------------------------------
A single frame counting three riders is **not** a violation: detector noise routinely
splits/merges overlapping rider boxes, so the per-frame count flickers 2→3→2. The
base takes a boolean per-step signal -- ``rider_count >= threshold`` is *active*,
below is *inactive* -- and confirms only when active support persists for at least
``min_persistence`` seconds. An isolated 3-rider frame between 2-rider frames opens
a run that the very next inactive step ends before persistence is reached, so it
never confirms. ``max_observation_gap`` bounds how long a run may bridge a dropped
observation, so a brief occlusion does not silently sustain a stale run.

Attribution: the motorcycle is the episode, the riders are named on the event
-----------------------------------------------------------------------------
The count is a property of the vehicle, so episodes are keyed by the motorcycle
track and ``RiderCountObservation.track_id`` names it. The specific rider tracks
are read from the frozen ``Association`` links in the confirmed window and added to
the event's ``track_ids`` (via the base's ``episode_enricher`` hook, before
``event_id`` is computed, because ``track_ids`` is identity-bearing). A confirmed
event therefore names the motorcycle **and** the riders it carried.

Taint
-----
Taint restarts are honoured verbatim through the base: a restart ends the open run
before the step is processed, so support never accumulates across an ID-switch
discontinuity (§13). A confirmed run is taint-free by construction.

Parameters (provisional, from configuration)
--------------------------------------------
``triple_riding_parameters(scene)`` reads the ``triple_riding`` block:
``min_persistence`` (seconds) is required; ``rider_count_threshold`` (count,
default 3 -- the ontology's definition) and ``max_observation_gap`` (seconds) are
optional. Persistence is expressed in seconds, not frames, for the same
frame-rate-independence reason no-helmet documents.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..contracts import (
    Association,
    ConfidenceBreakdown,
    ConfirmedEvent,
    MeasuredValue,
    ModelRef,
    ParameterStatus,
    RiderCountObservation,
    SceneConfig,
)
from ..contracts.enums import AssociationType, ViolationType
from ..observations.rider_count import RiderCountDerivation
from .engine import RuleEngine
from .temporal import ConfirmationDetails, EpisodeExtras, TemporalRunReasoner

RULE_ID = "triple_riding"
RULE_VERSION = "0.1.0-provisional"

# The ontology defines triple riding as three or more riders; used when the scene
# omits an explicit threshold.
DEFAULT_RIDER_COUNT_THRESHOLD = 3


# --- configuration -----------------------------------------------------------
@dataclass(frozen=True)
class TripleRidingParameters:
    """Provisional, scene-specific triple-riding parameters loaded from config."""

    min_persistence_seconds: float
    rider_count_threshold: int
    max_observation_gap_seconds: float | None
    persistence_status: ParameterStatus
    rider_count_threshold_status: ParameterStatus
    max_observation_gap_status: ParameterStatus


def triple_riding_parameters(scene: SceneConfig) -> TripleRidingParameters:
    """Load the triple-riding parameter block from a U5 ``SceneConfig``.

    Raises:
        ValueError: if the scene declares no ``triple_riding`` block, if
            ``min_persistence`` is absent or unset (reasoning cannot proceed
            without it), or if an explicit ``rider_count_threshold`` is below 1.
    """

    block = next(
        (b for b in scene.rule_parameters if b.violation_type is ViolationType.TRIPLE_RIDING),
        None,
    )
    if block is None:
        raise ValueError("scene has no triple_riding rule-parameter block")
    by_id = {p.id: p for p in block.parameters}
    persistence = by_id.get("min_persistence")
    if persistence is None or persistence.value is None:
        raise ValueError("triple_riding min_persistence is unset")

    threshold_param = by_id.get("rider_count_threshold")
    threshold = (
        int(threshold_param.value)
        if threshold_param is not None and threshold_param.value is not None
        else DEFAULT_RIDER_COUNT_THRESHOLD
    )
    if threshold < 1:
        raise ValueError("triple_riding rider_count_threshold must be >= 1")

    max_gap = by_id.get("max_observation_gap")
    return TripleRidingParameters(
        min_persistence_seconds=persistence.value,
        rider_count_threshold=threshold,
        max_observation_gap_seconds=(
            max_gap.value if max_gap is not None and max_gap.value is not None else None
        ),
        persistence_status=persistence.status,
        rider_count_threshold_status=(
            threshold_param.status if threshold_param is not None else ParameterStatus.UNSET
        ),
        max_observation_gap_status=(
            max_gap.status if max_gap is not None else ParameterStatus.UNSET
        ),
    )


class TripleRidingReasoner:
    """Deterministic triple-riding temporal reasoner over ``RiderCountObservation``.

    Triple-riding *semantics* (the count threshold, confidence aggregation, rider
    attribution) live here; all lifecycle mechanics are delegated to the shared
    :class:`TemporalRunReasoner` this reasoner holds, mirroring the sibling
    reasoners. Reasoning is per **motorcycle** track.
    """

    def __init__(
        self,
        engine: RuleEngine,
        params: TripleRidingParameters,
        *,
        scene_config_hash: str | None = None,
        rule_id: str = RULE_ID,
        rule_version: str | None = RULE_VERSION,
        models: tuple[ModelRef, ...] = (),
    ) -> None:
        self._params = params
        # Per-motorcycle indices, rebuilt on each run() call; read by the enricher.
        self._by_motorcycle: dict[str, list[RiderCountObservation]] = {}
        self._riders_by_motorcycle: dict[str, list[Association]] = {}
        self._machine = TemporalRunReasoner(
            engine,
            violation_type=ViolationType.TRIPLE_RIDING,
            threshold_seconds=params.min_persistence_seconds,
            detail_builder=self._details,
            scene_config_hash=scene_config_hash,
            rule_id=rule_id,
            rule_version=rule_version,
            models=models,
            max_observation_gap_seconds=params.max_observation_gap_seconds,
            episode_enricher=self._enrich,
        )

    @property
    def engine(self) -> RuleEngine:
        return self._machine.engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return self._machine.events

    def run(
        self,
        observations: Iterable[RiderCountObservation],
        *,
        associations: Iterable[Association] = (),
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Reason over a run's rider-count observations; return confirmed events.

        Observations are processed in ``(timestamp, observation_id)`` order and
        de-duplicated by the base, so the outcome is independent of input order.
        ``associations`` supply the rider identities named on a confirmed event.
        """

        collected = list(observations)
        self._index(collected, associations)
        threshold = self._params.rider_count_threshold
        steps = [
            (observation, observation.rider_count >= threshold)
            for observation in collected
            if observation.track_id is not None
        ]
        return self._machine.run(steps, taint_restart_ids=frozenset(taint_restart_ids))

    def run_derivation(self, derivation: RiderCountDerivation) -> tuple[ConfirmedEvent, ...]:
        """Convenience: run a :class:`RiderCountDerivation` with its associations."""

        return self.run(
            derivation.observations,
            associations=derivation.associations,
            taint_restart_ids=derivation.taint_restart_ids,
        )

    # --- indexing ------------------------------------------------------------
    def _index(
        self,
        observations: Sequence[RiderCountObservation],
        associations: Iterable[Association],
    ) -> None:
        self._by_motorcycle = {}
        self._riders_by_motorcycle = {}
        for observation in observations:
            if observation.track_id is not None:
                self._by_motorcycle.setdefault(observation.track_id, []).append(observation)
        for association in associations:
            if association.association_type is AssociationType.RIDER_OF_MOTORCYCLE:
                self._riders_by_motorcycle.setdefault(association.object_track_id, []).append(
                    association
                )

    def _window(
        self, motorcycle_track_id: str, start_at: datetime, trigger_at: datetime
    ) -> list[RiderCountObservation]:
        return [
            observation
            for observation in self._by_motorcycle.get(motorcycle_track_id, ())
            if start_at <= observation.timestamp <= trigger_at
        ]

    def _riders(
        self, motorcycle_track_id: str, start_at: datetime, trigger_at: datetime
    ) -> tuple[tuple[str, ...], float | None]:
        """The rider tracks linked in the window, and the weakest link to them.

        Returns ``((), None)`` when no association was recorded in the window --
        never invents a rider.
        """

        in_window = [
            association
            for association in self._riders_by_motorcycle.get(motorcycle_track_id, ())
            if start_at <= association.timestamp <= trigger_at
        ]
        if not in_window:
            return (), None
        rider_ids = tuple(sorted({a.subject_track_id for a in in_window}))
        # The weakest rider<->motorcycle overlap in the window: a conservative
        # summary of how well-established the links were, never the strongest.
        weakest = min(association.confidence for association in in_window)
        return rider_ids, weakest

    # --- injected policies ---------------------------------------------------
    def _details(self, start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
        thresholds = [
            MeasuredValue(
                name="min_persistence",
                value=self._params.min_persistence_seconds,
                unit="seconds",
            ),
            MeasuredValue(
                name="rider_count_threshold",
                value=float(self._params.rider_count_threshold),
                unit="count",
            ),
        ]
        if self._params.max_observation_gap_seconds is not None:
            thresholds.append(
                MeasuredValue(
                    name="max_observation_gap",
                    value=self._params.max_observation_gap_seconds,
                    unit="seconds",
                )
            )
        return ConfirmationDetails(
            measurements=(
                MeasuredValue(
                    name="persistence_seconds",
                    value=(trigger_at - start_at).total_seconds(),
                    unit="seconds",
                ),
            ),
            thresholds=tuple(thresholds),
        )

    def _enrich(
        self, motorcycle_track_id: str, start_at: datetime, trigger_at: datetime
    ) -> EpisodeExtras:
        window = self._window(motorcycle_track_id, start_at, trigger_at)
        threshold = self._params.rider_count_threshold
        supporting = [o for o in window if o.rider_count >= threshold]
        rider_ids, association_confidence = self._riders(motorcycle_track_id, start_at, trigger_at)
        max_rider_count = max((o.rider_count for o in window), default=0)

        measurements = (
            MeasuredValue(name="max_rider_count", value=float(max_rider_count), unit="count"),
            MeasuredValue(
                name="confirming_observations", value=float(len(supporting)), unit="count"
            ),
            MeasuredValue(
                name="observations_in_window", value=float(len(window)), unit="count"
            ),
        )
        # temporal_consistency degrades with flicker: an episode carried by every
        # frame scores 1.0; one bridged through drops below the threshold scores
        # lower. It is what a single per-frame count cannot express.
        consistency = len(supporting) / len(window) if window else None
        return EpisodeExtras(
            related_track_ids=rider_ids,
            confidence=ConfidenceBreakdown(
                temporal_consistency=consistency, association=association_confidence
            ),
            measurements=measurements,
        )

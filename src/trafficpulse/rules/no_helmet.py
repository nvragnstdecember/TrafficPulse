"""No-helmet temporal reasoning and confirmation (P4-U5).

Consumes ``HelmetStateObservation`` facts (from
``observations.helmet`` via the P4-U4 ``HelmetFrameObserver``) plus the
``Association`` links that say which motorcycle each rider was on, drives the
generic P1-U3 ``RuleEngine`` for hypothesis lifecycle mechanics through the shared
P3-U1 ``TemporalRunReasoner``, and mints frozen U2 ``ConfirmedEvent`` objects when
sustained bare-headed riding is observed.

**Reasoning begins here; perception ended at the observation stream.** This module
imports nothing from ``classifier`` or ``detector``, touches no pixels, and holds
no model. It reads frozen contracts only, which is what makes the decision
replayable from the observation log without a GPU (architecture-review §14/§15).

Separation of concerns
----------------------
This module owns only no-helmet *semantics*: which observations support a
candidate, how uncertainty is tolerated, when turban exempts, how long support
must persist, and how confidence aggregates. Lifecycle mechanics (ids, transition
validation, attachment, run tracking, taint reset, content-derived ``event_id``)
are delegated to the ``TemporalRunReasoner`` this reasoner *holds* -- the same
composition the wrong-way and illegal-stopping reasoners use.

The four-label ontology at the rule layer (U3's mapping, applied here)
----------------------------------------------------------------------
The contracts deliberately do not encode what each helmet label *means* for a
violation; that is this layer's job (phase-4 plan D4). The mapping:

* ``no_helmet`` -> **supports** the violation; starts/extends a run.
* ``helmet``    -> **contradicts**; ends the run. A rider seen wearing a helmet is
  not violating, and if support had not yet persisted long enough, no event is
  confirmed. This is "helmet recovery".
* ``turban``    -> **exempts** (see below).
* ``uncertain`` -> **abstains**: neither supports nor contradicts (see below).

Uncertainty is a gap, not a contradiction (the core temporal decision)
----------------------------------------------------------------------
The shared base takes a *boolean* per-step signal: active extends a run, inactive
ends it. Helmet state is three-valued, and neither boolean mapping is acceptable:
treating ``uncertain`` as active would **fabricate evidence** (asserting a bare
head that was never seen), while treating it as inactive would let a *single*
blurred frame destroy an otherwise solid episode -- exactly the classifier
instability, occlusion, and dropped-observation cases this rule must tolerate.

So ``uncertain`` observations are **not fed to the base at all**. The base already
specifies that "an ordinary missing/dropped observation is not a restart and keeps
its timestamp-driven bridging", so an uncertain stretch simply becomes a *gap the
run bridges* -- and ``max_observation_gap`` bounds how long that bridging may last.
Uncertainty therefore neither supports nor breaks a run; it just fails to add
evidence, which is precisely what abstention means. No change to the base's
semantics was needed to express this.

Two consequences worth stating:

* the ">= 2 observations to confirm" floor (architecture-review §13) applies to
  **supporting** observations only -- uncertain frames cannot make up the numbers;
* an episode bridged by uncertainty still had to *begin* and *trigger* on genuine
  ``no_helmet`` observations separated by at least ``min_persistence``.

Turban exemption is latching, and deliberately so
-------------------------------------------------
Any ``turban`` observation on a rider's track exempts that **rider for the whole
clip**: no event is ever confirmed for them. It is not merely "ends the current
run", because a later bare-headed stretch would then confirm -- and the phase-4
acceptance criterion is that a turban rider *never* confirms.

This is asymmetric on purpose. A single turban observation among many
``no_helmet`` ones causes a **false negative** (a missed violation); the opposite
policy would cause a **false positive** (penalising an exempt rider). For an
enforcement system those costs are not equal, and the ontology's own principle is
to prefer abstention over guessing. The exemption is recorded (see
:attr:`NoHelmetReasoner.exempt_track_ids`) rather than silent.

Note the ontology is explicit that the ``turban`` *label* "asserts no legal
exemption and no violation" -- it records what was seen. The exemption is this
**rule layer's** policy decision about that observation, which is exactly where U3
says such a decision belongs.

Attribution: the rider is the episode, the motorcycle is named on the event
---------------------------------------------------------------------------
Helmet state belongs to a *rider*, so episodes are keyed by the rider's track and
``HelmetStateObservation.track_id`` names the rider. But a violation is attributed
to the *vehicle*. The link is not duplicated into the observation: it is read from
the frozen ``Association`` contract (``subject_track_id`` = rider,
``object_track_id`` = motorcycle), and the confirmed event's ``track_ids`` carries
**both** -- resolved before ``event_id`` is computed, via the base's
``episode_enricher`` hook, because ``track_ids`` is identity-bearing.

Where a rider associated with more than one motorcycle inside the confirmed
window (rare; overlapping bikes), the **modal** motorcycle over that window is
chosen, ties breaking on the lowest track id. Deterministic, and stated rather
than hidden.

Taint
-----
Taint restarts are honoured verbatim through the base: a restart ends the open run
before the step is processed, so support never accumulates across an ID-switch
discontinuity (§13: tainted tracks may abstain but never confirm). A restarting
observation is passed through even when it is ``uncertain`` (as an inactive step),
so the discontinuity is never lost merely because the resuming frame happened to
be unreadable. A confirmed run is therefore taint-free by construction.

Parameters (provisional, from configuration)
--------------------------------------------
``no_helmet_parameters(scene)`` reads the ``no_helmet`` rule-parameter block:
``min_persistence`` (seconds) is required; ``max_observation_gap`` (seconds) and
``min_confirming_observations`` (count) are optional. Every value keeps its
configured ``ParameterStatus``; nothing is silently promoted to validated.

Persistence is expressed in **seconds, not frames**. The block's
``candidate_persistence_frames`` remains ``unset`` on purpose: P4-U1 measured the
same footage producing 234 tracks at ~10 fps versus 53 at ~30 fps, so a frame
count silently changes meaning with the clip, whereas a duration does not.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..contracts import (
    Association,
    ConfidenceBreakdown,
    ConfirmedEvent,
    HelmetStateObservation,
    MeasuredValue,
    ModelRef,
    ParameterStatus,
    SceneConfig,
)
from ..contracts.enums import AssociationType, HelmetState, ViolationType
from ..observations.helmet import HelmetDerivation
from .engine import RuleEngine
from .temporal import ConfirmationDetails, EpisodeExtras, TemporalRunReasoner

RULE_ID = "no_helmet"
RULE_VERSION = "0.1.0-provisional"

# The base structurally guarantees at least two supporting observations (a run
# needs a later observation than the one that opened it). A scene asking for more
# would need K-of-N counting, which is not implemented -- so it fails fast rather
# than being silently ignored.
STRUCTURAL_MIN_CONFIRMING_OBSERVATIONS = 2


# --- configuration -----------------------------------------------------------
@dataclass(frozen=True)
class NoHelmetParameters:
    """Provisional, scene-specific no-helmet parameters loaded from config."""

    min_persistence_seconds: float
    max_observation_gap_seconds: float | None
    min_confirming_observations: int | None
    persistence_status: ParameterStatus
    max_observation_gap_status: ParameterStatus
    min_confirming_observations_status: ParameterStatus


def no_helmet_parameters(scene: SceneConfig) -> NoHelmetParameters:
    """Load the no-helmet parameter block from a U5 ``SceneConfig``.

    Raises:
        ValueError: if the scene declares no ``no_helmet`` block, if
            ``min_persistence`` is absent or unset (reasoning cannot proceed
            without it), or if ``min_confirming_observations`` demands more
            supporting observations than the base structurally guarantees.
    """

    block = next(
        (b for b in scene.rule_parameters if b.violation_type is ViolationType.NO_HELMET), None
    )
    if block is None:
        raise ValueError("scene has no no_helmet rule-parameter block")
    by_id = {p.id: p for p in block.parameters}
    persistence = by_id.get("min_persistence")
    max_gap = by_id.get("max_observation_gap")
    min_confirming = by_id.get("min_confirming_observations")
    if persistence is None or persistence.value is None:
        raise ValueError("no_helmet min_persistence is unset")

    confirming_value = (
        int(min_confirming.value)
        if min_confirming is not None and min_confirming.value is not None
        else None
    )
    if confirming_value is not None and confirming_value > STRUCTURAL_MIN_CONFIRMING_OBSERVATIONS:
        raise ValueError(
            f"no_helmet min_confirming_observations={confirming_value} exceeds the "
            f"{STRUCTURAL_MIN_CONFIRMING_OBSERVATIONS} this reasoner structurally "
            "guarantees; K-of-N counting is not implemented, and silently ignoring "
            "the configured value would overstate the evidence behind a confirmation"
        )

    return NoHelmetParameters(
        min_persistence_seconds=persistence.value,
        max_observation_gap_seconds=(
            max_gap.value if max_gap is not None and max_gap.value is not None else None
        ),
        min_confirming_observations=confirming_value,
        persistence_status=persistence.status,
        max_observation_gap_status=(
            max_gap.status if max_gap is not None else ParameterStatus.UNSET
        ),
        min_confirming_observations_status=(
            min_confirming.status if min_confirming is not None else ParameterStatus.UNSET
        ),
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def exempt_riders(observations: Iterable[HelmetStateObservation]) -> frozenset[str]:
    """The riders a ``turban`` observation exempts (the latching rule; see module docs).

    The single source of truth for the exemption rule: :class:`NoHelmetReasoner`
    uses it to decide, and a caller (e.g. a run report) uses it to *describe* what
    was exempted -- so the description can never drift from the decision.
    """

    return frozenset(
        o.track_id
        for o in observations
        if o.track_id is not None and o.helmet_state is HelmetState.TURBAN
    )


class NoHelmetReasoner:
    """Deterministic no-helmet temporal reasoner over ``HelmetStateObservation``.

    No-helmet *semantics* live here (support/abstain/exempt mapping, latching
    turban exemption, confidence aggregation, motorcycle attribution); all
    lifecycle mechanics are delegated to the shared :class:`TemporalRunReasoner`
    this reasoner holds (P3-U1 composition), mirroring ``WrongWayReasoner``.

    Reasoning is per **rider** track. Feed a whole run's observations and their
    associations through :meth:`run` / :meth:`run_derivation`.
    """

    def __init__(
        self,
        engine: RuleEngine,
        params: NoHelmetParameters,
        *,
        scene_config_hash: str | None = None,
        rule_id: str = RULE_ID,
        rule_version: str | None = RULE_VERSION,
        models: tuple[ModelRef, ...] = (),
    ) -> None:
        self._params = params
        # Per-rider indices, rebuilt on each run() call; read by the enricher.
        self._by_rider: dict[str, list[HelmetStateObservation]] = {}
        self._bikes_by_rider: dict[str, list[Association]] = {}
        self._exempt: set[str] = set()
        self._machine = TemporalRunReasoner(
            engine,
            violation_type=ViolationType.NO_HELMET,
            threshold_seconds=params.min_persistence_seconds,
            detail_builder=self._details,
            scene_config_hash=scene_config_hash,
            rule_id=rule_id,
            rule_version=rule_version,
            models=models,
            # Bounds how long a run may bridge an uncertain/occluded stretch.
            max_observation_gap_seconds=params.max_observation_gap_seconds,
            episode_enricher=self._enrich,
        )

    @property
    def engine(self) -> RuleEngine:
        return self._machine.engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return self._machine.events

    @property
    def exempt_track_ids(self) -> frozenset[str]:
        """Riders exempted by a ``turban`` observation (recorded, never silent)."""

        return frozenset(self._exempt)

    def run(
        self,
        observations: Iterable[HelmetStateObservation],
        *,
        associations: Iterable[Association] = (),
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Reason over a run's helmet observations; return the confirmed events.

        Observations are processed in ``(timestamp, observation_id)`` order and
        de-duplicated by id by the base, so the outcome is independent of input
        order. ``associations`` supply the rider -> motorcycle attribution.
        """

        collected = list(observations)
        self._index(collected, associations)
        restarts = frozenset(taint_restart_ids)

        steps: list[tuple[HelmetStateObservation, bool]] = []
        for observation in collected:
            track_id = observation.track_id
            if track_id is None or track_id in self._exempt:
                continue  # untracked facts and exempt riders never reason
            state = observation.helmet_state
            if state is HelmetState.NO_HELMET:
                steps.append((observation, True))  # supports
            elif state is HelmetState.HELMET:
                steps.append((observation, False))  # contradicts: recovery ends the run
            elif observation.observation_id in restarts:
                # UNCERTAIN (or TURBAN on a non-exempt path) that carries a taint
                # restart: pass it through inactive so the discontinuity is never
                # lost just because the resuming frame was unreadable.
                steps.append((observation, False))
            # else: UNCERTAIN -> not fed at all; the run bridges the gap (see docs)

        return self._machine.run(steps, taint_restart_ids=restarts)

    def run_derivation(
        self, derivation: HelmetDerivation, *, associations: Iterable[Association] = ()
    ) -> tuple[ConfirmedEvent, ...]:
        """Convenience: run a ``HelmetDerivation`` with its taint restarts."""

        return self.run(
            derivation.observations,
            associations=associations,
            taint_restart_ids=derivation.taint_restart_ids,
        )

    # --- indexing ------------------------------------------------------------
    def _index(
        self, observations: Sequence[HelmetStateObservation], associations: Iterable[Association]
    ) -> None:
        self._by_rider = {}
        self._bikes_by_rider = {}
        self._exempt = set(exempt_riders(observations))  # latching (see module docs)
        for observation in observations:
            track_id = observation.track_id
            if track_id is None:
                continue
            self._by_rider.setdefault(track_id, []).append(observation)
        for association in associations:
            if association.association_type is not AssociationType.RIDER_OF_MOTORCYCLE:
                continue
            self._bikes_by_rider.setdefault(association.subject_track_id, []).append(association)

    def _window(
        self, track_id: str, start_at: datetime, trigger_at: datetime
    ) -> list[HelmetStateObservation]:
        return [
            o
            for o in self._by_rider.get(track_id, ())
            if start_at <= o.timestamp <= trigger_at
        ]

    def _motorcycle(
        self, track_id: str, start_at: datetime, trigger_at: datetime
    ) -> tuple[str | None, float | None]:
        """The modal motorcycle over the window, and the weakest link to it.

        Ties break on the lowest motorcycle track id. Returns ``(None, None)`` when
        the rider had no association in the window -- never invents a vehicle.
        """

        in_window = [
            a
            for a in self._bikes_by_rider.get(track_id, ())
            if start_at <= a.timestamp <= trigger_at
        ]
        if not in_window:
            return None, None
        counts: dict[str, int] = {}
        for association in in_window:
            counts[association.object_track_id] = counts.get(association.object_track_id, 0) + 1
        # Modal, lowest id on a tie -- deterministic and independent of input order.
        best = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        # The weakest association observed in the window: a conservative summary of
        # how well-established the rider<->motorcycle link was, never the strongest.
        weakest = min(a.confidence for a in in_window if a.object_track_id == best)
        return best, weakest

    # --- injected policies ---------------------------------------------------
    def _details(self, start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
        thresholds = [
            MeasuredValue(
                name="min_persistence",
                value=self._params.min_persistence_seconds,
                unit="seconds",
            )
        ]
        if self._params.max_observation_gap_seconds is not None:
            thresholds.append(
                MeasuredValue(
                    name="max_observation_gap",
                    value=self._params.max_observation_gap_seconds,
                    unit="seconds",
                )
            )
        if self._params.min_confirming_observations is not None:
            thresholds.append(
                MeasuredValue(
                    name="min_confirming_observations",
                    value=float(self._params.min_confirming_observations),
                    unit="count",
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

    def _enrich(self, track_id: str, start_at: datetime, trigger_at: datetime) -> EpisodeExtras:
        """Attribute the motorcycle and aggregate the episode's confidence.

        See the module docstring for attribution, and
        :meth:`_confidence` for how the components are derived.
        """

        window = self._window(track_id, start_at, trigger_at)
        motorcycle, association_confidence = self._motorcycle(track_id, start_at, trigger_at)
        supporting = [o for o in window if o.helmet_state is HelmetState.NO_HELMET]

        measurements = [
            MeasuredValue(
                name="confirming_observations", value=float(len(supporting)), unit="count"
            ),
            MeasuredValue(
                name="observations_in_window", value=float(len(window)), unit="count"
            ),
        ]
        crop_heights = [o.crop_height_px for o in supporting if o.crop_height_px is not None]
        if crop_heights:
            measurements.append(
                MeasuredValue(
                    name="min_crop_height_px", value=min(crop_heights), unit=None
                )
            )
        return EpisodeExtras(
            related_track_ids=(motorcycle,) if motorcycle is not None else (),
            confidence=self._confidence(window, supporting, association_confidence),
            measurements=tuple(measurements),
        )

    def _confidence(
        self,
        window: Sequence[HelmetStateObservation],
        supporting: Sequence[HelmetStateObservation],
        association_confidence: float | None,
    ) -> ConfidenceBreakdown:
        """Aggregate an episode's evidence into typed components (§13).

        Reasoning confidence is deliberately **not** the classifier's confidence:

        * ``classifier`` -- the **mean** score across the *supporting* observations,
          a summary of how strongly the model asserted a bare head across the whole
          episode rather than at the one triggering frame. Observations whose score
          was never measured (a gated crop, ``confidence=None``) contribute nothing
          rather than a fabricated zero; supporting observations are, by
          construction, ones that cleared the P4-U4 gates.
        * ``temporal_consistency`` -- supporting observations over all observations
          in the window. This is what a single classifier score cannot express: an
          episode carried by 10 consecutive ``no_helmet`` frames scores 1.0, while
          one bridged through heavy uncertainty scores low, even if every
          individual classifier score was high. It is the component that *degrades
          with instability*.
        * ``association`` -- the weakest rider<->motorcycle overlap in the window: how
          well-established it is that this rider was on this vehicle at all.
        * ``detector`` -- ``None``. Detection confidence does not travel on the
          observation stream, and this layer must not invent it.
        * ``track_continuity`` -- ``None``. A confirmed run is taint-free *by
          construction* (a restart ends the run), so reporting 1.0 would restate a
          structural invariant as if it were measured evidence.
        * ``geometric_margin`` / ``calibration_quality`` -- ``None``; neither exists
          for this rule (no metric calibration; Phase 5).
        * ``aggregate`` -- ``None``, deliberately. Collapsing un-calibrated
          components into one number invites reading it as a probability of guilt,
          which §13 forbids until calibration is demonstrated. The components are
          published; the reader may weigh them.
        """

        scores = [o.confidence for o in supporting if o.confidence is not None]
        consistency = len(supporting) / len(window) if window else None
        return ConfidenceBreakdown(
            classifier=_mean(scores),
            temporal_consistency=consistency,
            association=association_confidence,
        )

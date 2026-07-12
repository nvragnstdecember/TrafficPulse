"""Illegal-stopping temporal reasoning and confirmation (P2-U4).

Consumes ``InZoneObservation`` and ``StationaryObservation`` facts (from
``observations.zones`` and ``observations.stationary``), joins them into a
per-step *stopped-in-zone* signal, drives the generic P1-U3 ``RuleEngine`` for
hypothesis lifecycle mechanics, and mints frozen U2 ``ConfirmedEvent`` objects
when a track dwells stationary inside an eligible no-stopping zone for at least
``stationary_duration``.

Structurally this reasoner is the wrong-way reasoner's twin: it owns only
illegal-stopping *semantics* (what counts as stopped-in-zone, how long support
must persist, recovery/reset, confirmation) and reuses ``RuleEngine`` for all
lifecycle mechanics (ids, transition validation, attachment, duplicate
suppression, close/abandon). The only new concern relative to wrong-way is the
deterministic *join* of two observation streams into the single per-step boolean
the run logic consumes.

Two-stream join (deterministic; model-free)
-------------------------------------------
:func:`join_stopped_in_zone` pairs the two derivations on
``(camera_id, track_id, timestamp)``. The ``StationaryObservation`` is the
**carrier** (exactly one per usable step, so it drives engine ingestion and event
timing); the in-zone side may contribute several observations per step (one per
eligible zone), which are folded to ``is_inside_any`` over the
``ZoneKind.NO_STOPPING`` zones only. A step is *stopped-in-zone* iff::

    carrier.is_stationary AND any(o.is_inside for no-stopping o at the same key)

Conservative pairing (plan §9 D.1; Phase-3 "never fabricate evidence"): a step
with no matching in-zone fact folds to ``is_inside_any = False`` (not stopped) --
stationarity alone, with no proof of zone membership, cannot confirm an illegal
stop. Symmetrically, an in-zone fact at a timestamp with no stationary carrier
produces no step at all (no stationarity evidence -> nothing to reason over); it
is bridged like an ordinary gap, never silently treated as a stop. The two
derivations emit at the *same* timestamps for a clean contiguous run, so a
one-sided step only arises from pathological/hand-built input, where the
conservative fold is the safe choice.

Because both derivations reuse the ``HeadingDerivation`` taint mechanism, a taint
restart on either stream is unioned onto the carrier id, so a resumed-after-taint
step resets the run regardless of which stream flagged it.

Parameters (provisional, from configuration)
--------------------------------------------
``illegal_stopping_parameters(scene)`` reads the ``illegal_stopping`` rule-
parameter block from a U5 ``SceneConfig``: ``stationary_duration`` (seconds) is
required; ``motion_threshold`` (m/s) is loaded but **not applied** in this
uncalibrated synthetic slice -- converting m/s to the pixel space of synthetic
tracks needs a validated calibration that does not exist, so stationarity is the
geometric pixel-space test in the derivation layer and ``motion_threshold`` is
recorded on the event's thresholds purely for provenance. ``max_observation_gap``
(seconds) is an optional provisional tolerance: when set, an inter-observation
gap wider than it ends the current run (a fresh run may start at the next stopped
step); when unset, the run relies on timestamp bridging. Every value keeps its
configured ``ParameterStatus``; nothing is silently promoted to validated.

Temporal semantics (timestamp-driven; never wall-clock)
------------------------------------------------------
Per ``(camera_id, track_id)`` a contiguous *stopped-in-zone run* is tracked:

* a stopped-in-zone step with no open run starts a run and creates + promotes a
  hypothesis (engine ``NEW -> CANDIDATE``);
* a stopped-in-zone step on an open run attaches to the hypothesis; once
  ``observation.timestamp - run_start >= stationary_duration`` the hypothesis is
  activated (``CANDIDATE -> ACTIVE``) and exactly one ``ConfirmedEvent`` is
  emitted for the run;
* a non-stopped step (moving, or outside every eligible no-stopping zone) ends
  the run -- ``close`` if it had already confirmed (``ACTIVE -> CLOSED``),
  otherwise ``abandon`` (``-> ABANDONED``) -- so recovery before the dwell
  threshold prevents confirmation.

An explicit *taint restart* also ends the current run before the step is
processed, so dwell never accumulates across a tainted/ID-switch interval
(architecture-review §13: tainted tracks may abstain but never confirm), whereas
an ordinary missing/dropped observation is not a restart and keeps its
timestamp-driven bridging. Because a run needs a later observation than the one
that opened it, confirmation structurally requires at least two observations.

Event identity (deterministic; provisional per ADR-004)
-------------------------------------------------------
``event_id`` is a SHA-256 over canonical JSON of the identity-bearing fields
(scene hash, camera, violation, rule, track ids, start/trigger timestamps, source
hypothesis id) -- the same deterministic, process-independent scheme wrong-way
uses. Run-level ``models`` provenance is stamped onto the event but is
deliberately absent from the id and from every predicate, so the *decision*
(which events, ids, timing) is byte-identical with or without provenance.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ..contracts import (
    ConfirmedEvent,
    MeasuredValue,
    ModelRef,
    ParameterStatus,
    SceneConfig,
    StationaryObservation,
)
from ..contracts.enums import ViolationType, ZoneKind
from ..observations.stationary import StationaryDerivation
from ..observations.zones import InZoneDerivation
from .engine import RuleEngine
from .temporal import ConfirmationDetails, TemporalRunReasoner

RULE_ID = "illegal_stopping"
RULE_VERSION = "0.1.0-provisional"


# --- configuration -----------------------------------------------------------
@dataclass(frozen=True)
class IllegalStoppingParameters:
    """Provisional, scene-specific illegal-stopping parameters loaded from config.

    ``motion_threshold`` is carried for provenance but **not applied** in this
    uncalibrated synthetic slice (see module docstring). ``max_observation_gap``
    is an optional provisional run-break tolerance (``None`` -> pure timestamp
    bridging). Every ``*_status`` preserves the configured provisional/unset
    marker.
    """

    stationary_duration_seconds: float
    motion_threshold: float | None
    max_observation_gap_seconds: float | None
    duration_status: ParameterStatus
    motion_threshold_status: ParameterStatus
    max_observation_gap_status: ParameterStatus


def illegal_stopping_parameters(scene: SceneConfig) -> IllegalStoppingParameters:
    """Load the illegal-stopping parameter block from a U5 ``SceneConfig``.

    Raises:
        ValueError: if the scene declares no ``illegal_stopping`` block, or if
            ``stationary_duration`` is absent or unset (reasoning cannot proceed
            without the dwell threshold).
    """

    block = next(
        (b for b in scene.rule_parameters if b.violation_type is ViolationType.ILLEGAL_STOPPING),
        None,
    )
    if block is None:
        raise ValueError("scene has no illegal_stopping rule-parameter block")
    by_id = {p.id: p for p in block.parameters}
    duration = by_id.get("stationary_duration")
    motion = by_id.get("motion_threshold")
    max_gap = by_id.get("max_observation_gap")
    if duration is None or duration.value is None:
        raise ValueError("illegal_stopping stationary_duration is unset")
    return IllegalStoppingParameters(
        stationary_duration_seconds=duration.value,
        motion_threshold=motion.value if motion is not None else None,
        max_observation_gap_seconds=(
            max_gap.value if max_gap is not None and max_gap.value is not None else None
        ),
        duration_status=duration.status,
        motion_threshold_status=motion.status if motion is not None else ParameterStatus.UNSET,
        max_observation_gap_status=(
            max_gap.status if max_gap is not None else ParameterStatus.UNSET
        ),
    )


# --- two-stream join ---------------------------------------------------------
@dataclass(frozen=True)
class StoppedInZoneStep:
    """One joined per-step fact: the carrier observation + the stopped-in-zone flag.

    ``observation`` (the ``StationaryObservation``) is the engine-ingestion carrier
    and the source of the step's ``camera_id`` / ``track_id`` / ``timestamp`` /
    ``observation_id``; ``stopped_in_zone`` is the deterministic join result.
    """

    observation: StationaryObservation
    stopped_in_zone: bool


_JoinKey = tuple[str, str | None, datetime]


def join_stopped_in_zone(
    in_zone: InZoneDerivation, stationary: StationaryDerivation
) -> tuple[list[StoppedInZoneStep], frozenset[str]]:
    """Join the two derivations into per-step stopped-in-zone facts + taint restarts.

    Pairs on ``(camera_id, track_id, timestamp)``; the ``StationaryObservation`` is
    the carrier (one per step). In-zone facts are folded to ``is_inside_any`` over
    ``ZoneKind.NO_STOPPING`` zones only, so a broader in-zone derivation is
    tolerated (non-no-stopping zones contribute nothing). ``stopped_in_zone`` is
    ``carrier.is_stationary and is_inside_any`` -- a missing in-zone fact folds to
    ``False`` (never fabricated as a stop). Returns the steps in the stationary
    stream's input order (the reasoner re-sorts deterministically) plus the carrier
    ids that resume after a taint on *either* stream.
    """

    inside_by_key: dict[_JoinKey, bool] = {}
    inzone_restart_keys: set[_JoinKey] = set()
    for obs in in_zone.observations:
        if obs.zone_kind is not ZoneKind.NO_STOPPING:
            continue  # only no-stopping membership can support an illegal stop
        key: _JoinKey = (obs.camera_id, obs.track_id, obs.timestamp)
        inside_by_key[key] = inside_by_key.get(key, False) or obs.is_inside
        if obs.observation_id in in_zone.taint_restart_ids:
            inzone_restart_keys.add(key)

    steps: list[StoppedInZoneStep] = []
    restart_ids: set[str] = set()
    for carrier in stationary.observations:
        key = (carrier.camera_id, carrier.track_id, carrier.timestamp)
        stopped = carrier.is_stationary and inside_by_key.get(key, False)
        steps.append(StoppedInZoneStep(observation=carrier, stopped_in_zone=stopped))
        if carrier.observation_id in stationary.taint_restart_ids or key in inzone_restart_keys:
            restart_ids.add(carrier.observation_id)
    return steps, frozenset(restart_ids)


class IllegalStoppingReasoner:
    """Deterministic illegal-stopping temporal reasoner over joined stopped-in-zone steps.

    Illegal-stopping *semantics* live here: the per-step signal is the joined
    ``stopped_in_zone`` flag, an optional ``max_observation_gap`` breaks a stale
    dwell run, and confirmation records a ``dwell_seconds`` measurement against the
    ``stationary_duration`` (and provenance-only ``motion_threshold``) thresholds.
    All lifecycle mechanics -- run tracking, taint reset, the gap-break, engine
    transitions, ``models`` stamping, content-derived ``event_id`` -- are delegated
    to the shared :class:`TemporalRunReasoner` this reasoner *holds* (P3-U1
    composition); the public API is unchanged.
    """

    def __init__(
        self,
        engine: RuleEngine,
        params: IllegalStoppingParameters,
        *,
        scene_config_hash: str | None = None,
        rule_id: str = RULE_ID,
        rule_version: str | None = RULE_VERSION,
        models: tuple[ModelRef, ...] = (),
    ) -> None:
        self._params = params
        self._machine = TemporalRunReasoner(
            engine,
            violation_type=ViolationType.ILLEGAL_STOPPING,
            threshold_seconds=params.stationary_duration_seconds,
            detail_builder=self._details,
            scene_config_hash=scene_config_hash,
            rule_id=rule_id,
            rule_version=rule_version,
            models=models,
            max_observation_gap_seconds=params.max_observation_gap_seconds,
        )

    @property
    def engine(self) -> RuleEngine:
        return self._machine.engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return self._machine.events

    def observe(
        self, step: StoppedInZoneStep, *, is_taint_restart: bool = False
    ) -> ConfirmedEvent | None:
        """Process one joined step in timestamp order; return any emitted event.

        ``is_taint_restart`` marks the first clean step resuming after a tainted
        interval. It terminates any open run for the track *before* processing, so
        dwell cannot accumulate across the tainted (ID-switch) discontinuity
        (architecture-review §13). An ordinary missing/dropped observation is never
        a restart and keeps its timestamp-driven bridging.
        """

        return self._machine.observe(
            step.observation, active=step.stopped_in_zone, is_taint_restart=is_taint_restart
        )

    def run(
        self,
        steps: Iterable[StoppedInZoneStep],
        *,
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Process steps in ``(timestamp, observation_id)`` order, de-duplicated by id.

        ``taint_restart_ids`` are carrier observation ids that resume after a
        tainted interval; each resets the track's run before it is processed.
        Returns the events emitted during this call. Ordering follows the P1-U3
        policy, so the outcome is independent of input order.
        """

        return self._machine.run(
            ((s.observation, s.stopped_in_zone) for s in steps),
            taint_restart_ids=taint_restart_ids,
        )

    def run_join(
        self, in_zone: InZoneDerivation, stationary: StationaryDerivation
    ) -> tuple[ConfirmedEvent, ...]:
        """Convenience: join two derivations and run, honouring their taint restarts."""

        steps, restart_ids = join_stopped_in_zone(in_zone, stationary)
        return self.run(steps, taint_restart_ids=restart_ids)

    def _details(self, start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
        thresholds = [
            MeasuredValue(
                name="stationary_duration",
                value=self._params.stationary_duration_seconds,
                unit="seconds",
            ),
        ]
        if self._params.motion_threshold is not None:
            # Recorded for provenance, NOT applied in this uncalibrated slice.
            thresholds.append(
                MeasuredValue(
                    name="motion_threshold", value=self._params.motion_threshold, unit="m_per_s"
                )
            )
        return ConfirmationDetails(
            measurements=(
                MeasuredValue(
                    name="dwell_seconds",
                    value=(trigger_at - start_at).total_seconds(),
                    unit="seconds",
                ),
            ),
            thresholds=tuple(thresholds),
        )

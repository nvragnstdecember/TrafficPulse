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

import hashlib
import json
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
from .engine import HypothesisRecord, RuleEngine
from .states import EngineState

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


# --- per-track run state (engine-internal bookkeeping, not a contract) --------
@dataclass
class _Run:
    hypothesis_id: str
    start_at: datetime
    last_at: datetime
    confirmed: bool = False
    closed: bool = False


class IllegalStoppingReasoner:
    """Deterministic illegal-stopping temporal reasoner over joined stopped-in-zone steps."""

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
        self._engine = engine
        self._params = params
        self._scene_hash = scene_config_hash
        self._rule_id = rule_id
        self._rule_version = rule_version
        # Run-level provenance stamped onto every minted event (P2-U1 shape). Pure
        # metadata: no predicate, threshold, dwell timer, or transition ever reads
        # it, and it is deliberately absent from ``_event_id``, so the *decision* is
        # byte-identical with or without it. The composition boundary supplies the
        # sorted/de-duplicated tuple.
        self._models = models
        self._runs: dict[tuple[str, str], _Run] = {}
        self._events: list[ConfirmedEvent] = []

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return tuple(self._events)

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

        observation = step.observation
        track_id = observation.track_id
        if track_id is None:
            return None  # illegal-stopping episodes are per-track; ignore untracked facts
        key = (observation.camera_id, track_id)
        run = self._runs.get(key)
        if is_taint_restart:
            self._on_recovery(run)  # break episode continuity at the taint discontinuity
            run = self._runs.get(key)
        if step.stopped_in_zone:
            return self._on_stopped(key, run, observation)
        self._on_recovery(run)
        return None

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

        restarts = frozenset(taint_restart_ids)
        ordered = sorted(
            steps, key=lambda s: (s.observation.timestamp, s.observation.observation_id)
        )
        seen: set[str] = set()
        emitted: list[ConfirmedEvent] = []
        for step in ordered:
            observation_id = step.observation.observation_id
            if observation_id in seen:
                continue
            seen.add(observation_id)
            event = self.observe(step, is_taint_restart=observation_id in restarts)
            if event is not None:
                emitted.append(event)
        return tuple(emitted)

    def run_join(
        self, in_zone: InZoneDerivation, stationary: StationaryDerivation
    ) -> tuple[ConfirmedEvent, ...]:
        """Convenience: join two derivations and run, honouring their taint restarts."""

        steps, restart_ids = join_stopped_in_zone(in_zone, stationary)
        return self.run(steps, taint_restart_ids=restart_ids)

    def _on_stopped(
        self,
        key: tuple[str, str],
        run: _Run | None,
        observation: StationaryObservation,
    ) -> ConfirmedEvent | None:
        # An over-wide inter-observation gap (provisional tolerance) ends a stale
        # run; a fresh run may then open at this observation below.
        max_gap = self._params.max_observation_gap_seconds
        if (
            run is not None
            and not run.closed
            and max_gap is not None
            and (observation.timestamp - run.last_at).total_seconds() > max_gap
        ):
            self._on_recovery(run)
            run = None

        if run is None or run.closed:
            record = self._engine.ingest(
                observation,
                rule_id=self._rule_id,
                violation_type=ViolationType.ILLEGAL_STOPPING,
                rule_version=self._rule_version,
            )
            self._engine.promote(record.hypothesis_id)
            self._runs[key] = _Run(
                hypothesis_id=record.hypothesis_id,
                start_at=observation.timestamp,
                last_at=observation.timestamp,
            )
            return None

        record = self._engine.ingest(
            observation, rule_id=self._rule_id, violation_type=ViolationType.ILLEGAL_STOPPING
        )
        run.last_at = observation.timestamp
        if run.confirmed:
            return None
        elapsed = (observation.timestamp - run.start_at).total_seconds()
        if elapsed < self._params.stationary_duration_seconds:
            return None
        if record.state is EngineState.CANDIDATE:
            record = self._engine.activate(record.hypothesis_id)
        event = self._confirm(record, observation)
        run.confirmed = True
        self._events.append(event)
        return event

    def _on_recovery(self, run: _Run | None) -> None:
        if run is None or run.closed:
            return
        if run.confirmed:
            self._engine.close(run.hypothesis_id)
        else:
            self._engine.abandon(run.hypothesis_id)
        run.closed = True

    def _confirm(
        self, record: HypothesisRecord, trigger: StationaryObservation
    ) -> ConfirmedEvent:
        start_at = record.first_at
        assert start_at is not None  # an attached hypothesis always has a first observation
        trigger_at = trigger.timestamp
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
        return ConfirmedEvent(
            event_id=self._event_id(
                record.camera_id, record.track_ids, start_at, trigger_at, record.hypothesis_id
            ),
            violation_type=ViolationType.ILLEGAL_STOPPING,
            camera_id=record.camera_id,
            track_ids=record.track_ids,
            start_at=start_at,
            trigger_at=trigger_at,
            rule_id=self._rule_id,
            rule_version=self._rule_version,
            scene_config_hash=self._scene_hash,
            models=self._models,  # run-level provenance; never enters _event_id
            source_hypothesis_id=record.hypothesis_id,
            created_at=trigger_at,  # deterministic data timestamp, never wall-clock
            measurements=(
                MeasuredValue(
                    name="dwell_seconds",
                    value=(trigger_at - start_at).total_seconds(),
                    unit="seconds",
                ),
            ),
            thresholds=tuple(thresholds),
        )

    def _event_id(
        self,
        camera_id: str,
        track_ids: tuple[str, ...],
        start_at: datetime,
        trigger_at: datetime,
        hypothesis_id: str,
    ) -> str:
        material = json.dumps(
            {
                "scene_config_hash": self._scene_hash or "",
                "camera_id": camera_id,
                "violation_type": ViolationType.ILLEGAL_STOPPING.value,
                "rule_id": self._rule_id,
                "track_ids": list(track_ids),
                "start_at": start_at.isoformat(),
                "trigger_at": trigger_at.isoformat(),
                "source_hypothesis_id": hypothesis_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "evt-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

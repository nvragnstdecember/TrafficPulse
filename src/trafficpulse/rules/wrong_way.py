"""Wrong-way temporal reasoning and confirmation (P1-U4, concerns 2 & 3).

Consumes ``HeadingVsLaneObservation`` facts (from
``observations.heading.derive_heading_observations``), drives the generic P1-U3
``RuleEngine`` for hypothesis lifecycle mechanics, and -- as the first unit
permitted to do so -- mints frozen U2 ``ConfirmedEvent`` objects when sustained
wrong-way behavior is observed.

Separation of concerns
----------------------
This module owns only wrong-way *semantics*: whether an observation supports a
candidate, whether support has persisted long enough, recovery/reset, and
confirmation. It does **not** reimplement lifecycle mechanics (ids, transition
validation, attachment, duplicate suppression, lookup, close/abandon) -- those
come from ``RuleEngine``, which remains violation-agnostic.

Parameters (provisional, from configuration)
--------------------------------------------
``wrong_way_parameters(scene)`` reads the ``wrong_way`` rule-parameter block from
a U5 ``SceneConfig``: ``heading_deviation_max`` (degrees) and ``min_persistence``
(seconds) are required; ``min_speed`` (m/s) is loaded but **not applied** in this
uncalibrated synthetic slice -- converting m/s to the pixel space of synthetic
tracks needs a validated calibration that does not exist yet, so the usable-
movement gate is the geometric zero-displacement skip in the derivation layer.
Every value keeps its configured ``ParameterStatus`` (all ``provisional`` in the
example scene); nothing is silently promoted to validated.

Temporal semantics (timestamp-driven; never wall-clock)
------------------------------------------------------
Per ``(camera_id, track_id)`` a contiguous contradiction *run* is tracked:

* a contradiction observation with no open run starts a run and creates +
  promotes a hypothesis (engine ``NEW -> CANDIDATE``);
* a contradiction observation on an open run attaches to the hypothesis; once
  ``observation.timestamp - run_start >= min_persistence`` the hypothesis is
  activated (``CANDIDATE -> ACTIVE``) and exactly one ``ConfirmedEvent`` is
  emitted for the run;
* a legal (non-contradiction) observation ends the run -- ``close`` if it had
  already confirmed (``ACTIVE -> CLOSED``), otherwise ``abandon`` (``->
  ABANDONED``) -- so recovery before persistence prevents confirmation.

An explicit *taint restart* (an observation flagged by ``HeadingDerivation`` as
resuming after a tainted/ID-switch interval) also ends the current run before it
is processed. Support therefore never accumulates across a tainted interval
(architecture-review §13: tainted tracks may abstain but never confirm), whereas
an ordinary missing/dropped observation is not a restart and keeps its
timestamp-driven bridging. A genuinely sustained *clean* segment after the taint
starts a fresh run and may confirm on its own.

Because a run needs a later observation than the one that opened it, confirmation
structurally requires at least two observations (architecture-review §13).

Confirmation across the P1-U3/U2 boundary
-----------------------------------------
P1-U3's ``EngineState`` intentionally has no ``CONFIRMED``. Confirmation here is
represented by the *separate* ``ConfirmedEvent`` (linked via
``source_hypothesis_id``), not by mutating the hypothesis; the engine hypothesis
stays ``ACTIVE`` (which maps to the frozen ``LifecycleState.CANDIDATE``). This
preserves the documented P1-U3 mapping and leaves the generic engine untouched.

Event identity (deterministic; provisional per ADR-004)
-------------------------------------------------------
``event_id`` is a SHA-256 over canonical JSON of the identity-bearing fields
(scene hash, camera, violation, rule, track ids, start/trigger timestamps,
source hypothesis id). It is deterministic and process-independent. ADR-004 is
still *Proposed* and does not fix cross-run event identity; this content-derived
strategy is the smallest deterministic choice for the synthetic/replay context
and is revisitable when the event-store runtime lands. ADR-004's status is not
changed by this unit.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ..contracts import (
    ConfirmedEvent,
    HeadingVsLaneObservation,
    MeasuredValue,
    ModelRef,
    ParameterStatus,
    SceneConfig,
)
from ..contracts.enums import ViolationType
from ..observations.heading import HeadingDerivation
from .engine import HypothesisRecord, RuleEngine
from .states import EngineState

RULE_ID = "wrong_way"
RULE_VERSION = "0.1.0-provisional"


# --- configuration -----------------------------------------------------------
@dataclass(frozen=True)
class WrongWayParameters:
    """Provisional, scene-specific wrong-way parameters loaded from config.

    ``min_speed`` is carried for provenance but not applied in this uncalibrated
    synthetic slice (see module docstring). Every ``*_status`` preserves the
    configured provisional/unset marker.
    """

    deviation_max_degrees: float
    min_persistence_seconds: float
    min_speed: float | None
    deviation_status: ParameterStatus
    persistence_status: ParameterStatus
    min_speed_status: ParameterStatus


def wrong_way_parameters(scene: SceneConfig) -> WrongWayParameters:
    """Load the wrong-way parameter block from a U5 ``SceneConfig``.

    Raises:
        ValueError: if the scene declares no ``wrong_way`` block, or if
            ``heading_deviation_max`` / ``min_persistence`` are absent or unset
            (reasoning cannot proceed without them).
    """

    block = next(
        (b for b in scene.rule_parameters if b.violation_type is ViolationType.WRONG_WAY), None
    )
    if block is None:
        raise ValueError("scene has no wrong_way rule-parameter block")
    by_id = {p.id: p for p in block.parameters}
    deviation = by_id.get("heading_deviation_max")
    persistence = by_id.get("min_persistence")
    speed = by_id.get("min_speed")
    if deviation is None or deviation.value is None:
        raise ValueError("wrong_way heading_deviation_max is unset")
    if persistence is None or persistence.value is None:
        raise ValueError("wrong_way min_persistence is unset")
    return WrongWayParameters(
        deviation_max_degrees=deviation.value,
        min_persistence_seconds=persistence.value,
        min_speed=speed.value if speed is not None else None,
        deviation_status=deviation.status,
        persistence_status=persistence.status,
        min_speed_status=speed.status if speed is not None else ParameterStatus.UNSET,
    )


# --- per-track run state (engine-internal bookkeeping, not a contract) --------
@dataclass
class _Run:
    hypothesis_id: str
    start_at: datetime
    confirmed: bool = False
    closed: bool = False


class WrongWayReasoner:
    """Deterministic wrong-way temporal reasoner over ``HeadingVsLaneObservation``."""

    def __init__(
        self,
        engine: RuleEngine,
        params: WrongWayParameters,
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
        # Run-level provenance stamped onto every minted event (P2-U1). Pure
        # metadata: no rule predicate, threshold, or timer ever reads it, and it
        # is deliberately absent from ``_event_id``, so the *decision* (which
        # events, ids, timing) is byte-identical with or without it. The caller
        # (the composition boundary) supplies the sorted/de-duplicated tuple.
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
        self, observation: HeadingVsLaneObservation, *, is_taint_restart: bool = False
    ) -> ConfirmedEvent | None:
        """Process one observation in timestamp order; return any emitted event.

        ``is_taint_restart`` marks the first clean observation resuming after a
        tainted interval. It terminates any open run for the track *before*
        processing, so wrong-way support cannot accumulate across the tainted
        (ID-switch) discontinuity (architecture-review §13: tainted tracks may
        abstain but never confirm). An ordinary missing/dropped observation is
        never a restart and keeps its timestamp-driven bridging.
        """

        track_id = observation.track_id
        if track_id is None:
            return None  # wrong-way episodes are per-track; ignore untracked facts
        key = (observation.camera_id, track_id)
        run = self._runs.get(key)
        if is_taint_restart:
            self._on_recovery(run)  # break episode continuity at the taint discontinuity
            run = self._runs.get(key)
        if observation.is_contradiction:
            return self._on_contradiction(key, run, observation)
        self._on_recovery(run)
        return None

    def run(
        self,
        observations: Iterable[HeadingVsLaneObservation],
        *,
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Process observations in ``(timestamp, id)`` order, de-duplicated by id.

        ``taint_restart_ids`` are observation ids that resume after a tainted
        interval (from ``HeadingDerivation``); each resets the track's run before
        it is processed. Returns the events emitted during this call. Ordering
        follows the P1-U3 policy, so the outcome is independent of input order.
        """

        restarts = frozenset(taint_restart_ids)
        ordered = sorted(observations, key=lambda o: (o.timestamp, o.observation_id))
        seen: set[str] = set()
        emitted: list[ConfirmedEvent] = []
        for observation in ordered:
            if observation.observation_id in seen:
                continue
            seen.add(observation.observation_id)
            event = self.observe(
                observation, is_taint_restart=observation.observation_id in restarts
            )
            if event is not None:
                emitted.append(event)
        return tuple(emitted)

    def run_derivation(self, derivation: HeadingDerivation) -> tuple[ConfirmedEvent, ...]:
        """Convenience: run a ``HeadingDerivation`` with its taint restarts."""

        return self.run(derivation.observations, taint_restart_ids=derivation.taint_restart_ids)

    def _on_contradiction(
        self,
        key: tuple[str, str],
        run: _Run | None,
        observation: HeadingVsLaneObservation,
    ) -> ConfirmedEvent | None:
        if run is None or run.closed:
            record = self._engine.ingest(
                observation,
                rule_id=self._rule_id,
                violation_type=ViolationType.WRONG_WAY,
                rule_version=self._rule_version,
            )
            self._engine.promote(record.hypothesis_id)
            self._runs[key] = _Run(
                hypothesis_id=record.hypothesis_id, start_at=observation.timestamp
            )
            return None

        record = self._engine.ingest(
            observation, rule_id=self._rule_id, violation_type=ViolationType.WRONG_WAY
        )
        if run.confirmed:
            return None
        elapsed = (observation.timestamp - run.start_at).total_seconds()
        if elapsed < self._params.min_persistence_seconds:
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
        self, record: HypothesisRecord, trigger: HeadingVsLaneObservation
    ) -> ConfirmedEvent:
        start_at = record.first_at
        assert start_at is not None  # an attached hypothesis always has a first observation
        trigger_at = trigger.timestamp
        return ConfirmedEvent(
            event_id=self._event_id(record.camera_id, record.track_ids, start_at, trigger_at,
                                    record.hypothesis_id),
            violation_type=ViolationType.WRONG_WAY,
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
                    name="persistence_seconds",
                    value=(trigger_at - start_at).total_seconds(),
                    unit="seconds",
                ),
            ),
            thresholds=(
                MeasuredValue(
                    name="heading_deviation_max",
                    value=self._params.deviation_max_degrees,
                    unit="degrees",
                ),
                MeasuredValue(
                    name="min_persistence",
                    value=self._params.min_persistence_seconds,
                    unit="seconds",
                ),
            ),
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
                "violation_type": ViolationType.WRONG_WAY.value,
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

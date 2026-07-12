"""Generalized per-track temporal-run reasoner base (P3-U1, composition).

``TemporalRunReasoner`` is the shared machine that the two shipped violation
reasoners were byte-for-byte duplicating: a per-``(camera_id, track_id)``
contiguous *support run* over a boolean per-step signal, driven by the generic
P1-U3 ``RuleEngine`` for hypothesis lifecycle mechanics, that mints exactly one
frozen U2 ``ConfirmedEvent`` once support has persisted for a configured
duration. Each violation reasoner *holds* one of these and feeds it the boolean
support signal it derived from its own observation contract.

Composition, not inheritance
----------------------------
This is a **collaborator the violation reasoners hold and delegate to**, not a
superclass they extend. It carries **no** violation-specific knowledge: the
violation type, the confirmation threshold, the optional gap tolerance, and the
per-event ``measurements`` / ``thresholds`` payload are all *injected* per
instance. A reasoner supplies, per step, a generic ``ObservationBase`` carrier
and the boolean ``active`` support signal it computed from its own observation
contract; everything downstream of that boolean is shared. No rule/predicate/
threshold token from any specific violation appears in this module.

Temporal semantics (timestamp-driven; never wall-clock)
------------------------------------------------------
Per ``(camera_id, track_id)`` a contiguous *support run* is tracked:

* an ``active`` step with no open run starts a run and creates + promotes a
  hypothesis (engine ``NEW -> CANDIDATE``);
* an ``active`` step on an open run attaches to the hypothesis; once
  ``carrier.timestamp - run_start >= threshold_seconds`` the hypothesis is
  activated (``CANDIDATE -> ACTIVE``) and exactly one ``ConfirmedEvent`` is
  emitted for the run;
* an inactive step ends the run -- ``close`` if it had already confirmed
  (``ACTIVE -> CLOSED``), otherwise ``abandon`` (``-> ABANDONED``) -- so recovery
  before the threshold prevents confirmation.

An explicit *taint restart* also ends the current run before the step is
processed, so support never accumulates across a tainted/ID-switch interval
(architecture-review §13: tainted tracks may abstain but never confirm), whereas
an ordinary missing/dropped observation is not a restart and keeps its
timestamp-driven bridging. Because a run needs a later observation than the one
that opened it, confirmation structurally requires at least two observations.

``max_observation_gap_seconds`` is an optional provisional run-break tolerance:
when set, an inter-observation gap wider than it ends the current run (a fresh
run may start at the next active step); when ``None`` the run relies purely on
timestamp bridging. It is inert (never triggers, and the ``last_at`` bookkeeping
it depends on is never read) when unset.

Event identity (deterministic; provisional per ADR-004)
-------------------------------------------------------
``event_id`` is a SHA-256 over canonical JSON of the identity-bearing fields
(scene hash, camera, violation, rule, track ids, start/trigger timestamps, source
hypothesis id) -- the same deterministic, process-independent scheme both shipped
reasoners used, with the ``violation_type`` injected rather than hard-coded.
Run-level ``models`` provenance is stamped onto the event but is deliberately
absent from the id and from every predicate, so the *decision* (which events,
ids, timing) is byte-identical with or without provenance.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from ..contracts import ConfirmedEvent, MeasuredValue, ModelRef
from ..contracts.enums import ViolationType
from ..contracts.observations import ObservationBase
from .engine import HypothesisRecord, RuleEngine
from .states import EngineState


@dataclass(frozen=True)
class ConfirmationDetails:
    """The only two per-violation fields of a confirmed event.

    Everything else on the ``ConfirmedEvent`` (id, violation type, camera, track
    ids, timestamps, rule id/version, scene hash, models, source hypothesis id)
    is invariant across violations and assembled by the base. A reasoner's
    injected detail builder returns this for the confirmed run's timing.
    """

    measurements: tuple[MeasuredValue, ...]
    thresholds: tuple[MeasuredValue, ...]


# A detail builder maps a confirmed run's (start_at, trigger_at) to its
# per-violation measurements/thresholds. It reads no wall-clock and no shared
# mutable state; it is a pure function of the run's timing (plus whatever
# scene-configured parameters the reasoner closed over).
DetailBuilder = Callable[[datetime, datetime], ConfirmationDetails]


# --- per-track run state (engine-internal bookkeeping, not a contract) --------
@dataclass
class _Run:
    hypothesis_id: str
    start_at: datetime
    last_at: datetime
    confirmed: bool = False
    closed: bool = False


class TemporalRunReasoner:
    """Deterministic per-track temporal-run confirmer (see module docstring)."""

    def __init__(
        self,
        engine: RuleEngine,
        *,
        violation_type: ViolationType,
        threshold_seconds: float,
        detail_builder: DetailBuilder,
        scene_config_hash: str | None = None,
        rule_id: str,
        rule_version: str | None,
        models: tuple[ModelRef, ...] = (),
        max_observation_gap_seconds: float | None = None,
    ) -> None:
        self._engine = engine
        self._violation_type = violation_type
        self._threshold_seconds = threshold_seconds
        self._detail_builder = detail_builder
        self._scene_hash = scene_config_hash
        self._rule_id = rule_id
        self._rule_version = rule_version
        # Run-level provenance stamped onto every minted event (P2-U1). Pure
        # metadata: no predicate, threshold, timer, or transition ever reads it,
        # and it is deliberately absent from ``_event_id``, so the *decision*
        # (which events, ids, timing) is byte-identical with or without it. The
        # caller (the composition boundary) supplies the sorted/de-duplicated
        # tuple.
        self._models = models
        self._max_gap = max_observation_gap_seconds
        self._runs: dict[tuple[str, str], _Run] = {}
        self._events: list[ConfirmedEvent] = []

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return tuple(self._events)

    def observe(
        self, carrier: ObservationBase, *, active: bool, is_taint_restart: bool = False
    ) -> ConfirmedEvent | None:
        """Process one step in timestamp order; return any emitted event.

        ``active`` is the per-step support signal the reasoner computed from its
        own observation contract. ``is_taint_restart`` marks the first clean step
        resuming after a tainted interval; it terminates any open run for the
        track *before* processing, so support cannot accumulate across the tainted
        (ID-switch) discontinuity (architecture-review §13). An ordinary
        missing/dropped observation is never a restart and keeps its
        timestamp-driven bridging.
        """

        track_id = carrier.track_id
        if track_id is None:
            return None  # episodes are per-track; ignore untracked facts
        key = (carrier.camera_id, track_id)
        run = self._runs.get(key)
        if is_taint_restart:
            self._on_recovery(run)  # break episode continuity at the taint discontinuity
            run = self._runs.get(key)
        if active:
            return self._on_active(key, run, carrier)
        self._on_recovery(run)
        return None

    def run(
        self,
        steps: Iterable[tuple[ObservationBase, bool]],
        *,
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Process ``(carrier, active)`` steps in deterministic order.

        Steps are sorted by ``(timestamp, observation_id)`` of the carrier and
        de-duplicated by ``observation_id``. ``taint_restart_ids`` are carrier
        observation ids that resume after a tainted interval; each resets the
        track's run before it is processed. Returns the events emitted during this
        call. Ordering follows the P1-U3 policy, so the outcome is independent of
        input order.
        """

        restarts = frozenset(taint_restart_ids)
        ordered = sorted(steps, key=lambda s: (s[0].timestamp, s[0].observation_id))
        seen: set[str] = set()
        emitted: list[ConfirmedEvent] = []
        for carrier, active in ordered:
            observation_id = carrier.observation_id
            if observation_id in seen:
                continue
            seen.add(observation_id)
            event = self.observe(
                carrier, active=active, is_taint_restart=observation_id in restarts
            )
            if event is not None:
                emitted.append(event)
        return tuple(emitted)

    def _on_active(
        self,
        key: tuple[str, str],
        run: _Run | None,
        carrier: ObservationBase,
    ) -> ConfirmedEvent | None:
        # An over-wide inter-observation gap (optional provisional tolerance) ends
        # a stale run; a fresh run may then open at this observation below. When
        # ``max_gap`` is None the check never fires and the ``last_at`` bookkeeping
        # it depends on is inert.
        max_gap = self._max_gap
        if (
            run is not None
            and not run.closed
            and max_gap is not None
            and (carrier.timestamp - run.last_at).total_seconds() > max_gap
        ):
            self._on_recovery(run)
            run = None

        if run is None or run.closed:
            record = self._engine.ingest(
                carrier,
                rule_id=self._rule_id,
                violation_type=self._violation_type,
                rule_version=self._rule_version,
            )
            self._engine.promote(record.hypothesis_id)
            self._runs[key] = _Run(
                hypothesis_id=record.hypothesis_id,
                start_at=carrier.timestamp,
                last_at=carrier.timestamp,
            )
            return None

        record = self._engine.ingest(
            carrier, rule_id=self._rule_id, violation_type=self._violation_type
        )
        run.last_at = carrier.timestamp
        if run.confirmed:
            return None
        elapsed = (carrier.timestamp - run.start_at).total_seconds()
        if elapsed < self._threshold_seconds:
            return None
        if record.state is EngineState.CANDIDATE:
            record = self._engine.activate(record.hypothesis_id)
        event = self._confirm(record, carrier)
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

    def _confirm(self, record: HypothesisRecord, trigger: ObservationBase) -> ConfirmedEvent:
        start_at = record.first_at
        assert start_at is not None  # an attached hypothesis always has a first observation
        trigger_at = trigger.timestamp
        details = self._detail_builder(start_at, trigger_at)
        return ConfirmedEvent(
            event_id=self._event_id(
                record.camera_id, record.track_ids, start_at, trigger_at, record.hypothesis_id
            ),
            violation_type=self._violation_type,
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
            measurements=details.measurements,
            thresholds=details.thresholds,
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
                "violation_type": self._violation_type.value,
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

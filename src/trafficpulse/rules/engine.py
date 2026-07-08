"""Deterministic, rule-agnostic hypothesis-lifecycle engine (P1-U3).

``RuleEngine`` is the first reasoning component of TrafficPulse. It consumes
``Observation`` objects (the frozen U2 perception->reasoning contract) and
manages the lifecycle of :class:`~trafficpulse.contracts.ViolationHypothesis`
candidates. It is **generic**: it knows nothing about wrong-way, red-light,
helmet, speed, or stopping rules, and it embeds no threshold. It only creates,
looks up, updates, transitions, closes, and abandons hypotheses, and it produces
``ViolationHypothesis`` snapshots. It never produces a ``ConfirmedEvent`` or
``EvidenceManifest`` (that is P1-U4) and never reaches ``CONFIRMED``.

Routing
-------
The engine cannot infer *which* rule/violation an observation supports (that is
rule knowledge), so the caller supplies ``rule_id`` and ``violation_type`` when
ingesting. The engine derives ``camera_id``, ``track_id``, and ``timestamp`` from
the observation's base fields only. Observations are grouped by
:class:`HypothesisKey` ``(camera_id, violation_type, rule_id, track_id)``.

Determinism
-----------
* Hypothesis IDs are content-derived (SHA-256 over the key plus a per-key
  generation counter) -- stable across identical replays, distinct across
  re-openings of the same key.
* There is no wall-clock use and no randomness; all timestamps come from the
  observations.
* A hypothesis's interval is order-independent: it always spans the earliest to
  the latest attached observation timestamp. Attached observations are kept
  sorted by ``(timestamp, observation_id)``.
* Duplicate suppression: an observation already attached to a hypothesis is
  ignored (idempotent); while a hypothesis for a key is open no second one is
  created for that key. After a hypothesis becomes terminal, a further
  observation for the same key opens a new (next-generation) hypothesis.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import NamedTuple

from ..contracts import TimeInterval, ViolationHypothesis
from ..contracts.enums import LifecycleState, ViolationType
from ..contracts.observations import ObservationBase
from .states import (
    EngineState,
    RuleEngineError,
    is_terminal,
    to_lifecycle_state,
    validate_transition,
)

_SEP = "\x1f"  # unit separator; avoids delimiter collisions in the ID preimage


class UnknownHypothesisError(RuleEngineError):
    """Raised when an operation references an unknown hypothesis id."""


class HypothesisKey(NamedTuple):
    """The routing identity a hypothesis accumulates under."""

    camera_id: str
    violation_type: ViolationType
    rule_id: str
    track_id: str | None


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """The engine's immutable working snapshot of one hypothesis.

    Mutating operations return a *new* record; the engine replaces the stored
    instance rather than mutating in place. ``ViolationHypothesis`` is derived
    from this via :func:`to_violation_hypothesis`.
    """

    hypothesis_id: str
    camera_id: str
    violation_type: ViolationType
    rule_id: str
    rule_version: str | None
    track_id: str | None
    state: EngineState
    attached: tuple[tuple[datetime, str], ...]
    track_ids: tuple[str, ...]
    first_at: datetime | None
    last_at: datetime | None
    reasons: tuple[str, ...]
    generation: int

    @property
    def key(self) -> HypothesisKey:
        return HypothesisKey(self.camera_id, self.violation_type, self.rule_id, self.track_id)

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(obs_id for _, obs_id in self.attached)

    @property
    def observation_count(self) -> int:
        return len(self.attached)

    @property
    def lifecycle_state(self) -> LifecycleState:
        return to_lifecycle_state(self.state)

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.state)


def _hypothesis_id(key: HypothesisKey, generation: int) -> str:
    """Deterministic, collision-resistant id for a key + generation."""

    preimage = _SEP.join(
        (
            key.camera_id,
            key.violation_type.value,
            key.rule_id,
            key.track_id or "",
            str(generation),
        )
    )
    return "hyp-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def to_violation_hypothesis(record: HypothesisRecord) -> ViolationHypothesis:
    """Materialize the frozen ``ViolationHypothesis`` contract from a record.

    The engine fills only structural/lifecycle fields; ``confidence``,
    ``measurements``, and ``thresholds`` are left at their contract defaults
    because those are rule/confidence concerns, not the engine's.

    Raises:
        ValueError: if the record has no observations to bound its interval
            (never occurs for engine-created records, which always attach the
            observation that created them).
    """

    if record.first_at is None or record.last_at is None:
        raise ValueError("hypothesis has no observations to bound its interval")
    return ViolationHypothesis(
        hypothesis_id=record.hypothesis_id,
        violation_type=record.violation_type,
        camera_id=record.camera_id,
        track_ids=record.track_ids,
        interval=TimeInterval(start=record.first_at, end=record.last_at),
        state=record.lifecycle_state,
        rule_id=record.rule_id,
        rule_version=record.rule_version,
        reasons=record.reasons,
    )


class RuleEngine:
    """Generic, deterministic hypothesis-lifecycle manager (see module docstring)."""

    def __init__(self) -> None:
        self._records: dict[str, HypothesisRecord] = {}
        self._open_by_key: dict[HypothesisKey, str] = {}
        self._generation: dict[HypothesisKey, int] = {}

    # --- ingestion -----------------------------------------------------------
    def ingest(
        self,
        observation: ObservationBase,
        *,
        rule_id: str,
        violation_type: ViolationType,
        rule_version: str | None = None,
    ) -> HypothesisRecord:
        """Route ``observation`` to its hypothesis, creating or attaching.

        Returns the resulting record. If no hypothesis is open for the derived
        key, a new one is created in state ``NEW``; otherwise the observation is
        attached to the open hypothesis (or ignored if already attached).
        """

        key = HypothesisKey(observation.camera_id, violation_type, rule_id, observation.track_id)
        open_id = self._open_by_key.get(key)
        if open_id is not None:
            record = self._records[open_id]
            if observation.observation_id in record.observation_ids:
                return record  # duplicate suppression: idempotent attach
            record = self._attach(record, observation)
        else:
            record = self._attach(self._create(key, rule_version), observation)
        self._records[record.hypothesis_id] = record
        return record

    def ingest_all(
        self,
        observations: Iterable[ObservationBase],
        *,
        rule_id: str,
        violation_type: ViolationType,
        rule_version: str | None = None,
    ) -> tuple[HypothesisRecord, ...]:
        """Ingest many observations for one ``(rule_id, violation_type)``.

        Observations are processed in ``(timestamp, observation_id)`` order so
        the outcome does not depend on iteration order. Returns the distinct
        touched records in deterministic id order.
        """

        ordered = sorted(observations, key=lambda o: (o.timestamp, o.observation_id))
        touched: set[str] = set()
        for obs in ordered:
            record = self.ingest(
                obs, rule_id=rule_id, violation_type=violation_type, rule_version=rule_version
            )
            touched.add(record.hypothesis_id)
        return tuple(self._records[hid] for hid in sorted(touched))

    def _create(self, key: HypothesisKey, rule_version: str | None) -> HypothesisRecord:
        generation = self._generation.get(key, 0)
        self._generation[key] = generation + 1
        hypothesis_id = _hypothesis_id(key, generation)
        record = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            camera_id=key.camera_id,
            violation_type=key.violation_type,
            rule_id=key.rule_id,
            rule_version=rule_version,
            track_id=key.track_id,
            state=EngineState.NEW,
            attached=(),
            track_ids=(),
            first_at=None,
            last_at=None,
            reasons=(),
            generation=generation,
        )
        self._records[hypothesis_id] = record
        self._open_by_key[key] = hypothesis_id
        return record

    @staticmethod
    def _attach(record: HypothesisRecord, observation: ObservationBase) -> HypothesisRecord:
        attached = tuple(
            sorted(record.attached + ((observation.timestamp, observation.observation_id),))
        )
        track_ids = record.track_ids
        if observation.track_id is not None and observation.track_id not in track_ids:
            track_ids = tuple(sorted({*track_ids, observation.track_id}))
        return replace(
            record,
            attached=attached,
            track_ids=track_ids,
            first_at=attached[0][0],
            last_at=attached[-1][0],
        )

    # --- lifecycle transitions ----------------------------------------------
    def transition(
        self, hypothesis_id: str, target: EngineState, *, reason: str | None = None
    ) -> HypothesisRecord:
        """Apply a validated lifecycle transition to a hypothesis.

        Raises:
            UnknownHypothesisError: if ``hypothesis_id`` is not known.
            IllegalTransitionError: if the transition is not permitted.
        """

        record = self._require(hypothesis_id)
        validate_transition(record.state, target)
        reasons = record.reasons + (reason,) if reason is not None else record.reasons
        updated = replace(record, state=target, reasons=reasons)
        self._records[hypothesis_id] = updated
        if is_terminal(target) and self._open_by_key.get(record.key) == hypothesis_id:
            del self._open_by_key[record.key]
        return updated

    def promote(self, hypothesis_id: str, *, reason: str | None = None) -> HypothesisRecord:
        """Transition ``NEW -> CANDIDATE``."""

        return self.transition(hypothesis_id, EngineState.CANDIDATE, reason=reason)

    def activate(self, hypothesis_id: str, *, reason: str | None = None) -> HypothesisRecord:
        """Transition ``CANDIDATE -> ACTIVE``."""

        return self.transition(hypothesis_id, EngineState.ACTIVE, reason=reason)

    def close(self, hypothesis_id: str, *, reason: str | None = None) -> HypothesisRecord:
        """Transition ``ACTIVE -> CLOSED`` (terminal)."""

        return self.transition(hypothesis_id, EngineState.CLOSED, reason=reason)

    def abandon(self, hypothesis_id: str, *, reason: str | None = None) -> HypothesisRecord:
        """Transition any non-terminal state to ``ABANDONED`` (terminal)."""

        return self.transition(hypothesis_id, EngineState.ABANDONED, reason=reason)

    # --- lookup --------------------------------------------------------------
    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        """Return the record for ``hypothesis_id`` or ``None`` if unknown."""

        return self._records.get(hypothesis_id)

    def get_open(self, key: HypothesisKey) -> HypothesisRecord | None:
        """Return the open (non-terminal) record for ``key`` or ``None``."""

        open_id = self._open_by_key.get(key)
        return self._records[open_id] if open_id is not None else None

    def records(self) -> tuple[HypothesisRecord, ...]:
        """Return all records in deterministic (id-sorted) order."""

        return tuple(self._records[hid] for hid in sorted(self._records))

    def snapshot(self) -> tuple[ViolationHypothesis, ...]:
        """Return every hypothesis as a ``ViolationHypothesis``, id-sorted."""

        return tuple(to_violation_hypothesis(record) for record in self.records())

    def _require(self, hypothesis_id: str) -> HypothesisRecord:
        record = self._records.get(hypothesis_id)
        if record is None:
            raise UnknownHypothesisError(f"unknown hypothesis id: {hypothesis_id!r}")
        return record

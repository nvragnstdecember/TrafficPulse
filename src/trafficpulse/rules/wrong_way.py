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
from .engine import RuleEngine
from .temporal import ConfirmationDetails, TemporalRunReasoner

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


class WrongWayReasoner:
    """Deterministic wrong-way temporal reasoner over ``HeadingVsLaneObservation``.

    Wrong-way *semantics* live here: the per-step signal is the observation's
    ``is_contradiction`` flag, and confirmation records a ``persistence_seconds``
    measurement against the ``heading_deviation_max`` / ``min_persistence``
    thresholds. All lifecycle mechanics -- run tracking, taint reset, engine
    transitions, ``models`` stamping, content-derived ``event_id`` -- are
    delegated to the shared :class:`TemporalRunReasoner` this reasoner *holds*
    (P3-U1 composition); the public API is unchanged.
    """

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
        self._params = params
        self._machine = TemporalRunReasoner(
            engine,
            violation_type=ViolationType.WRONG_WAY,
            threshold_seconds=params.min_persistence_seconds,
            detail_builder=self._details,
            scene_config_hash=scene_config_hash,
            rule_id=rule_id,
            rule_version=rule_version,
            models=models,
        )

    @property
    def engine(self) -> RuleEngine:
        return self._machine.engine

    @property
    def events(self) -> tuple[ConfirmedEvent, ...]:
        return self._machine.events

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

        return self._machine.observe(
            observation, active=observation.is_contradiction, is_taint_restart=is_taint_restart
        )

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

        return self._machine.run(
            ((o, o.is_contradiction) for o in observations),
            taint_restart_ids=taint_restart_ids,
        )

    def run_derivation(self, derivation: HeadingDerivation) -> tuple[ConfirmedEvent, ...]:
        """Convenience: run a ``HeadingDerivation`` with its taint restarts."""

        return self.run(derivation.observations, taint_restart_ids=derivation.taint_restart_ids)

    def _details(self, start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
        return ConfirmationDetails(
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

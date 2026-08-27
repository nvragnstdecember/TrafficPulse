"""Red-light-jumping reasoning: latch the signal at the stop line (H13).

The fifth violation slice, and the first whose support is **latched rather than
sustained**. It composes the two observation streams that were written for it --
:mod:`trafficpulse.observations.crossing` (validated stop-line crossing into the
junction) and :mod:`trafficpulse.observations.signal` (declared signal schedule) --
and delegates every lifecycle mechanic to the shared
:class:`~trafficpulse.rules.temporal.TemporalRunReasoner`, exactly as its four
siblings do.

Why the support signal is latched, and why that is not a detail
----------------------------------------------------------------
The other four violations are *sustained conditions*: a vehicle is going the wrong
way, is stopped where it may not stop, a rider has no helmet. Support is true for as
long as the condition holds, so the obvious per-step predicate is the right one.

Red-light jumping is not a condition. It is an **act**, committed at one instant --
the moment the vehicle crosses the stop line against a red signal -- whose
consequence (being in the junction) then persists. The naive predicate

    active = in_junction AND signal_is_red

is therefore wrong, and it fails in the single most common real case: the light
turns green a second after the violator entered. Support collapses, the run is
abandoned, and the violation silently disappears. Worse, it disappears *selectively*
-- only for vehicles that entered late in the red phase, which are precisely the
ones an enforcement system exists to catch. A rule that under-reports the marginal
cases while confidently reporting the flagrant ones is more dangerous than one that
reports nothing.

So the signal state is read **once**, at the forward stop-line crossing, and latched
onto the track:

```
forward stop-line crossing  ->  read signal state  ->  latch
        -> vehicle enters the junction polygon
        -> temporal debounce (min_persistence)
        -> ConfirmedEvent
```

Once latched, no later signal change can invalidate it. The latch clears only when
the vehicle **leaves** the junction, or when an ID-switch taint makes it impossible
to say the entry belonged to this track at all.

Why the crossing instant, not the polygon-entry instant
--------------------------------------------------------
A stop line and the junction it guards are generally not contiguous, so a vehicle
crosses the line several steps before its ground-contact point reaches the polygon.
Reading the signal at polygon entry would exonerate a vehicle that crossed on red
and arrived after the change -- exactly the wrong answer, and for the same class of
vehicle the naive predicate loses. ``CrossingDerivation.forward_crossing_ids``
reports the crossing step, and this join reads the signal there.

What is *not* claimed
----------------------
The signal state is **declared** (a schedule), not perceived: nothing in
TrafficPulse classifies a signal head from pixels. The observations carry a
``HEURISTIC`` producer and every confirmed event records the latched state as a
measurement, so a reviewer can see what the system was told rather than what it saw.
``UNKNOWN`` (before the schedule starts, or where no phase covers the instant) and
``AMBER`` never confirm -- see :func:`join_entry_on_red`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..contracts import (
    ConfirmedEvent,
    InZoneObservation,
    MeasuredValue,
    ModelRef,
    SceneConfig,
    SignalStateObservation,
)
from ..contracts.enums import SignalState, ViolationType
from ..contracts.scene import ParameterStatus
from ..observations.crossing import CrossingDerivation
from .engine import RuleEngine
from .temporal import ConfirmationDetails, TemporalRunReasoner
from .vehicles import VEHICLE_CLASSES as VEHICLE_CLASSES

RULE_ID = "red-light-jumping-v1"
RULE_VERSION = "0.1.0-provisional"

#: ``VEHICLE_CLASSES`` is imported above from :mod:`trafficpulse.rules.vehicles`,
#: which owns the single definition now shared with wrong-way, and is re-exported
#: here so this module's existing consumers keep working unchanged. The rationale
#: -- including why ``person`` and ``bicycle`` are excluded -- lives beside the
#: constant rather than being restated per rule.

#: The only signal state that confirms. ``AMBER`` is deliberately excluded: in most
#: jurisdictions it means "stop if it is safe to do so", which is a judgement about
#: stopping distance and road conditions that no geometry in this system can make.
#: ``UNKNOWN`` and ``OFF`` are excluded because they are the absence of evidence.
CONFIRMING_STATES: frozenset[SignalState] = frozenset({SignalState.RED})


@dataclass(frozen=True)
class RedLightParameters:
    """Scene-configured red-light parameters (provisional; see the scene block).

    ``min_persistence_seconds`` is a **debounce**, not a dwell threshold. The
    violation is complete the moment the vehicle enters the junction on red; the
    window exists only so a single frame of boundary jitter cannot mint an event.
    It is deliberately short, and it is why confirmation still requires at least two
    observations (the shared reasoner's structural guarantee).
    """

    min_persistence_seconds: float
    max_observation_gap_seconds: float | None = None
    persistence_status: ParameterStatus = ParameterStatus.UNSET
    max_observation_gap_status: ParameterStatus = ParameterStatus.UNSET


def red_light_parameters(scene: SceneConfig) -> RedLightParameters:
    """Load the red-light parameter block from a scene.

    Raises:
        ValueError: if the scene declares no ``red_light_jumping`` block, or its
            ``min_persistence`` is absent or unset -- reasoning cannot proceed
            without the debounce window, and defaulting it silently would hide a
            misconfigured scene behind plausible behaviour.
    """

    block = next(
        (b for b in scene.rule_parameters if b.violation_type is ViolationType.RED_LIGHT_JUMPING),
        None,
    )
    if block is None:
        raise ValueError("scene has no red_light_jumping rule-parameter block")
    by_id = {p.id: p for p in block.parameters}
    persistence = by_id.get("min_persistence")
    max_gap = by_id.get("max_observation_gap")
    if persistence is None or persistence.value is None:
        raise ValueError("red_light_jumping min_persistence is unset")
    return RedLightParameters(
        min_persistence_seconds=persistence.value,
        max_observation_gap_seconds=(
            max_gap.value if max_gap is not None and max_gap.value is not None else None
        ),
        persistence_status=persistence.status,
        max_observation_gap_status=(
            max_gap.status if max_gap is not None else ParameterStatus.UNSET
        ),
    )


# --- the latching join ---------------------------------------------------------
@dataclass(frozen=True)
class EntryOnRedStep:
    """One joined per-step fact: the carrier observation plus the latched verdict.

    ``observation`` (the junction-membership ``InZoneObservation``) is the engine
    carrier and the source of the step's ``camera_id`` / ``track_id`` / ``timestamp``
    / ``observation_id``. ``entered_on_red`` is the **latched** support signal:
    true while the track occupies the junction *having entered it against a red
    signal*, regardless of the signal's state now. ``entry_state`` and ``entry_at``
    record what was latched and when, for the confirmed event's measurements.
    """

    observation: InZoneObservation
    entered_on_red: bool
    entry_state: SignalState
    entry_at: datetime | None


def _signal_by_instant(
    signal: Sequence[SignalStateObservation],
) -> dict[tuple[str, datetime], SignalState]:
    """Index the scene-level signal stream by ``(camera_id, timestamp)``.

    Signal observations are scene-level (``track_id`` is ``None``), so unlike the
    illegal-stopping join the key carries no track: one signal fact serves every
    vehicle at that instant. A later observation for the same instant wins, matching
    the schedule resolver's own "later declaration overrides" rule.
    """

    return {(obs.camera_id, obs.timestamp): obs.signal_state for obs in signal}


def join_entry_on_red(
    crossing: CrossingDerivation,
    signal: Sequence[SignalStateObservation],
) -> tuple[list[EntryOnRedStep], frozenset[str]]:
    """Join junction membership with the signal state latched at the stop line.

    Walks each track's crossing observations in ``(timestamp, observation_id)``
    order and maintains, per track:

    * ``pending`` -- the signal state read at the most recent forward stop-line
      crossing, held until the vehicle actually reaches the junction polygon;
    * ``latched`` -- set when membership transitions ``False -> True``, from
      ``pending``. **Never re-evaluated afterwards**: this is the guarantee that a
      light turning green mid-junction cannot un-commit the violation.

    The latch clears when membership returns to ``False`` (the vehicle left the
    junction) and on a taint restart (an ID switch means the entry may not belong to
    this track, and architecture-review §13 forbids confirming across one).

    A membership transition with **no recorded forward crossing** resolves to
    ``UNKNOWN`` and never latches. Without a crossing instant there is no way to say
    what the signal was when the line was passed, and substituting the state at the
    transition would confirm a vehicle whose entry was never validated -- the
    "already inside the junction" case, and the case immediately after a taint
    reset. Both must be silent.

    Returns the steps (the reasoner re-sorts deterministically) and the carrier ids
    that resume after a taint.
    """

    by_instant = _signal_by_instant(signal)
    by_track: dict[str, list[InZoneObservation]] = {}
    for observation in crossing.observations:
        if observation.track_id is None:
            continue  # junction membership is per-track by construction
        by_track.setdefault(observation.track_id, []).append(observation)

    steps: list[EntryOnRedStep] = []
    restart_ids: set[str] = set()
    for track_observations in by_track.values():
        ordered = sorted(track_observations, key=lambda o: (o.timestamp, o.observation_id))
        pending_state: SignalState | None = None
        pending_at: datetime | None = None
        latched = False
        entry_state = SignalState.UNKNOWN
        entry_at: datetime | None = None
        previously_inside = False

        for observation in ordered:
            if observation.observation_id in crossing.taint_restart_ids:
                # The discontinuity may hide an ID switch: neither the crossing nor
                # the entry can be attributed to this track any more.
                pending_state, pending_at = None, None
                latched, entry_state, entry_at = False, SignalState.UNKNOWN, None
                previously_inside = False
                restart_ids.add(observation.observation_id)

            if observation.observation_id in crossing.forward_crossing_ids:
                # The instant the act is committed. Read the signal here -- not at
                # polygon entry, which is later and may be under a changed light.
                pending_state = by_instant.get(
                    (observation.camera_id, observation.timestamp), SignalState.UNKNOWN
                )
                pending_at = observation.timestamp

            if observation.is_inside and not previously_inside:
                # No recorded crossing for this episode means we cannot say when the
                # stop line was passed, so we cannot say what the signal was then.
                # That is the "already inside the junction" case -- and the case
                # immediately after a taint reset -- and both must resolve to UNKNOWN
                # rather than to whatever the signal happens to be right now.
                # Latching on the current instant here would confirm a vehicle whose
                # entry was never validated, which is precisely what the crossing
                # derivation's forward-crossing gate exists to prevent.
                entry_state = pending_state if pending_state is not None else SignalState.UNKNOWN
                entry_at = pending_at
                latched = entry_state in CONFIRMING_STATES
            elif not observation.is_inside:
                # Left the junction (or never entered): the episode is over.
                latched = False
                entry_state = SignalState.UNKNOWN
                entry_at = None

            steps.append(
                EntryOnRedStep(
                    observation=observation,
                    entered_on_red=observation.is_inside and latched,
                    entry_state=entry_state,
                    entry_at=entry_at,
                )
            )
            previously_inside = observation.is_inside

    return steps, frozenset(restart_ids)


# --- the reasoner ---------------------------------------------------------------
class RedLightReasoner:
    """Deterministic red-light reasoner over joined entry-on-red steps.

    Red-light *semantics* live here -- the latched support signal (computed by
    :func:`join_entry_on_red`) and the confirmation's measurements/thresholds. Every
    lifecycle mechanic (run tracking, taint reset, gap break, engine transitions,
    ``models`` stamping, content-derived ``event_id``) is delegated to the shared
    :class:`~trafficpulse.rules.temporal.TemporalRunReasoner` this reasoner *holds*,
    mirroring its four siblings. Nothing about the temporal base is modified or
    re-implemented.
    """

    def __init__(
        self,
        engine: RuleEngine,
        params: RedLightParameters,
        *,
        scene_config_hash: str | None = None,
        rule_id: str = RULE_ID,
        rule_version: str | None = RULE_VERSION,
        models: tuple[ModelRef, ...] = (),
    ) -> None:
        self._params = params
        # The latched entry state of the episode currently being confirmed, so the
        # detail builder can record it. Set immediately before the base confirms and
        # read back inside the callback it invokes -- never read anywhere else.
        self._entry_state = SignalState.UNKNOWN
        self._machine = TemporalRunReasoner(
            engine,
            violation_type=ViolationType.RED_LIGHT_JUMPING,
            threshold_seconds=params.min_persistence_seconds,
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
        self, step: EntryOnRedStep, *, is_taint_restart: bool = False
    ) -> ConfirmedEvent | None:
        """Process one joined step in timestamp order; return any emitted event."""

        self._entry_state = step.entry_state
        return self._machine.observe(
            step.observation, active=step.entered_on_red, is_taint_restart=is_taint_restart
        )

    def run(
        self,
        steps: Iterable[EntryOnRedStep],
        *,
        taint_restart_ids: Iterable[str] = (),
    ) -> tuple[ConfirmedEvent, ...]:
        """Process steps in ``(timestamp, observation_id)`` order, de-duplicated by id.

        The entry state is threaded through a per-step lookup rather than the base's
        ``(carrier, active)`` tuple, because the base is deliberately generic over
        the support signal and must not learn a fifth violation's vocabulary.
        """

        restarts = frozenset(taint_restart_ids)
        ordered = sorted(
            steps, key=lambda s: (s.observation.timestamp, s.observation.observation_id)
        )
        emitted: list[ConfirmedEvent] = []
        seen: set[str] = set()
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
        self, crossing: CrossingDerivation, signal: Sequence[SignalStateObservation]
    ) -> tuple[ConfirmedEvent, ...]:
        """Convenience: join the two streams and run, honouring their taint restarts."""

        steps, restart_ids = join_entry_on_red(crossing, signal)
        return self.run(steps, taint_restart_ids=restart_ids)

    def _details(self, start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
        """The per-violation payload of a confirmed event.

        ``signal_state_at_entry`` is recorded as an ordinal measurement so a
        reviewer can see *what the system was told* the signal was, and audit it
        against the declared schedule. It is not a physical quantity, which is why
        it carries no unit.
        """

        elapsed = (trigger_at - start_at).total_seconds()
        measurements = [
            MeasuredValue(name="persistence_seconds", value=elapsed, unit="seconds"),
            MeasuredValue(
                name="signal_state_at_entry",
                value=float(_STATE_ORDINALS[self._entry_state]),
                unit=None,
            ),
        ]
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
        return ConfirmationDetails(
            measurements=tuple(measurements), thresholds=tuple(thresholds)
        )


# ``MeasuredValue.value`` is numeric, so the latched state is recorded as a stable
# ordinal. Kept beside the enum it encodes so the two cannot drift apart.
_STATE_ORDINALS: dict[SignalState, int] = {
    SignalState.UNKNOWN: 0,
    SignalState.OFF: 1,
    SignalState.GREEN: 2,
    SignalState.AMBER: 3,
    SignalState.RED: 4,
}

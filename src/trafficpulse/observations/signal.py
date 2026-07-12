"""Signal-state context derivation (P3-U3, dynamic traffic context).

Deterministically produces **scene-level** ``SignalStateObservation`` facts from a
**declared / simulated signal schedule** -- the first *dynamic traffic context*
stream in TrafficPulse. Unlike the per-track derivations (heading, in-zone,
stationary), a signal-state observation is a property of the *scene at an instant*,
not of a tracked object: it carries ``track_id = None`` and is keyed on
``(camera_id, timestamp)``.

Declared log, not a learned classifier (offline-honest)
------------------------------------------------------
The signal state is read from a **declared schedule** (a piecewise-constant step
function of media time), not inferred from pixels. This mirrors the detector seam
pattern: a learned ROI signal classifier is a documented *later* concern
(``SignalSourceMode.ROI_CLASSIFIER``), and nothing here pretends a model produced
the state. The producer is therefore an honest ``HEURISTIC``, and the derivation
makes **no** accuracy claim. The schedule is the offline analogue of
``SignalSourceMode.SIMULATED_SCHEDULE`` / ``MANUAL_ANNOTATION``.

Schedule + query timestamps -> observations
-------------------------------------------
A :class:`SignalPhase` declares "state ``state`` is in effect from ``start``
(inclusive)"; a schedule is a sequence of phases forming a step function. The
derivation samples that step function at a caller-supplied set of **query
timestamps** (in a red-light run these are the frame/observation timestamps, so
the context pairs exactly to the per-track carriers at the same instant) and emits
one scene-level ``SignalStateObservation`` per query timestamp.

* :func:`signal_state_at` resolves the state in effect at a timestamp: the latest
  phase whose ``start <= timestamp``. Among phases sharing a ``start`` the
  later-declared one wins (a natural override). **Before the first phase (or for an
  empty schedule) the state is honestly ``SignalState.UNKNOWN``** -- the
  conservative "no evidence" value a downstream rule must not confirm on.

Determinism
-----------
Output is a pure function of the inputs: the schedule is ordered by ``start``
(stably, so equal starts keep declaration order), observations are emitted in the
given ``timestamps`` order, ids are content-derived SHA-256 digests, and no
wall-clock or randomness is used. Replaying the same schedule + timestamps yields
byte-identical observations.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..contracts import Producer, SignalStateObservation
from ..contracts.enums import ProducerKind, SignalState

DEFAULT_SIGNAL_PRODUCER = Producer(
    name="signal-state", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)


@dataclass(frozen=True)
class SignalPhase:
    """One declared phase of a signal schedule.

    ``state`` is in effect from ``start`` (inclusive) until the next phase's
    ``start``. A schedule is a sequence of these forming a step function of media
    time (PTS-anchored, never wall-clock).
    """

    start: datetime
    state: SignalState


def signal_state_at(schedule: Sequence[SignalPhase], timestamp: datetime) -> SignalState:
    """Resolve the signal state in effect at ``timestamp`` from a declared schedule.

    Returns the ``state`` of the latest phase whose ``start <= timestamp`` (equal
    starts: the later-declared phase wins). Before the first phase, or for an empty
    schedule, returns ``SignalState.UNKNOWN`` -- the honest, conservative value a
    downstream rule must never confirm on.
    """

    ordered = sorted(schedule, key=lambda phase: phase.start)  # stable: equal starts keep order
    state = SignalState.UNKNOWN
    for phase in ordered:
        if phase.start <= timestamp:
            state = phase.state
        else:
            break
    return state


def _signal_observation_id(
    camera_id: str, roi_id: str, iso_timestamp: str, state: str
) -> str:
    preimage = "\x1f".join((camera_id, roi_id, iso_timestamp, state))
    return "sig-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def derive_signal_state_observations(
    schedule: Sequence[SignalPhase],
    *,
    timestamps: Sequence[datetime],
    camera_id: str,
    roi_id: str | None = None,
    producer: Producer | None = None,
) -> tuple[SignalStateObservation, ...]:
    """Sample a declared signal ``schedule`` at ``timestamps`` into scene-level facts.

    Emits one ``SignalStateObservation`` per query timestamp (in input order), each
    scene-level (``track_id = None``) and carrying the ``signal_state`` in effect at
    that timestamp (:func:`signal_state_at`; ``UNKNOWN`` before the schedule
    starts). Ids are deterministic content-derived digests.

    Args:
        schedule: the declared/simulated signal phases (a step function). May be
            empty (every timestamp then resolves to ``UNKNOWN``).
        timestamps: the query timestamps to sample at -- in a red-light run, the
            frame/observation timestamps, so the context pairs exactly to the
            per-track carriers at the same instant.
        camera_id: the scene camera the signal governs.
        roi_id: optional signal-ROI id recorded on the observation (e.g. a
            ``SignalGroup`` roi identifier); it does not affect the sampled state.
        producer: observation provenance (defaults to a synthetic heuristic --
            honestly a declared log, not a learned classifier).
    """

    prod = producer if producer is not None else DEFAULT_SIGNAL_PRODUCER
    roi = roi_id if roi_id is not None else ""
    observations: list[SignalStateObservation] = []
    for timestamp in timestamps:
        state = signal_state_at(schedule, timestamp)
        iso_timestamp = timestamp.isoformat()
        observations.append(
            SignalStateObservation(
                observation_id=_signal_observation_id(camera_id, roi, iso_timestamp, state.value),
                camera_id=camera_id,
                track_id=None,  # scene-level: signal state is not track-bound
                timestamp=timestamp,
                producer=prod,
                signal_state=state,
                roi_id=roi_id,
            )
        )
    return tuple(observations)

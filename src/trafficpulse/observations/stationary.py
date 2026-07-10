"""Stationary observation derivation (P2-U3).

Deterministically converts an ordered ``TrackState`` sequence into
``StationaryObservation`` facts, using only P1-U1 geometry. This is *observation
derivation*, not reasoning: it computes a per-step geometric stationarity fact
(``is_stationary``) and makes no dwell, zone, violation, persistence, or
confirmation decision.

Position source
---------------
Stationarity is measured on the **bbox bottom-center** ``((x1 + x2) / 2, y2)`` --
the ground-contact reference (architecture-review §17), the same point P2-U2
in-zone membership uses, so the two evidence streams the illegal-stopping
reasoner joins agree on where the vehicle *is*. This deliberately differs from
the heading derivation's bbox-*center* (heading needs a displacement-direction-
invariant reference); the choice is provisional and revisitable once calibrated
ground-plane reasoning exists.

Method (sliding-window net displacement; pixel space)
-----------------------------------------------------
A step is **stationary** iff the **net** displacement of the bottom-center across
a short *trailing window* is at or below a small pixel-space epsilon
(:data:`STATIONARY_EPSILON_PX`). "Net" means endpoint-to-endpoint: the distance
between the window's first sample and the current sample, not the path length.

A window (rather than a single pairwise step) is used because both in-place
jitter and slow drift have a small *per-step* displacement, yet must read
differently (plan §8b C.2, C.5):

* oscillating in-place **jitter** nets back to roughly its origin over the window
  -> small net displacement -> **stationary** (jitter-robust);
* slow but steady **drift** accumulates over the window -> net displacement grows
  past epsilon -> **moving**.

The window length (:data:`STATIONARY_WINDOW`, a trailing *sample* count) and the
epsilon are **provisional derivation parameters**, exposed as keyword arguments
and labelled provisional. They are a numeric/behavioural pixel-space policy, not
a calibrated physical threshold.

No elapsed time in the decision
-------------------------------
The stationarity decision is a pure pixel/sample-count test: it computes **no**
elapsed time and never uses frame count as elapsed time. ``TrackState.timestamp``
(PTS-anchored media time, never wall-clock) flows through only as the recorded
observation ``timestamp`` (and into the deterministic id). Genuine elapsed-time
accumulation (dwell) is the reasoning layer's job (plan §8b C.6, §9 D.2), so
``dwell_seconds`` is left ``None`` here. Because the decision ignores timestamps,
it is robust to ordinary gaps, sparse sampling, and (pathological) duplicate or
non-monotonic timestamps -- those affect only the recorded metadata, not
``is_stationary``. Timestamp monotonicity/uniqueness per ``(camera_id,
track_id)`` remains a tracker invariant, not defended here.

Uncalibrated-slice honesty: ``motion_threshold``
------------------------------------------------
The scene's ``motion_threshold`` (0.5 m/s, provisional) is **loaded and recorded
for provenance but not applied** in this uncalibrated synthetic slice (plan §8b
C.7/C.8): converting m/s to the pixel space of synthetic tracks needs a validated
calibration that does not exist (``calibration.status: provisional``,
``verification_status: unverified``). It is accepted as an optional argument,
carried inertly on :class:`StationaryDerivation.recorded_motion_threshold`, and
**never** read by any stationarity decision -- exactly mirroring the accepted
wrong-way pattern where ``min_speed`` is carried but not applied and the
usable-motion gate is a geometric pixel-space test. ``speed_estimate`` is left
``None`` (no calibrated speed is claimed).

Two-state minimum, taint handling
----------------------------------
Observations are emitted for the *current* sample of each usable step, so a track
shorter than two clean states yields nothing (mirroring the heading/in-zone
">= 2 states" shape); N clean states yield N-1 observations. A tainted
``TrackState`` drops the step, **resets the trailing window** (so stationarity can
never bridge an ID-switch discontinuity), and the next clean observation is
flagged a **taint restart** via ``StationaryDerivation.taint_restart_ids`` --
reusing the ``HeadingDerivation`` mechanism verbatim (architecture-review §13:
tainted tracks may abstain but never confirm). An ordinary missing/dropped sample
is not a restart and keeps its bridging.

Determinism
-----------
Output is a pure function of the inputs: samples are processed in input order, no
wall-clock, no randomness, no set/hash iteration in the emit path, and the
``TrackState`` sequence is not mutated. Both ``True`` and ``False`` stationarity
facts are emitted -- the illegal-stopping reasoner joins stationarity with in-zone
membership later, so it needs the negative facts too.
"""

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..contracts import Producer, StationaryObservation, TrackState
from ..contracts.enums import ProducerKind
from ..geometry import Point, displacement, magnitude

DEFAULT_STATIONARY_PRODUCER = Producer(
    name="stationary", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC
)

# Provisional trailing-window sample count. The window distinguishes oscillating
# jitter (nets ~0 -> stationary) from slow drift (net accumulates -> moving); it
# is NOT an elapsed-time window. Requires tuning on held-out data.
STATIONARY_WINDOW = 5

# Provisional pixel-space net-displacement epsilon. Net trailing-window
# displacement at or below this reads as stationary. A numeric/behavioural
# pixel-space policy, not a calibrated physical threshold; requires tuning.
STATIONARY_EPSILON_PX = 2.0


@dataclass(frozen=True)
class StationaryDerivation:
    """Observations plus taint restarts and the inert recorded motion threshold.

    ``taint_restart_ids`` are the ``observation_id``s of clean observations that
    immediately follow one or more tainted steps; the reasoner treats them as
    explicit discontinuities and resets its persistence run there.

    ``recorded_motion_threshold`` is the scene ``motion_threshold`` (m/s) carried
    for provenance only. It is **not applied** to any stationarity decision in
    this uncalibrated slice; it is recorded here so the composition boundary can
    propagate it (e.g. onto the event's thresholds) without the observation layer
    ever pretending to convert m/s to pixels.
    """

    observations: tuple[StationaryObservation, ...]
    taint_restart_ids: frozenset[str]
    recorded_motion_threshold: float | None = None


def _bottom_center(track_state: TrackState) -> Point:
    """Ground-contact reference point: bbox bottom-center ``((x1+x2)/2, y2)``."""

    box = track_state.bbox
    return ((box.x1 + box.x2) / 2.0, box.y2)


def _observation_id(camera_id: str, track_id: str, iso_timestamp: str) -> str:
    preimage = "\x1f".join((camera_id, track_id, iso_timestamp))
    return "sta-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


def _iter_derivation(
    track: Sequence[TrackState],
    *,
    window: int,
    epsilon_px: float,
    producer: Producer | None,
) -> Iterator[tuple[StationaryObservation, bool]]:
    """Yield ``(observation, is_taint_restart)`` for each usable step.

    ``is_taint_restart`` is ``True`` for the first clean observation resuming
    after one or more tainted steps. Stationarity is the net bottom-center
    displacement across the trailing ``window`` samples of the current clean run;
    a tainted sample resets that run so the window cannot bridge an ID switch.
    """

    prod = producer if producer is not None else DEFAULT_STATIONARY_PRODUCER
    pending_restart = False
    run: list[TrackState] = []  # current contiguous clean run (window source)

    for state in track:
        if state.tainted:
            pending_restart = True  # abstain on tainted data; mark discontinuity
            run = []  # reset the window: stationarity never bridges an ID switch
            continue
        run.append(state)
        if len(run) < 2:
            continue  # first sample of a run: no prior sample, no observation
        window_start = run[-window:][0]
        net = magnitude(displacement(_bottom_center(window_start), _bottom_center(state)))
        observation = StationaryObservation(
            observation_id=_observation_id(
                state.camera_id, state.track_id, state.timestamp.isoformat()
            ),
            camera_id=state.camera_id,
            track_id=state.track_id,
            timestamp=state.timestamp,
            producer=prod,
            is_stationary=net <= epsilon_px,
            speed_estimate=None,  # no calibrated speed is claimed (uncalibrated slice)
            dwell_seconds=None,  # dwell is a reasoning-layer accumulation, not per-step
        )
        yield observation, pending_restart
        pending_restart = False


def derive_stationary_observations(
    track: Sequence[TrackState],
    *,
    window: int = STATIONARY_WINDOW,
    epsilon_px: float = STATIONARY_EPSILON_PX,
    motion_threshold: float | None = None,
    producer: Producer | None = None,
) -> list[StationaryObservation]:
    """Derive ``StationaryObservation`` facts from a TrackState sequence.

    Returns one observation per usable consecutive step, in input order, emitted
    for the *current* sample; the first sample of each clean run never emits.
    Tainted steps are skipped and reset the trailing window. Use
    :func:`derive_stationary_observations_with_taint` when the taint-discontinuity
    markers (and the recorded motion threshold) are needed for reasoning.

    Args:
        track: ordered TrackStates for a single ``(camera_id, track_id)`` (as
            produced by the P1-U2 synth source or the real tracker).
        window: provisional trailing *sample* count for the net-displacement test
            (``>= 2``). Not an elapsed-time window. Defaults to
            :data:`STATIONARY_WINDOW`.
        epsilon_px: provisional pixel-space net-displacement epsilon (``>= 0``); a
            step is stationary iff its net trailing-window displacement is at or
            below it. Defaults to :data:`STATIONARY_EPSILON_PX`.
        motion_threshold: the scene ``motion_threshold`` (m/s), accepted for
            provenance parity but **not applied** in this uncalibrated slice (see
            the module docstring). It is ignored by this list-returning helper;
            :func:`derive_stationary_observations_with_taint` records it on its
            result.
        producer: observation provenance (defaults to a synthetic heuristic).

    Raises:
        ValueError: if ``window < 2`` (a net-displacement test needs at least two
            samples) or ``epsilon_px < 0``.
    """

    _validate(window, epsilon_px)
    return [
        observation
        for observation, _ in _iter_derivation(
            track, window=window, epsilon_px=epsilon_px, producer=producer
        )
    ]


def derive_stationary_observations_with_taint(
    track: Sequence[TrackState],
    *,
    window: int = STATIONARY_WINDOW,
    epsilon_px: float = STATIONARY_EPSILON_PX,
    motion_threshold: float | None = None,
    producer: Producer | None = None,
) -> StationaryDerivation:
    """Like :func:`derive_stationary_observations`, but also return taint restarts.

    The returned ``taint_restart_ids`` name the observations that resume after a
    tainted interval; the reasoning layer resets its persistence run there so
    stationarity/dwell cannot bridge the tainted (ID-switch) discontinuity. The
    supplied ``motion_threshold`` is carried inertly on
    ``recorded_motion_threshold`` -- recorded for provenance, never applied.

    Raises:
        ValueError: if ``window < 2`` or ``epsilon_px < 0``.
    """

    _validate(window, epsilon_px)
    observations: list[StationaryObservation] = []
    restart_ids: set[str] = set()
    for observation, is_restart in _iter_derivation(
        track, window=window, epsilon_px=epsilon_px, producer=producer
    ):
        observations.append(observation)
        if is_restart:
            restart_ids.add(observation.observation_id)
    return StationaryDerivation(
        tuple(observations), frozenset(restart_ids), recorded_motion_threshold=motion_threshold
    )


def _validate(window: int, epsilon_px: float) -> None:
    if window < 2:
        raise ValueError("window must be >= 2 (a net-displacement test needs two samples)")
    if epsilon_px < 0.0:
        raise ValueError("epsilon_px must be >= 0")

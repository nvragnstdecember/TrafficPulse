"""Illegal-stopping vertical-slice orchestration (P2-U5).

A thin, deterministic **offline** orchestration that runs one recorded stream end
to end through *existing* components -- it wires, it does not compute:

```
FrameRecord (P1-U5 ingestion)
  -> detector Frame (identity + opaque image)
  -> Detector + DetectionAdapter (P1-U6 seam)             -> Detection
  -> Tracker (P1-U8 seam; StubTracker or IouTracker)      -> TrackState
  -> group by (camera_id, track_id) in timestamp order
  -> derive_in_zone_observations_with_taint (P2-U2)       -> InZoneDerivation
  +  derive_stationary_observations_with_taint (P2-U3)    -> StationaryDerivation
  -> IllegalStoppingReasoner.run_join (P2-U4)             -> ConfirmedEvent
```

Sibling to ``WrongWayPipeline``, not a generic runner
-----------------------------------------------------
This is the wrong-way pipeline's structural twin for the second violation: the
detect + track + group + provenance-collect front half is identical, and only the
observation-derivation and reasoning back half differs (two evidence streams
joined by the reasoner instead of one heading stream). Per the Phase 2 plan's
decision (E.8/E.9) a **thin second pipeline** is used rather than prematurely
generalising a multi-rule runner -- two violations do not justify that
abstraction. Wrong-way behaviour is untouched.

Design: thin composition
------------------------
Every stage is an injected or existing component consumed **only** across its
frozen contract seam. This module implements no detection, no association, no
zone geometry, no stationarity, and no rule logic; it re-uses
``DetectionAdapter.adapt_from``, the P2-U2/U3 derivations, and
``IllegalStoppingReasoner.run_join`` -- the composition points that already exist
-- so the wiring provably adds no behaviour (the acceptance test asserts the
pipeline yields the *same* ``ConfirmedEvent`` set as calling the derivations +
reasoner directly on the same ``TrackState``s).

Backend independence
--------------------
The orchestrator depends on the ``Detector`` and ``Tracker`` **abstractions**, the
frozen contracts, and the existing observation/rule APIs -- never on
``RTDetrDetector``, ``StubDetector``, ``IouTracker``, ``StubTracker``, torch,
transformers, or any backend-native object. Imports are taken from the detector
*submodules* (never the ``detector`` package root) and the FrameRecord -> Frame
conversion is reused from :mod:`trafficpulse.pipeline.wrong_way`, so no backend is
pulled into this module's namespace. Any implementation of the two seams drops in
through the constructor unchanged.

Scene configuration (fail-fast)
-------------------------------
The governing illegal-stopping parameters (``stationary_duration``, and the
inertly-recorded ``motion_threshold`` / optional ``max_observation_gap``) are
loaded once at construction via ``illegal_stopping_parameters``; the eligible
enabled no-stopping zones are resolved once too. A scene that declares no
``illegal_stopping`` rule block, no ``stationary_duration``, or no enabled
no-stopping zone fails fast (``ValueError`` / ``SceneConfigurationError``) at
construction -- mirroring ``WrongWayPipeline``'s ``_resolve_legal_direction`` -- so
a misconfigured scene never silently produces zero events.

Stationarity parameters (provisional, pixel-space)
--------------------------------------------------
The stationarity trailing-window sample count and net-displacement epsilon are
provisional *derivation* parameters (not scene parameters); they are surfaced as
constructor arguments defaulting to the P2-U3 module defaults
(:data:`~trafficpulse.observations.stationary.STATIONARY_WINDOW` /
:data:`~trafficpulse.observations.stationary.STATIONARY_EPSILON_PX`) so the
composition boundary makes the provisional pixel-space policy explicit and
configurable without a scene-schema change. ``motion_threshold`` is passed to the
stationary derivation for provenance only and is **never applied** to a
stationarity decision (uncalibrated slice).

Determinism
-----------
No wall-clock, no randomness. Track groups are iterated in ``(camera_id,
track_id)`` order, each track's states in ``(timestamp, frame_index)`` order, the
reasoner processes joined steps in ``(timestamp, observation_id)`` order, and the
emitted events are sorted by ``(trigger_at, event_id)`` -- so the result is a pure
function of the injected components, the frame stream, and the scene, independent
of transient ordering. ``finalize`` builds a fresh reasoner from the scene each
call, so it is idempotent over the accumulated history; ``reset`` returns the
orchestration to a replayable initial state.
"""

from collections.abc import Iterable

from ..contracts import (
    ConfirmedEvent,
    Detection,
    ModelRef,
    SceneConfig,
    TrackState,
    scene_config_hash,
)
from ..contracts.scene import Zone, ZoneType
from ..detector.adapter import DetectionAdapter
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.stationary import (
    STATIONARY_EPSILON_PX,
    STATIONARY_WINDOW,
    derive_stationary_observations_with_taint,
)
from ..observations.zones import derive_in_zone_observations_with_taint
from ..rules.engine import RuleEngine
from ..rules.illegal_stopping import IllegalStoppingReasoner, illegal_stopping_parameters
from ..tracking.interface import Tracker
from .errors import SceneConfigurationError
from .provenance import normalize_model_refs
from .wrong_way import frame_record_to_frame


def _resolve_no_stopping_zones(scene: SceneConfig) -> tuple[Zone, ...]:
    """Resolve the enabled no-stopping zones the illegal-stopping slice reasons over.

    Only ``ZoneType.NO_STOPPING`` zones that are ``enabled`` are eligible (the
    P2-U2 derivation applies the same filter); scene-declaration order is
    preserved for deterministic multi-zone emission.

    Raises:
        SceneConfigurationError: if the scene declares no enabled no-stopping zone
            (there is nothing to reason over; fail fast rather than silently emit
            no events).
    """

    zones = tuple(
        zone
        for zone in scene.zones
        if zone.enabled and zone.zone_type is ZoneType.NO_STOPPING
    )
    if not zones:
        raise SceneConfigurationError(
            "scene declares no enabled no-stopping zone; illegal-stopping "
            "orchestration needs at least one"
        )
    return zones


class IllegalStoppingPipeline:
    """Deterministic offline orchestration for the illegal-stopping vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the existing P2-U2
    in-zone derivation, P2-U3 stationary derivation, and P2-U4 illegal-stopping
    reasoner over one ``SceneConfig``. The ``detector_config`` configures the
    shared ``DetectionAdapter`` seam (label map + provenance). The illegal-stopping
    rule parameters and eligible no-stopping zones are resolved once at
    construction (fail-fast on a misconfigured scene).

    ``stationary_window`` / ``stationary_epsilon_px`` are the provisional
    pixel-space stationarity parameters (see the module docstring); they default to
    the P2-U3 module defaults.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        stationary_window: int = STATIONARY_WINDOW,
        stationary_epsilon_px: float = STATIONARY_EPSILON_PX,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._scene = scene
        self._adapter = DetectionAdapter(detector_config)
        self._params = illegal_stopping_parameters(scene)
        self._zones = _resolve_no_stopping_zones(scene)
        self._stationary_window = stationary_window
        self._stationary_epsilon_px = stationary_epsilon_px
        self._scene_hash = scene_config_hash(scene)
        self._history: dict[tuple[str, str], list[TrackState]] = {}
        # Run-level model provenance accumulated across frames (P2-U1 shape): the
        # distinct truthful ``ModelRef``s the detector/tracker adapters stamp onto
        # ``Detection.source_model`` / ``TrackState.tracker``. Collected here at the
        # composition boundary (the only place that sees both), de-duplicated and
        # ordered in :meth:`finalize`, and stamped onto every minted event. Never
        # read by any reasoning predicate.
        self._model_refs: list[ModelRef] = []

    @property
    def no_stopping_zone_ids(self) -> tuple[str, ...]:
        """The ids of the enabled no-stopping zones this pipeline reasons over."""

        return tuple(zone.zone_id for zone in self._zones)

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state.

        Resets the injected (stateful) ``Tracker``, clears the accumulated
        per-track history, and drops accumulated provenance. The detector needs no
        reset -- it holds no temporal state across frames -- and the
        reasoner/engine are ephemeral (rebuilt per :meth:`finalize`), so the same
        frame stream replays to an equal result.
        """

        self._tracker.reset()
        self._history = {}
        self._model_refs = []

    def process_frame(self, frame_record: FrameRecord) -> tuple[TrackState, ...]:
        """Detect + track one frame, accumulate its states, and return them.

        Runs the detector + adapter (P1-U6 seam) and the tracker (P1-U8 seam) for
        exactly one frame, appends the emitted ``TrackState``s to the per-track
        history grouped by ``(camera_id, track_id)``, and returns that frame's
        states. A zero-detection frame yields ``()`` and changes no track state
        (the empty batch is inert at the tracker seam). Events are computed by
        :meth:`finalize` from the full history, because observation derivation is
        per-track and needs the whole track.
        """

        camera_id = frame_record.camera_id or self._scene.scene.camera_id
        frame = frame_record_to_frame(frame_record, camera_id=camera_id)
        detections: tuple[Detection, ...] = self._adapter.adapt_from(self._detector, frame)
        states = self._tracker.update(detections)
        # Collect truthful run-level provenance from the two seams (P2-U1): the
        # detector's stamped ``source_model`` and the tracker's stamped
        # ``tracker``. ``None`` (a stub that supplied no ref) contributes nothing;
        # de-duplication/ordering is deferred to :meth:`finalize`.
        self._model_refs.extend(d.source_model for d in detections if d.source_model is not None)
        self._model_refs.extend(s.tracker for s in states if s.tracker is not None)
        for state in states:
            self._history.setdefault((state.camera_id, state.track_id), []).append(state)
        return tuple(states)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Derive observations + reason over the accumulated history; return events.

        Groups by ``(camera_id, track_id)``, derives the P2-U2 in-zone and P2-U3
        stationary observation streams per track (over the resolved no-stopping
        zones, with the recorded-but-unapplied ``motion_threshold``), and feeds
        each pair -- taint restarts included -- to a fresh
        ``IllegalStoppingReasoner`` via ``run_join``. Returns the confirmed events
        sorted by ``(trigger_at, event_id)``. The reasoner is built with the
        run-level ``models`` provenance -- the de-duplicated, sorted union of the
        ``ModelRef``s collected during :meth:`process_frame` -- so every minted
        event carries the truthful detector/tracker refs (or an empty tuple when
        the injected components supplied none). Idempotent: it is a pure function of
        the accumulated history + provenance (the reasoner is rebuilt here, not held
        across frames).
        """

        reasoner = IllegalStoppingReasoner(
            RuleEngine(),
            self._params,
            scene_config_hash=self._scene_hash,
            models=normalize_model_refs(self._model_refs),
        )
        events: list[ConfirmedEvent] = []
        for key in sorted(self._history):
            track = sorted(self._history[key], key=lambda s: (s.timestamp, s.frame_index or 0))
            in_zone = derive_in_zone_observations_with_taint(track, zones=self._zones)
            stationary = derive_stationary_observations_with_taint(
                track,
                window=self._stationary_window,
                epsilon_px=self._stationary_epsilon_px,
                motion_threshold=self._params.motion_threshold,
            )
            events.extend(reasoner.run_join(in_zone, stationary))
        return tuple(sorted(events, key=lambda e: (e.trigger_at, e.event_id)))

    def process(self, frames: Iterable[FrameRecord]) -> tuple[ConfirmedEvent, ...]:
        """Run one complete offline stream: ``reset`` -> stream frames -> ``finalize``.

        A self-contained run: it resets first, so repeated calls on one instance
        (and fresh instances) replay an identical frame stream to an equal event
        set. Frames must arrive in ascending ``frame_index`` order (the real
        ingestion order); the tracker seam enforces strict frame monotonicity.
        """

        self.reset()
        for frame_record in frames:
            self.process_frame(frame_record)
        return self.finalize()

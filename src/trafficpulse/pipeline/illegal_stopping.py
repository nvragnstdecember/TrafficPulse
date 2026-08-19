"""Illegal-stopping vertical-slice orchestration (P2-U5; generalized P3-U2).

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

Composition on the shared base (P3-U2)
--------------------------------------
The detect -> track -> group -> provenance-collect front half and the deterministic
``finalize`` scaffold live in the shared
:class:`~trafficpulse.pipeline.base.CompositionPipeline`, which this pipeline
*holds* (composition, not inheritance) and delegates to -- the same base the
wrong-way pipeline uses. Illegal-stopping contributes only its **reasoning back
half** as an injected :class:`FinalizeStrategy` (:class:`_IllegalStoppingFinalize`):
build an ``IllegalStoppingReasoner`` for the run and, per track, derive the two
evidence streams (in-zone + stationary) and join them. Behaviour is unchanged; the
public constructor, methods, and ``no_stopping_zone_ids`` property are identical.
Per the Phase 2 decision (E.8/E.9) this remains a **thin sibling** configuration,
not a generalised multi-rule runner -- the base is a collaborator, not a config
mega-runner.

Design: thin composition
------------------------
Every stage is an injected or existing component consumed **only** across its
frozen contract seam. This module implements no detection, no association, no zone
geometry, no stationarity, and no rule logic; it re-uses the P2-U2/U3 derivations
and ``IllegalStoppingReasoner.run_join`` -- the composition points that already
exist -- so the wiring provably adds no behaviour (the acceptance test asserts the
pipeline yields the *same* ``ConfirmedEvent`` set as calling the derivations +
reasoner directly on the same ``TrackState``s).

Backend independence
--------------------
The orchestrator depends on the ``Detector`` and ``Tracker`` **abstractions**, the
frozen contracts, and the existing observation/rule APIs -- never on a detector or
tracker *backend*. No backend is pulled into this module's namespace; any
implementation of the two seams drops in through the constructor unchanged.

Scene configuration (fail-fast)
-------------------------------
The governing illegal-stopping parameters (``stationary_duration``, and the
inertly-recorded ``motion_threshold`` / optional ``max_observation_gap``) are
loaded once at construction via ``illegal_stopping_parameters``; the eligible
enabled no-stopping zones are resolved once too. A scene that declares no
``illegal_stopping`` rule block, no ``stationary_duration``, or no enabled
no-stopping zone fails fast (``ValueError`` / ``SceneConfigurationError``) at
construction -- mirroring ``WrongWayPipeline`` -- so a misconfigured scene never
silently produces zero events.

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
emitted events are sorted by ``(trigger_at, event_id)``. ``finalize`` builds a
fresh reasoner from the scene each call (idempotent over the accumulated history);
``reset`` returns the orchestration to a replayable initial state.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    ConfirmedEvent,
    InZoneObservation,
    ModelRef,
    SceneConfig,
    StationaryObservation,
    TrackState,
)
from ..contracts.scene import Zone, ZoneType
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
from ..rules.illegal_stopping import (
    IllegalStoppingParameters,
    IllegalStoppingReasoner,
    illegal_stopping_parameters,
)
from ..tracking.interface import Tracker
from .base import _MEDIA_TIME_EPOCH, CompositionPipeline
from .errors import SceneConfigurationError


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


# --- overlay capture (R6) ----------------------------------------------------------
@dataclass(frozen=True)
class IllegalStoppingTrackFrame:
    """One track's state on one frame, as the illegal-stopping slice saw it.

    Recorded during reasoning so the overlay provider can redraw *what the rule
    concluded* without re-running anything or touching pixels -- the pattern H13
    established for red-light. Every field is read off the ``TrackState`` and the
    in-zone / stationary observations the derivations already produced.

    ``zone_id`` names the no-stopping zone the track was found inside on this frame
    (``None`` when it was inside none of them), and ``dwell_seconds`` is the dwell
    the stationarity derivation itself measured -- ``None`` while it has none, never
    a substituted zero.
    """

    frame_index: int
    media_seconds: float
    track_id: str
    bbox: tuple[float, float, float, float]
    zone_id: str | None
    is_inside: bool
    is_stationary: bool
    dwell_seconds: float | None


@dataclass
class IllegalStoppingOverlayCapture:
    """Everything the illegal-stopping overlay provider needs, from reasoning.

    Mutable and owned by the strategy: cleared at the start of every ``finalize``
    (via ``build_reasoner``) so a replayed run captures exactly the run it replayed.

    ``zone_polygons`` are the scene's own no-stopping polygons -- the very geometry
    the rule tested membership against -- so the overlay can show the prohibited
    area rather than implying one.
    """

    zone_polygons: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
    frames: list[IllegalStoppingTrackFrame] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()


@dataclass(frozen=True)
class _IllegalStoppingFinalize:
    """The illegal-stopping reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved run parameters, the eligible no-stopping zones, the
    provisional pixel-space stationarity parameters, and the overlay capture it
    populates as it reasons. Builds an ``IllegalStoppingReasoner`` for the run and,
    per track, derives the P2-U2 in-zone and P2-U3 stationary streams and joins them
    -- the exact operations the pre-generalization ``finalize`` performed.
    """

    params: IllegalStoppingParameters
    zones: tuple[Zone, ...]
    stationary_window: int
    stationary_epsilon_px: float
    capture: IllegalStoppingOverlayCapture

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> IllegalStoppingReasoner:
        # A cleared capture per finalize, so a replay describes only the replayed run.
        self.capture.clear()
        return IllegalStoppingReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: IllegalStoppingReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        in_zone = derive_in_zone_observations_with_taint(track, zones=self.zones)
        stationary = derive_stationary_observations_with_taint(
            track,
            window=self.stationary_window,
            epsilon_px=self.stationary_epsilon_px,
            motion_threshold=self.params.motion_threshold,
        )
        self._capture(track, in_zone.observations, stationary.observations)
        return reasoner.run_join(in_zone, stationary)

    def _capture(
        self,
        track: Sequence[TrackState],
        in_zone: Sequence[InZoneObservation],
        stationary: Sequence[StationaryObservation],
    ) -> None:
        """Record per-frame geometry + zone/dwell verdict for the overlay.

        The two streams are joined back onto the track by timestamp -- the same key
        the reasoner's own join uses. A frame present in neither stream is not
        recorded: the rule formed no view of it, so the overlay states none.
        """

        inside_at: dict[object, InZoneObservation] = {}
        for observation in in_zone:
            # A track can be inside several no-stopping zones on one frame; the one
            # that matters is any zone it is actually inside, so a positive wins.
            current = inside_at.get(observation.timestamp)
            if current is None or (observation.is_inside and not current.is_inside):
                inside_at[observation.timestamp] = observation
        stationary_at = {observation.timestamp: observation for observation in stationary}

        for state in track:
            if state.frame_index is None:
                continue
            zone = inside_at.get(state.timestamp)
            still = stationary_at.get(state.timestamp)
            if zone is None and still is None:
                continue
            box = state.bbox
            self.capture.frames.append(
                IllegalStoppingTrackFrame(
                    frame_index=state.frame_index,
                    media_seconds=(state.timestamp - _MEDIA_TIME_EPOCH).total_seconds(),
                    track_id=state.track_id,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    zone_id=zone.zone_id if zone is not None and zone.is_inside else None,
                    is_inside=zone.is_inside if zone is not None else False,
                    is_stationary=still.is_stationary if still is not None else False,
                    dwell_seconds=still.dwell_seconds if still is not None else None,
                )
            )


def illegal_stopping_finalize_strategy(
    scene: SceneConfig,
    *,
    stationary_window: int = STATIONARY_WINDOW,
    stationary_epsilon_px: float = STATIONARY_EPSILON_PX,
) -> _IllegalStoppingFinalize:
    """Build the illegal-stopping reasoning back half for one scene (public factory).

    The exact strategy ``IllegalStoppingPipeline`` injects into the shared
    ``CompositionPipeline`` -- exposed so a multi-rule composition (the real-time
    engine) can run this rule alongside others over **one** shared detect+track
    front half. Applies the same fail-fast scene resolution as the pipeline
    constructor.

    Raises:
        SceneConfigurationError: if the scene declares no enabled no-stopping zone.
        ValueError: if the scene declares no usable ``illegal_stopping`` block.
    """

    zones = _resolve_no_stopping_zones(scene)
    return _IllegalStoppingFinalize(
        params=illegal_stopping_parameters(scene),
        zones=zones,
        stationary_window=stationary_window,
        stationary_epsilon_px=stationary_epsilon_px,
        # The strategy owns its capture (rather than returning it alongside, as
        # red-light does) so this factory's signature -- which capability probing
        # and every existing caller depend on -- is unchanged. Drivers read
        # ``strategy.capture``.
        capture=IllegalStoppingOverlayCapture(
            zone_polygons=tuple(
                (zone.zone_id, tuple(zone.polygon)) for zone in zones
            )
        ),
    )


class IllegalStoppingPipeline:
    """Deterministic offline orchestration for the illegal-stopping vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the existing P2-U2
    in-zone derivation, P2-U3 stationary derivation, and P2-U4 illegal-stopping
    reasoner over one ``SceneConfig``. The ``detector_config`` configures the
    shared ``DetectionAdapter`` seam (label map + provenance). The illegal-stopping
    rule parameters and eligible no-stopping zones are resolved once at
    construction (fail-fast on a misconfigured scene).

    The shared orchestration is delegated to a held
    :class:`~trafficpulse.pipeline.base.CompositionPipeline`; this class contributes
    the illegal-stopping reasoning strategy and the zone resolution/fail-fast.

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
        strategy = illegal_stopping_finalize_strategy(
            scene,
            stationary_window=stationary_window,
            stationary_epsilon_px=stationary_epsilon_px,
        )
        self._zones = strategy.zones
        self._core = CompositionPipeline(
            detector=detector,
            tracker=tracker,
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=strategy,
        )

    @property
    def no_stopping_zone_ids(self) -> tuple[str, ...]:
        """The ids of the enabled no-stopping zones this pipeline reasons over."""

        return tuple(zone.zone_id for zone in self._zones)

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state (delegated)."""

        self._core.reset()

    def process_frame(self, frame_record: FrameRecord) -> tuple[TrackState, ...]:
        """Detect + track one frame, accumulate its states, and return them (delegated)."""

        return self._core.process_frame(frame_record)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Derive + reason over the accumulated history; return events (delegated)."""

        return self._core.finalize()

    def process(self, frames: Iterable[FrameRecord]) -> tuple[ConfirmedEvent, ...]:
        """Run one complete offline stream: ``reset`` -> stream frames -> ``finalize``."""

        return self._core.process(frames)

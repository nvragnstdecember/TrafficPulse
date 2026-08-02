"""Red-light-jumping vertical-slice orchestration (H13).

The thin sibling of ``WrongWayPipeline`` / ``IllegalStoppingPipeline`` /
``NoHelmetPipeline`` / ``TripleRidingPipeline``: it wires existing components
across their frozen seams and computes nothing itself.

```
FrameRecord -> Frame -> Detector + DetectionAdapter -> Detection
  -> Tracker                                          -> TrackState
  -> group by (camera_id, track_id) in timestamp order
  -> derive_crossing_observations_with_taint (P3-U4)   -> junction membership
  +  derive_signal_state_observations (P3-U3)          -> declared signal state
  -> join_entry_on_red (H13)                           -> latched support
  -> RedLightReasoner.run                              -> ConfirmedEvent
```

Scene resolution (fail-fast)
-----------------------------
The governing ``(stop_line, junction_zone, signal_group)`` triple is resolved once
at construction. A scene with no stop line, no intersection / signal-controlled
zone, or an ambiguous choice among several fails immediately -- mirroring
``WrongWayPipeline``'s treatment of multiple legal directions -- so a misconfigured
scene never silently produces zero events.

Signal timing is per-run, not per-scene
----------------------------------------
The schedule is injected by the caller, **not** read from ``SceneConfig``. A
``SignalPhase`` names a media-time instant, which is a property of one clip; a
``SceneConfig`` is per-camera and (since H12) content-addressed and shared across
many videos. Embedding per-clip offsets in a shared scene would mint a new scene
revision per upload and misstate what a scene is. The scene stays camera geometry;
the rule config carries the timing.

Signal observations are sampled at exactly the crossing stream's timestamps, so the
scene-level context pairs to the per-track carriers instant-for-instant with no
interpolation and no nearest-neighbour guessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from ..contracts import ConfirmedEvent, ModelRef, SceneConfig, TrackState
from ..contracts.enums import ObjectClass, SignalState
from ..contracts.scene import StopLine, Zone, ZoneType
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.crossing import derive_crossing_observations_with_taint
from ..observations.signal import SignalPhase, derive_signal_state_observations
from ..rules.engine import RuleEngine
from ..rules.red_light import (
    VEHICLE_CLASSES,
    EntryOnRedStep,
    RedLightParameters,
    RedLightReasoner,
    join_entry_on_red,
    red_light_parameters,
)
from ..tracking.interface import Tracker
from .base import _MEDIA_TIME_EPOCH, CompositionPipeline
from .errors import SceneConfigurationError

__all__ = [
    "RedLightPipeline",
    "RedLightOverlayCapture",
    "RedLightTrackFrame",
    "red_light_finalize_strategy",
    "resolve_red_light_geometry",
    "phases_from_offsets",
]


def resolve_red_light_geometry(
    scene: SceneConfig,
    *,
    stop_line_id: str | None = None,
    zone_id: str | None = None,
) -> tuple[StopLine, Zone]:
    """Resolve the single governing ``(stop_line, junction_zone)`` for the slice.

    A stop line names the signal group that governs it, and the zones it guards.
    When the scene declares exactly one enabled stop line and one junction zone the
    ids may be omitted; otherwise they must be supplied, because guessing which
    approach an operator meant would silently reason about the wrong one.

    Raises:
        SceneConfigurationError: no enabled stop line, no junction /
            signal-controlled zone, an unknown id, or an ambiguous choice.
    """

    lines = tuple(line for line in scene.stop_lines if line.enabled)
    if not lines:
        raise SceneConfigurationError(
            "scene declares no enabled stop line; red-light reasoning needs one"
        )
    if stop_line_id is not None:
        match = next((line for line in lines if line.stop_line_id == stop_line_id), None)
        if match is None:
            raise SceneConfigurationError(
                f"scene has no enabled stop line with stop_line_id={stop_line_id!r}"
            )
        stop_line = match
    elif len(lines) > 1:
        available = tuple(line.stop_line_id for line in lines)
        raise SceneConfigurationError(
            "scene declares more than one enabled stop line; red-light reasoning "
            f"needs an explicit stop_line_id (available: {available})"
        )
    else:
        stop_line = lines[0]

    junction_types = (ZoneType.INTERSECTION, ZoneType.SIGNAL_CONTROLLED_REGION)
    candidates = tuple(
        zone for zone in scene.zones if zone.enabled and zone.zone_type in junction_types
    )
    # A stop line that names the zones it guards narrows the choice for us; that is
    # what the contract's `stop_line.zone_ids` is for, and honouring it means a
    # multi-approach scene needs no extra configuration here.
    if stop_line.zone_ids:
        guarded = tuple(zone for zone in candidates if zone.zone_id in stop_line.zone_ids)
        if guarded:
            candidates = guarded
    if not candidates:
        raise SceneConfigurationError(
            "scene declares no enabled intersection or signal-controlled zone; "
            "red-light reasoning needs the junction the stop line guards"
        )
    if zone_id is not None:
        zone_match = next((zone for zone in candidates if zone.zone_id == zone_id), None)
        if zone_match is None:
            raise SceneConfigurationError(
                f"scene has no enabled junction zone with zone_id={zone_id!r}"
            )
        return stop_line, zone_match
    if len(candidates) > 1:
        raise SceneConfigurationError(
            "scene declares more than one enabled junction zone; red-light reasoning "
            f"needs an explicit zone_id (available: {tuple(z.zone_id for z in candidates)})"
        )
    return stop_line, candidates[0]


# --- overlay capture -------------------------------------------------------------
@dataclass(frozen=True)
class RedLightTrackFrame:
    """One track's state on one frame, as the red-light slice saw it.

    Recorded during reasoning so the overlay provider can redraw *what the rule
    concluded* without re-running anything or accessing pixels. Red-light needs no
    ``FrameObserver``: it reasons purely over ``TrackState`` geometry, which the
    finalize strategy already holds.
    """

    frame_index: int
    media_seconds: float
    track_id: str
    bbox: tuple[float, float, float, float]
    is_inside: bool
    entered_on_red: bool
    entry_state: SignalState


@dataclass
class RedLightOverlayCapture:
    """Everything the red-light overlay provider needs, produced by reasoning.

    Mutable and owned by the strategy: it is cleared at the start of every
    ``finalize`` (via ``build_reasoner``) so a replayed run captures exactly the run
    it replayed, never an accumulation of both.
    """

    stop_line: tuple[tuple[float, float], tuple[float, float]]
    zone_polygon: tuple[tuple[float, float], ...]
    frames: list[RedLightTrackFrame] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()


# --- the reasoning back half -------------------------------------------------------
@dataclass
class _RedLightFinalize:
    """The red-light reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved geometry, the per-run signal schedule, and the overlay
    capture it populates as it reasons.
    """

    params: RedLightParameters
    stop_line: StopLine
    zone: Zone
    schedule: tuple[SignalPhase, ...]
    signal_roi_id: str | None
    vehicle_classes: frozenset[ObjectClass]
    capture: RedLightOverlayCapture

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> RedLightReasoner:
        # A fresh reasoner per finalize (matching every sibling), and a cleared
        # capture so a replay describes only the replayed run.
        self.capture.clear()
        return RedLightReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: RedLightReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        if not track:
            return ()
        if track[0].object_class not in self.vehicle_classes:
            # Pedestrians and cyclists are detected (helmet reasoning needs riders)
            # but cannot commit this violation; see VEHICLE_CLASSES.
            return ()

        crossing = derive_crossing_observations_with_taint(
            track, stop_line=self.stop_line, zone=self.zone
        )
        if not crossing.observations:
            return ()

        signal = derive_signal_state_observations(
            self.schedule,
            timestamps=[obs.timestamp for obs in crossing.observations],
            camera_id=track[0].camera_id,
            roi_id=self.signal_roi_id,
        )
        steps, restart_ids = join_entry_on_red(crossing, signal)
        self._capture(track, steps)
        return reasoner.run(steps, taint_restart_ids=restart_ids)

    def _capture(
        self, track: Sequence[TrackState], steps: Sequence[EntryOnRedStep]
    ) -> None:
        """Record per-frame geometry + verdict for the overlay (no recomputation)."""

        by_timestamp = {state.timestamp: state for state in track}
        for step in steps:
            state = by_timestamp.get(step.observation.timestamp)
            if state is None or state.frame_index is None:
                continue
            box = state.bbox
            self.capture.frames.append(
                RedLightTrackFrame(
                    frame_index=state.frame_index,
                    media_seconds=(state.timestamp - _MEDIA_TIME_EPOCH).total_seconds(),
                    track_id=state.track_id,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    is_inside=step.observation.is_inside,
                    entered_on_red=step.entered_on_red,
                    entry_state=step.entry_state,
                )
            )


def red_light_finalize_strategy(
    scene: SceneConfig,
    *,
    schedule: Sequence[SignalPhase],
    stop_line_id: str | None = None,
    zone_id: str | None = None,
    vehicle_classes: frozenset[ObjectClass] = VEHICLE_CLASSES,
) -> tuple[_RedLightFinalize, RedLightOverlayCapture]:
    """Build the red-light back half for one scene + run (public factory).

    Returns the strategy **and** the overlay capture it populates, so a composition
    root that wants an annotated video can hand the capture to the provider -- the
    same shape the observer-bearing rules use, without needing a pixel observer.

    Raises:
        SceneConfigurationError: the governing geometry cannot be resolved.
        ValueError: the scene declares no usable ``red_light_jumping`` parameters.
    """

    params = red_light_parameters(scene)
    stop_line, zone = resolve_red_light_geometry(
        scene, stop_line_id=stop_line_id, zone_id=zone_id
    )
    group = next(
        (g for g in scene.signal_groups if g.signal_group_id == stop_line.signal_group_id), None
    )
    capture = RedLightOverlayCapture(
        stop_line=(stop_line.endpoints.a, stop_line.endpoints.b),
        zone_polygon=zone.polygon,
    )
    return (
        _RedLightFinalize(
            params=params,
            stop_line=stop_line,
            zone=zone,
            schedule=tuple(schedule),
            signal_roi_id=group.signal_group_id if group is not None else None,
            vehicle_classes=vehicle_classes,
            capture=capture,
        ),
        capture,
    )


class RedLightPipeline:
    """Deterministic offline orchestration for the red-light vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the crossing + signal
    derivations and the red-light reasoner over one ``SceneConfig`` and one run's
    signal schedule. Geometry and parameters are resolved once at construction
    (fail-fast); the shared orchestration is delegated to a held
    ``CompositionPipeline``.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        schedule: Sequence[SignalPhase],
        stop_line_id: str | None = None,
        zone_id: str | None = None,
    ) -> None:
        strategy, capture = red_light_finalize_strategy(
            scene, schedule=schedule, stop_line_id=stop_line_id, zone_id=zone_id
        )
        self._capture = capture
        self._core = CompositionPipeline(
            detector=detector,
            tracker=tracker,
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=strategy,
        )

    @property
    def overlay_capture(self) -> RedLightOverlayCapture:
        return self._capture

    def process_frame(self, record: FrameRecord) -> None:
        self._core.process_frame(record)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        return self._core.finalize()

    def reset(self) -> None:
        self._core.reset()


def phases_from_offsets(
    offsets: Sequence[tuple[float, SignalState]],
) -> tuple[SignalPhase, ...]:
    """Convert ``(media_seconds, state)`` pairs into media-time ``SignalPhase``\\ s.

    The bridge between how an operator *states* a schedule (seconds from the start
    of the clip, read off a player's scrub bar) and how the observation layer
    consumes it (absolute media-time instants anchored at the fixed epoch). Keeping
    the conversion in one place is what stops an offset and an instant from being
    confused at a call site.
    """

    return tuple(
        SignalPhase(start=_MEDIA_TIME_EPOCH + timedelta(seconds=at), state=state)
        for at, state in offsets
    )

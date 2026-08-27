"""Wrong-way vertical-slice orchestration (P1-U10; generalized P3-U2).

A thin, deterministic **offline** orchestration that runs one recorded stream end
to end through *existing* components -- it wires, it does not compute:

```
FrameRecord (P1-U5 ingestion)
  -> detector Frame (identity + opaque image)
  -> Detector + DetectionAdapter (P1-U6 seam)          -> Detection
  -> Tracker (P1-U8 seam; StubTracker or IouTracker)   -> TrackState
  -> group by (camera_id, track_id) in timestamp order
  -> derive_heading_observations_with_taint (P1-U4)    -> HeadingDerivation
  -> WrongWayReasoner.run_derivation (P1-U4)            -> ConfirmedEvent
```

Composition on the shared base (P3-U2)
--------------------------------------
The detect -> track -> group -> provenance-collect front half and the deterministic
``finalize`` scaffold now live in the shared
:class:`~trafficpulse.pipeline.base.CompositionPipeline`, which this pipeline
*holds* (composition, not inheritance) and delegates to. Wrong-way contributes
only its **reasoning back half** as an injected :class:`FinalizeStrategy`
(:class:`_WrongWayFinalize`): build a ``WrongWayReasoner`` for the run and, per
track, derive the heading stream and reason over it. Behaviour is unchanged; the
public constructor, methods, and ``lane_id`` property are identical.

Design: thin composition
------------------------
Every stage is an injected or existing component consumed **only** across its
frozen contract seam. This module implements no detection, no association, no
heading calculation, and no rule logic; it re-uses ``DetectionAdapter.adapt_from``
and ``WrongWayReasoner.run_derivation`` -- the composition points that already
exist -- so the wiring provably adds no behaviour (the acceptance test asserts the
pipeline yields the *same* ``ConfirmedEvent`` set as calling the derivation +
reasoner directly on the same ``TrackState``s).

Backend independence
--------------------
The orchestrator depends on the ``Detector`` and ``Tracker`` **abstractions**, the
frozen contracts, and the existing observation/rule APIs -- never on
``RTDetrDetector``, ``StubDetector``, ``IouTracker``, ``StubTracker``, torch,
transformers, or any backend-native object. Any implementation of the two seams
drops in through the constructor unchanged.

The FrameRecord -> detector Frame conversion (:func:`frame_record_to_frame`, at a
fixed media-time epoch) is owned by the shared base and re-exported here for the
call sites and tests that reference it through this module.

Determinism
-----------
No wall-clock, no randomness. Track groups are iterated in ``(camera_id,
track_id)`` order, each track's states in ``(timestamp, frame_index)`` order, and
the emitted events sorted by ``(trigger_at, event_id)`` -- so the result is a pure
function of the injected components, the frame stream, and the scene. ``finalize``
builds a fresh reasoner from the scene each call (idempotent over the accumulated
history); ``reset`` returns the orchestration to a replayable initial state.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..contracts import (
    ConfirmedEvent,
    HeadingVsLaneObservation,
    ModelRef,
    SceneConfig,
    TrackState,
)
from ..contracts.enums import ObjectClass
from ..contracts.scene import DirectionVector, LegalDirection
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..geometry import Point
from ..ingestion.video import FrameRecord
from ..observations.heading import derive_heading_observations_with_taint
from ..rules.engine import RuleEngine
from ..rules.vehicles import VEHICLE_CLASSES
from ..rules.wrong_way import WrongWayParameters, WrongWayReasoner, wrong_way_parameters
from ..tracking.interface import Tracker
from .base import _MEDIA_TIME_EPOCH, CompositionPipeline, frame_record_to_frame
from .errors import SceneConfigurationError

# ``frame_record_to_frame`` and ``_MEDIA_TIME_EPOCH`` are defined in the shared
# base and re-exported here (listed in ``__all__``) so the callers and tests that
# reference them through :mod:`trafficpulse.pipeline.wrong_way` resolve unchanged.
__all__ = [
    "WrongWayOverlayCapture",
    "WrongWayPipeline",
    "WrongWayTrackFrame",
    "frame_record_to_frame",
    "wrong_way_finalize_strategy",
    "_MEDIA_TIME_EPOCH",
]


def _resolve_legal_direction(
    scene: SceneConfig, direction_id: str | None
) -> tuple[DirectionVector, str, tuple[Point, ...]]:
    """Resolve the governing ``(legal_direction, lane_id, lane_polygon)``.

    The polygon is resolved here, beside the lane id it belongs to, because the
    derivation must evaluate headings **only inside the lane the direction
    governs** (architecture-review §5a). Returning the id alone is what allowed
    the direction to be applied to every track in frame, including lawful traffic
    on an opposing carriageway.

    Raises:
        SceneConfigurationError: if the single lane cannot be picked (no legal
            direction; more than one with no ``direction_id``; an unknown
            ``direction_id``; or the chosen direction has no zone/lane id), or if
            the zone it names is absent from the scene or disabled -- both leave
            the run with no lane to contain reasoning to, and silently reasoning
            over the whole frame instead is the bug this guard exists to prevent.
    """

    directions = scene.legal_directions
    chosen: LegalDirection
    if direction_id is not None:
        match = next((d for d in directions if d.direction_id == direction_id), None)
        if match is None:
            raise SceneConfigurationError(
                f"scene has no legal direction with direction_id={direction_id!r}"
            )
        chosen = match
    elif not directions:
        raise SceneConfigurationError(
            "scene declares no legal direction; wrong-way orchestration needs one"
        )
    elif len(directions) > 1:
        available = tuple(d.direction_id for d in directions)
        raise SceneConfigurationError(
            "scene declares more than one legal direction; the single-lane slice "
            f"needs an explicit direction_id (available: {available})"
        )
    else:
        chosen = directions[0]

    if not chosen.zone_ids:
        raise SceneConfigurationError(
            f"legal direction {chosen.direction_id!r} carries no zone/lane id"
        )
    lane_id = chosen.zone_ids[0]
    zone = next((z for z in scene.zones if z.zone_id == lane_id), None)
    if zone is None:
        raise SceneConfigurationError(
            f"legal direction {chosen.direction_id!r} governs zone {lane_id!r}, "
            "which the scene does not declare; wrong-way reasoning needs its "
            "polygon to know which traffic the direction applies to"
        )
    if not zone.enabled:
        raise SceneConfigurationError(
            f"legal direction {chosen.direction_id!r} governs zone {lane_id!r}, "
            "which is disabled; wrong-way reasoning has no lane to contain to"
        )
    return chosen.vector, lane_id, tuple(zone.polygon)


# --- overlay capture (R6) ----------------------------------------------------------
@dataclass(frozen=True)
class WrongWayTrackFrame:
    """One track's state on one frame, as the wrong-way slice saw it.

    Recorded during reasoning so the overlay provider can redraw *what the rule
    concluded* without re-running anything or touching pixels -- the pattern H13
    established for red-light. Every field is read straight off the
    ``TrackState`` and the ``HeadingVsLaneObservation`` the derivation already
    produced; nothing here is recomputed or estimated.

    ``heading_degrees`` is the direction the vehicle was **actually** measured
    travelling and ``legal_heading_degrees`` the direction the lane declares (``None``
    when the scene left it unstated -- an honest absence, never a fabricated arrow).
    ``is_contradiction`` is the per-frame verdict the reasoner accumulated.
    """

    frame_index: int
    media_seconds: float
    track_id: str
    bbox: tuple[float, float, float, float]
    heading_degrees: float
    legal_heading_degrees: float | None
    deviation_degrees: float
    is_contradiction: bool


@dataclass
class WrongWayOverlayCapture:
    """Everything the wrong-way overlay provider needs, produced by reasoning.

    Mutable and owned by the strategy: cleared at the start of every ``finalize``
    (via ``build_reasoner``) so a replayed run captures exactly the run it replayed,
    never an accumulation of both -- identical to ``RedLightOverlayCapture``.

    ``lane_id`` and ``legal_direction`` are the scene facts the rule was configured
    with, carried so the provider can state which lane governed the decision.
    """

    lane_id: str
    legal_direction: tuple[float, float]
    frames: list[WrongWayTrackFrame] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()


@dataclass(frozen=True)
class _WrongWayFinalize:
    """The wrong-way reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved run parameters (deviation threshold, min persistence,
    boundary abstain margin), the single governing legal direction / lane / lane
    polygon, the classes that may commit the violation, and the overlay capture it
    populates as it reasons.

    Two gates decide whether a track is this rule's business at all, and both
    exist because it was reasoning about traffic it should never have scored:
    **who** (``vehicle_classes`` -- a pedestrian is not committing a vehicle
    movement violation) and **where** (``lane_polygon`` -- a track outside the
    governed lane yields no heading observations, so it can never support a run).

    Builds a ``WrongWayReasoner`` for the run and, per track, derives the P1-U4
    heading stream and reasons over it -- the exact operations the
    pre-generalization ``finalize`` performed.
    """

    params: WrongWayParameters
    legal_direction: DirectionVector
    lane_id: str
    capture: WrongWayOverlayCapture
    #: Polygon of the lane ``lane_id`` names. Reasoning is contained to it, so a
    #: track outside the governed lane produces no heading facts at all.
    lane_polygon: tuple[Point, ...] = ()
    #: The classes that can commit this violation. A pedestrian's track opposes the
    #: traffic direction as a matter of course; scoring it against a vehicle's
    #: thresholds is a false positive waiting on a threshold. See VEHICLE_CLASSES.
    vehicle_classes: frozenset[ObjectClass] = VEHICLE_CLASSES

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> WrongWayReasoner:
        # A cleared capture per finalize, so a replay describes only the replayed
        # run (matching the fresh-reasoner guarantee beside it).
        self.capture.clear()
        return WrongWayReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: WrongWayReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        if not track:
            return ()
        if track[0].object_class not in self.vehicle_classes:
            # Pedestrians and cyclists are detected (helmet and triple-riding
            # reasoning need riders) but cannot commit this violation; see
            # VEHICLE_CLASSES. Returning before the derivation also keeps them out
            # of the overlay capture, so the annotated video does not show a
            # heading verdict the rule never intended to reach.
            return ()

        derivation = derive_heading_observations_with_taint(
            track,
            legal_direction=self.legal_direction,
            lane_id=self.lane_id,
            deviation_max_degrees=self.params.deviation_max_degrees,
            lane_polygon=self.lane_polygon,
            boundary_abstain_margin=self.params.boundary_abstain_margin,
        )
        self._capture(track, derivation.observations)
        return reasoner.run_derivation(derivation)

    def _capture(
        self,
        track: Sequence[TrackState],
        observations: Sequence[HeadingVsLaneObservation],
    ) -> None:
        """Record per-frame geometry + heading verdict for the overlay.

        Pairs each observation back to the ``TrackState`` it was derived from by
        timestamp -- the same join red-light uses. A state with no frame index
        cannot be drawn on a frame and is skipped rather than guessed at.
        """

        by_timestamp = {state.timestamp: state for state in track}
        for observation in observations:
            state = by_timestamp.get(observation.timestamp)
            if state is None or state.frame_index is None:
                continue
            box = state.bbox
            self.capture.frames.append(
                WrongWayTrackFrame(
                    frame_index=state.frame_index,
                    media_seconds=(state.timestamp - _MEDIA_TIME_EPOCH).total_seconds(),
                    track_id=state.track_id,
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    heading_degrees=observation.heading_degrees,
                    legal_heading_degrees=observation.legal_heading_degrees,
                    deviation_degrees=observation.deviation_degrees,
                    is_contradiction=observation.is_contradiction,
                )
            )


def wrong_way_finalize_strategy(
    scene: SceneConfig,
    *,
    direction_id: str | None = None,
    vehicle_classes: frozenset[ObjectClass] = VEHICLE_CLASSES,
) -> _WrongWayFinalize:
    """Build the wrong-way reasoning back half for one scene (public factory).

    ``vehicle_classes`` defaults to the shared
    :data:`~trafficpulse.rules.vehicles.VEHICLE_CLASSES` and is a parameter rather
    than a scene field for the same reason red-light's is: it is a deployment
    judgement about which road users a jurisdiction adjudicates, not a fact about
    the camera's geometry.

    The exact strategy ``WrongWayPipeline`` injects into the shared
    ``CompositionPipeline`` -- exposed so a multi-rule composition (the P?-H6
    real-time engine) can run this rule alongside others over **one** shared
    detect+track front half instead of duplicating detection per rule. Applies
    the same fail-fast scene resolution as the pipeline constructor.

    Raises:
        SceneConfigurationError: if the single governing legal direction cannot
            be resolved (see :func:`_resolve_legal_direction`).
        ValueError: if the scene declares no usable ``wrong_way`` parameter block.
    """

    params = wrong_way_parameters(scene)
    legal_direction, lane_id, lane_polygon = _resolve_legal_direction(scene, direction_id)
    return _WrongWayFinalize(
        params=params,
        legal_direction=legal_direction,
        lane_id=lane_id,
        lane_polygon=lane_polygon,
        vehicle_classes=vehicle_classes,
        # The strategy owns its capture rather than returning it alongside (as
        # red-light does): this factory has callers that only probe whether the
        # scene resolves, and widening its return type would churn every one of
        # them for a value they do not want. Drivers read ``strategy.capture``.
        capture=WrongWayOverlayCapture(
            lane_id=lane_id, legal_direction=(legal_direction.dx, legal_direction.dy)
        ),
    )


class WrongWayPipeline:
    """Deterministic offline orchestration for the first wrong-way vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the existing P1-U4
    heading derivation and wrong-way reasoner over one ``SceneConfig``. The
    ``detector_config`` configures the shared ``DetectionAdapter`` seam
    (label map + provenance); ``direction_id`` selects which legal direction
    governs the run when the scene declares more than one (the single-lane slice).

    The shared orchestration is delegated to a held
    :class:`~trafficpulse.pipeline.base.CompositionPipeline`; this class contributes
    the wrong-way reasoning strategy and the single-lane resolution/fail-fast.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        direction_id: str | None = None,
    ) -> None:
        strategy = wrong_way_finalize_strategy(scene, direction_id=direction_id)
        self._lane_id = strategy.lane_id
        self._core = CompositionPipeline(
            detector=detector,
            tracker=tracker,
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=strategy,
        )

    @property
    def lane_id(self) -> str:
        """The resolved single-lane id this pipeline reasons over."""

        return self._lane_id

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

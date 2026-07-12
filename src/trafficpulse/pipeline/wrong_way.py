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

from collections.abc import Iterable
from dataclasses import dataclass

from ..contracts import ConfirmedEvent, ModelRef, SceneConfig, TrackState
from ..contracts.scene import DirectionVector, LegalDirection
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.heading import derive_heading_observations_with_taint
from ..rules.engine import RuleEngine
from ..rules.wrong_way import WrongWayParameters, WrongWayReasoner, wrong_way_parameters
from ..tracking.interface import Tracker
from .base import _MEDIA_TIME_EPOCH, CompositionPipeline, frame_record_to_frame
from .errors import SceneConfigurationError

# ``frame_record_to_frame`` and ``_MEDIA_TIME_EPOCH`` are defined in the shared
# base and re-exported here (listed in ``__all__``) so the callers and tests that
# reference them through :mod:`trafficpulse.pipeline.wrong_way` resolve unchanged.
__all__ = ["WrongWayPipeline", "frame_record_to_frame", "_MEDIA_TIME_EPOCH"]


def _resolve_legal_direction(
    scene: SceneConfig, direction_id: str | None
) -> tuple[DirectionVector, str]:
    """Resolve the single governing ``(legal_direction, lane_id)`` for the slice.

    Raises:
        SceneConfigurationError: if the single lane cannot be picked (no legal
            direction; more than one with no ``direction_id``; an unknown
            ``direction_id``; or the chosen direction has no zone/lane id).
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
    return chosen.vector, chosen.zone_ids[0]


@dataclass(frozen=True)
class _WrongWayFinalize:
    """The wrong-way reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved run parameters (deviation threshold, min persistence) and
    the single governing legal direction / lane. Builds a ``WrongWayReasoner`` for
    the run and, per track, derives the P1-U4 heading stream and reasons over it --
    the exact operations the pre-generalization ``finalize`` performed.
    """

    params: WrongWayParameters
    legal_direction: DirectionVector
    lane_id: str

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> WrongWayReasoner:
        return WrongWayReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: WrongWayReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        derivation = derive_heading_observations_with_taint(
            track,
            legal_direction=self.legal_direction,
            lane_id=self.lane_id,
            deviation_max_degrees=self.params.deviation_max_degrees,
        )
        return reasoner.run_derivation(derivation)


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
        params = wrong_way_parameters(scene)
        legal_direction, lane_id = _resolve_legal_direction(scene, direction_id)
        self._lane_id = lane_id
        self._core = CompositionPipeline(
            detector=detector,
            tracker=tracker,
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=_WrongWayFinalize(
                params=params, legal_direction=legal_direction, lane_id=lane_id
            ),
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

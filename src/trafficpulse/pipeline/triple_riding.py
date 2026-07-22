"""Triple-riding vertical-slice orchestration (v1.1 U3).

A thin, deterministic orchestration that runs one stream end to end through
*existing* components -- it wires, it does not compute:

```
FrameRecord (P1-U5 ingestion)
  -> Detector + DetectionAdapter (P1-U6 seam)        -> Detection
  -> Tracker (P1-U8 seam)                            -> TrackState
  -> RiderCountFrameObserver (v1.1 U3, over U1 perception + P4-U4 association)
                                                     -> RiderCountObservation + Association
  -> group by (camera_id, track_id) in timestamp order
  -> TripleRidingReasoner.run                        -> ConfirmedEvent
```

Composition on the shared base (P3-U2)
--------------------------------------
The detect → track → group front half and the deterministic ``finalize`` scaffold
live in the shared :class:`~trafficpulse.pipeline.base.CompositionPipeline`, which
this slice *holds* (composition, not inheritance) -- the same base wrong-way,
illegal-stopping, and no-helmet use. Triple-riding contributes a
:class:`RiderCountFrameObserver` (a per-frame ``FrameObserver`` -- rider counting
is cross-track, so it must see every track in a frame at once, exactly like the
helmet observer) and a reasoning ``FinalizeStrategy`` that reads the observer's
accumulated stream per motorcycle.

Backend independence
--------------------
Depends on the ``Detector`` and ``Tracker`` **abstractions** only -- never on a
backend. Rider counting is pure geometry (no pixels, no classifier), so unlike
no-helmet this slice needs no ``HelmetClassifier``: any detector/tracker drops in
through the constructor unchanged, and no ML framework enters this import graph.

Determinism
-----------
No wall-clock, no randomness. The observer sees frames in stream order; track
groups are iterated in ``(camera_id, track_id)`` order; the reasoner processes
steps in ``(timestamp, observation_id)`` order; emitted events are sorted by
``(trigger_at, event_id)``. ``reset`` returns the orchestration -- observer
included -- to a replayable initial state.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..association import RiderAssociationConfig
from ..contracts import (
    Association,
    ConfirmedEvent,
    ModelRef,
    RiderCountObservation,
    SceneConfig,
    TrackState,
)
from ..contracts.enums import ObjectClass
from ..detector.config import DetectorConfig
from ..detector.frame import Frame
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.rider_count import RiderCountDerivation, derive_rider_count_observations
from ..rules.engine import RuleEngine
from ..rules.triple_riding import (
    TripleRidingParameters,
    TripleRidingReasoner,
    triple_riding_parameters,
)
from ..tracking.interface import Tracker
from .base import CompositionPipeline


class RiderCountFrameObserver:
    """Accumulates ``RiderCountObservation``s from tracked states (a ``FrameObserver``).

    Per frame it derives one count observation per tracked motorcycle (reusing the
    v1.1 U1 perception + P4-U4 association through
    :func:`~trafficpulse.observations.rider_count.derive_rider_count_observations`)
    and accumulates them plus the rider↔motorcycle associations. It reads no pixels;
    the ``frame`` argument is ignored. The accumulated stream is exposed via
    :meth:`derivation`; nothing is persisted and no event is produced.
    """

    def __init__(self, *, association_config: RiderAssociationConfig | None = None) -> None:
        self._association_config = association_config
        self._observations: list[RiderCountObservation] = []
        self._associations: list[Association] = []
        # Motorcycles seen tainted since their last emitted observation; the next
        # clean observation for such a motorcycle is a taint restart.
        self._tainted_since_emit: set[str] = set()
        self._restart_ids: set[str] = set()

    # --- FrameObserver protocol ---------------------------------------------
    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        """Derive this frame's rider-count observations; accumulate them internally."""

        for state in states:
            if state.tainted and state.object_class is ObjectClass.MOTORCYCLE:
                self._tainted_since_emit.add(state.track_id)

        derivation = derive_rider_count_observations(
            states, association_config=self._association_config
        )
        self._associations.extend(derivation.associations)
        for observation in derivation.observations:
            self._emit(observation)

    def reset(self) -> None:
        """Return the observer to its initial (pre-stream) state for replay."""

        self._observations = []
        self._associations = []
        self._tainted_since_emit = set()
        self._restart_ids = set()

    # --- accumulated output --------------------------------------------------
    def _emit(self, observation: RiderCountObservation) -> None:
        if observation.track_id in self._tainted_since_emit:
            self._restart_ids.add(observation.observation_id)
            self._tainted_since_emit.discard(observation.track_id)
        self._observations.append(observation)

    def derivation(self) -> RiderCountDerivation:
        """The accumulated stream (observations + associations + taint restarts).

        Observations and associations are sorted deterministically, so the stream
        is a pure function of the frames seen, not of emission order.
        """

        return RiderCountDerivation(
            observations=tuple(
                sorted(self._observations, key=lambda o: (o.timestamp, o.observation_id))
            ),
            associations=tuple(
                sorted(self._associations, key=lambda a: (a.timestamp, a.association_id))
            ),
            taint_restart_ids=frozenset(self._restart_ids),
        )


@dataclass(frozen=True)
class _TripleRidingFinalize:
    """The triple-riding reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved run parameters and the observer whose accumulated stream it
    reasons over. Builds a ``TripleRidingReasoner`` for the run and, per motorcycle
    track, selects that motorcycle's count observations + associations and reasons.
    """

    params: TripleRidingParameters
    observer: RiderCountFrameObserver

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> TripleRidingReasoner:
        return TripleRidingReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: TripleRidingReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        if not track:
            return ()
        motorcycle_track_id = track[0].track_id
        derivation = self.observer.derivation()
        observations = tuple(
            o for o in derivation.observations if o.track_id == motorcycle_track_id
        )
        if not observations:
            return ()  # not a motorcycle (a rider/vehicle group), or nothing observed
        associations = tuple(
            a for a in derivation.associations if a.object_track_id == motorcycle_track_id
        )
        return reasoner.run(
            observations,
            associations=associations,
            taint_restart_ids=derivation.taint_restart_ids,
        )


def triple_riding_finalize_strategy(
    scene: SceneConfig,
    *,
    association_config: RiderAssociationConfig | None = None,
) -> tuple[_TripleRidingFinalize, RiderCountFrameObserver]:
    """Build the triple-riding back half for one scene (public factory).

    Returns the reasoning strategy **and** the frame observer it reads, because a
    caller composing this rule onto a shared ``CompositionPipeline`` (the real-time
    engine) must register the observer as the pipeline's frame observer -- the
    strategy alone never sees the per-frame track sets.

    Raises:
        ValueError: if the scene declares no usable ``triple_riding`` parameter
            block (fail-fast, mirroring the sibling slices).
    """

    params = triple_riding_parameters(scene)
    observer = RiderCountFrameObserver(association_config=association_config)
    return _TripleRidingFinalize(params=params, observer=observer), observer


class TripleRidingPipeline:
    """Deterministic offline orchestration for the triple-riding vertical slice.

    Composes an injected ``Detector`` and ``Tracker`` with the U3 rider-count
    observation derivation and reasoner over one ``SceneConfig``. The triple-riding
    parameters are resolved once at construction (fail-fast on a misconfigured
    scene). The shared orchestration is delegated to a held ``CompositionPipeline``.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        association_config: RiderAssociationConfig | None = None,
    ) -> None:
        strategy, observer = triple_riding_finalize_strategy(
            scene, association_config=association_config
        )
        self._observer = observer
        self._core = CompositionPipeline(
            detector=detector,
            tracker=tracker,
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=strategy,
            frame_observer=observer,
        )

    @property
    def observer(self) -> RiderCountFrameObserver:
        """The accumulated rider-count stream (for inspection/diagnostics)."""

        return self._observer

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state (delegated)."""

        self._core.reset()

    def process_frame(self, frame_record: FrameRecord) -> tuple[TrackState, ...]:
        """Detect + track + observe one frame, and return its states (delegated)."""

        return self._core.process_frame(frame_record)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Reason over the accumulated observation stream; return events (delegated)."""

        return self._core.finalize()

    def process(self, frames: Iterable[FrameRecord]) -> tuple[ConfirmedEvent, ...]:
        """Run one complete offline stream: ``reset`` → stream frames → ``finalize``."""

        return self._core.process(frames)

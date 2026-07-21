"""No-helmet vertical-slice orchestration (P4-U5).

A thin, deterministic **offline** orchestration that runs one recorded stream end
to end through *existing* components -- it wires, it does not compute:

```
FrameRecord (P1-U5 ingestion)
  -> detector Frame (identity + pixels)
  -> Detector + DetectionAdapter (P1-U6 seam)          -> Detection
  -> Tracker (P1-U8 seam)                              -> TrackState
  -> HelmetFrameObserver (P4-U4, the P4-U2 pixel hook) -> HelmetStateObservation
                                                        + Association
  -> group by (camera_id, track_id) in timestamp order
  -> NoHelmetReasoner.run (P4-U5)                      -> ConfirmedEvent
```

Composition on the shared base (P3-U2)
--------------------------------------
The detect -> track -> group -> provenance-collect front half and the deterministic
``finalize`` scaffold live in the shared
:class:`~trafficpulse.pipeline.base.CompositionPipeline`, which this pipeline
*holds* (composition, not inheritance) -- the same base wrong-way and
illegal-stopping use. No-helmet contributes two injected pieces:

* a :class:`~trafficpulse.pipeline.base.FrameObserver` -- the P4-U4
  ``HelmetFrameObserver`` -- because this is the project's first slice whose
  observations need **pixels**, which ``finalize`` never sees; and
* a ``FinalizeStrategy`` reasoning back half that reads the observer's accumulated
  stream and reasons over it per rider.

Per the Phase 2 decision (E.8/E.9) this stays a **thin sibling** configuration,
not a generalised multi-rule runner.

Why the strategy filters by track
---------------------------------
``FinalizeStrategy.events_for_track`` is invoked once per ``(camera_id,
track_id)`` group. No-helmet episodes are keyed by the **rider**, so each call
selects that track's helmet observations from the observer's stream. Motorcycle
and vehicle groups simply have none and yield nothing -- no special-casing needed.

Backend independence
--------------------
Depends on the ``Detector``, ``Tracker``, and ``HelmetClassifier`` **abstractions**
only -- never on a backend (``RTDetrDetector``, ``ZeroShotHelmetClassifier``, torch,
transformers). Any implementation of the three seams drops in through the
constructor unchanged, so the stub classifier and a real one are interchangeable
and no ML framework enters this module's import graph.

Scene configuration (fail-fast)
-------------------------------
The governing no-helmet parameters are loaded once at construction via
``no_helmet_parameters``; a scene declaring no ``no_helmet`` block or no
``min_persistence`` fails fast at construction -- mirroring the sibling pipelines --
so a misconfigured scene never silently produces zero events.

Determinism
-----------
No wall-clock, no randomness. The observer sees frames in stream order; track
groups are iterated in ``(camera_id, track_id)`` order; the reasoner processes
steps in ``(timestamp, observation_id)`` order; emitted events are sorted by
``(trigger_at, event_id)``. ``finalize`` builds a fresh reasoner each call and
``reset`` returns the orchestration -- observer included -- to a replayable initial
state.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ..classifier.interface import HelmetClassifier
from ..contracts import ConfirmedEvent, ModelRef, SceneConfig, TrackState
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..observations.helmet import HelmetObservationConfig
from ..rules.engine import RuleEngine
from ..rules.no_helmet import NoHelmetParameters, NoHelmetReasoner, no_helmet_parameters
from ..tracking.interface import Tracker
from .base import CompositionPipeline
from .helmet_observer import HelmetFrameObserver


@dataclass(frozen=True)
class _NoHelmetFinalize:
    """The no-helmet reasoning back half injected into ``CompositionPipeline``.

    Holds the resolved run parameters and the observer whose accumulated stream it
    reasons over. Builds a ``NoHelmetReasoner`` for the run and, per track, selects
    that rider's helmet observations + associations and reasons over them.
    """

    params: NoHelmetParameters
    observer: HelmetFrameObserver

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> NoHelmetReasoner:
        return NoHelmetReasoner(
            RuleEngine(), self.params, scene_config_hash=scene_config_hash, models=models
        )

    def events_for_track(
        self, reasoner: NoHelmetReasoner, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        if not track:
            return ()
        track_id = track[0].track_id
        derivation = self.observer.derivation()
        observations = tuple(o for o in derivation.observations if o.track_id == track_id)
        if not observations:
            return ()  # not a rider (a motorcycle/vehicle group), or nothing observed
        associations = tuple(
            a for a in self.observer.associations() if a.subject_track_id == track_id
        )
        return reasoner.run(
            observations,
            associations=associations,
            taint_restart_ids=derivation.taint_restart_ids,
        )


def no_helmet_finalize_strategy(
    scene: SceneConfig,
    *,
    classifier: HelmetClassifier,
    observation_config: HelmetObservationConfig | None = None,
) -> tuple[_NoHelmetFinalize, HelmetFrameObserver]:
    """Build the no-helmet back half for one scene (public factory).

    Returns the reasoning strategy **and** the pixel observer it reads, because a
    caller composing this rule onto a shared ``CompositionPipeline`` (the
    real-time engine) must register the observer as the pipeline's frame observer
    -- the strategy alone never sees pixels. Applies the same fail-fast scene
    resolution as the pipeline constructor.

    Raises:
        ValueError: if the scene declares no usable ``no_helmet`` parameter block.
    """

    params = no_helmet_parameters(scene)
    observer = HelmetFrameObserver(classifier=classifier, config=observation_config)
    return _NoHelmetFinalize(params=params, observer=observer), observer


class NoHelmetPipeline:
    """Deterministic offline orchestration for the no-helmet vertical slice.

    Composes an injected ``Detector``, ``Tracker``, and ``HelmetClassifier`` with
    the P4-U4 observation derivation and the P4-U5 reasoner over one
    ``SceneConfig``. The no-helmet rule parameters are resolved once at
    construction (fail-fast on a misconfigured scene).

    The shared orchestration is delegated to a held ``CompositionPipeline``; this
    class contributes the pixel observer and the reasoning strategy.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        classifier: HelmetClassifier,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        observation_config: HelmetObservationConfig | None = None,
    ) -> None:
        strategy, observer = no_helmet_finalize_strategy(
            scene, classifier=classifier, observation_config=observation_config
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
    def observer(self) -> HelmetFrameObserver:
        """The accumulated perception stream (observations + associations).

        Exposed for inspection/diagnostics: a run that confirms nothing can be
        distinguished from a run that observed nothing.
        """

        return self._observer

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state (delegated).

        Resets the observer too, so the accumulated observation stream does not
        leak across runs.
        """

        self._core.reset()

    def process_frame(self, frame_record: FrameRecord) -> tuple[TrackState, ...]:
        """Detect + track + observe one frame, and return its states (delegated)."""

        return self._core.process_frame(frame_record)

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Reason over the accumulated observation stream; return events (delegated)."""

        return self._core.finalize()

    def process(self, frames: Iterable[FrameRecord]) -> tuple[ConfirmedEvent, ...]:
        """Run one complete offline stream: ``reset`` -> stream frames -> ``finalize``."""

        return self._core.process(frames)

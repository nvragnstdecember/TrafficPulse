"""Generalized composition-pipeline base (P3-U2, composition).

``CompositionPipeline`` is the shared detect -> track -> group-by-``(camera,
track)`` -> provenance-collect orchestration that the two shipped vertical-slice
pipelines -- ``WrongWayPipeline`` (P1-U10) and ``IllegalStoppingPipeline``
(P2-U5) -- were byte-for-byte duplicating. It wires existing components across
their frozen contract seams; it computes nothing itself.

Composition, not inheritance
----------------------------
This is a **collaborator the violation pipelines hold and delegate to**, not a
superclass they extend. It carries **no** violation-specific knowledge: the
reasoning back half (which reasoner to build, and how to derive + reason over one
track's states) is supplied by an injected :class:`FinalizeStrategy`. A new
violation slice is a *configuration* -- a small strategy plus a thin pipeline that
holds this core -- not a subclass override cascade.

```
FrameRecord (P1-U5 ingestion)
  -> detector Frame (frame_record_to_frame; fixed media-time anchor)
  -> Detector + DetectionAdapter (P1-U6 seam)          -> Detection
  -> Tracker (P1-U8 seam)                               -> TrackState
  -> FrameObserver.observe (P4-U2; optional, pixels)    -> (accumulated externally)
  -> group by (camera_id, track_id) in timestamp order
  -> FinalizeStrategy.build_reasoner + events_for_track -> ConfirmedEvent
```

Two injected extension points, deliberately different
-----------------------------------------------------
:class:`FinalizeStrategy` is the *reasoning* back half and sees only
``TrackState``s -- geometry, no pixels -- which is what keeps reasoning replayable
without a model. :class:`FrameObserver` is an *optional perception* side-channel
that sees the decoded image while it still exists (finalize cannot: the pipeline
retains no frames). It is ``None`` for every pre-Phase-4 slice, whose observations
are pure geometry.

The one conversion this layer owns: FrameRecord -> detector Frame
-----------------------------------------------------------------
Ingestion emits ``FrameRecord.timestamp_seconds`` as a **media-relative** float
(PTS seconds), while the detector ``Frame`` / frozen ``Detection`` require a
timezone-aware ``datetime``. :func:`frame_record_to_frame` bridges that at a
**fixed UTC epoch anchor** (``1970-01-01T00:00:00Z + timedelta(seconds=pts)``): it
preserves media-time *semantics* exactly -- the inter-frame delta equals the PTS
delta, the only temporal quantity the reasoners' duration thresholds consume --
while using **no wall-clock and no nominal FPS**. The anchor is a deterministic
media-time marker, not a fabricated absolute capture date; mapping to true capture
time remains a later concern requiring external metadata (ingestion docstring /
architecture-review §17, §239). ``camera_id`` resolves to the frame's own id when
present (real ingestion stamps the scene camera onto every frame) else the scene
camera; ``frame_index`` and the opaque ``image`` carry through unchanged (the
image is never copied).

Empty-detection-frame semantics (preserved, not hidden)
-------------------------------------------------------
When the detector returns zero detections for a frame, the ``Tracker`` seam
receives an empty ``Sequence[Detection]``. Per the documented P1-U8/U9 behaviour
that empty batch is **inert**: it returns ``()``, does **not** age tracks, and
does **not** advance the frame-progress guard. The orchestrator honours that
faithfully -- it fabricates no detection and no frame metadata to advance time.

Backend independence
--------------------
Depends only on the ``Detector`` / ``Tracker`` abstractions, the frozen contracts,
the provenance helper, and the injected strategy -- never on a detector/tracker
*backend* (``RTDetrDetector``, ``IouTracker``, torch, transformers, ...). Any
implementation of the two seams drops in through the constructor unchanged, so
``import trafficpulse.pipeline`` stays backend-free.

Determinism
-----------
No wall-clock, no randomness. Track groups are iterated in ``(camera_id,
track_id)`` order, each track's states in ``(timestamp, frame_index)`` order, and
the emitted events sorted by ``(trigger_at, event_id)`` -- so the result is a pure
function of the injected components, the frame stream, and the scene, independent
of transient ordering. :meth:`finalize` builds a fresh reasoner (via the strategy)
each call, so it is idempotent over the accumulated history; :meth:`reset` returns
the orchestration to a replayable initial state.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypeVar

from ..contracts import (
    ConfirmedEvent,
    Detection,
    ModelRef,
    SceneConfig,
    TrackState,
    scene_config_hash,
)
from ..detector.adapter import DetectionAdapter
from ..detector.config import DetectorConfig
from ..detector.frame import Frame
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..tracking.interface import Tracker
from .provenance import normalize_model_refs

# Fixed media-time anchor: media-relative PTS seconds are added to this epoch to
# form the timezone-aware datetime the detector/contract seams require. It is a
# deterministic marker (no wall-clock), and only inter-frame deltas are load-
# bearing downstream -- see the module docstring.
_MEDIA_TIME_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def frame_record_to_frame(frame_record: FrameRecord, *, camera_id: str) -> Frame:
    """Convert an ingestion ``FrameRecord`` into a detector ``Frame``.

    ``camera_id`` is the resolved non-empty camera id to stamp (the frame's own id
    when present, else the scene camera). The media-relative PTS timestamp is
    anchored at a fixed UTC epoch (see module docstring); ``frame_index`` and the
    opaque ``image`` carry through unchanged (the image is not copied).
    """

    return Frame(
        camera_id=camera_id,
        frame_index=frame_record.frame_index,
        timestamp=_MEDIA_TIME_EPOCH + timedelta(seconds=frame_record.timestamp_seconds),
        image=frame_record.image,
    )


ReasonerT = TypeVar("ReasonerT")


class FrameObserver(Protocol):
    """An optional per-frame side-channel with access to **pixels** (P4-U2).

    Why this hook exists
    --------------------
    Every observation derivation shipped before Phase 4 -- heading (P1-U4), in-zone
    (P2-U2), stationary (P2-U3), crossing (P3-U4) -- is pure geometry over
    ``TrackState``, so :meth:`CompositionPipeline.finalize` needs only boxes and
    can reason long after the pixels are gone. Helmet state is the project's first
    observation that requires the **image**, and by finalize time the image no
    longer exists: :meth:`CompositionPipeline.process_frame` deliberately keeps
    only ``TrackState``s (retaining decoded frames for a whole clip would be
    unbounded memory). This hook is the only place where pixels and tracked
    identity coexist.

    What an implementation may and may not do
    -----------------------------------------
    An observer *derives and accumulates*; it does not decide. It may read the
    frame's pixels and that frame's states and buffer whatever it produces (P4-U4
    buffers ``HelmetStateObservation``s). It must **not** mutate the frame, the
    states, or any pipeline state, and it must never confirm, persist, or suppress
    anything -- the reasoning layer remains the sole authority on violations
    (architecture-review §14/§15: models produce observations, rules decide).

    Determinism and replay
    ----------------------
    :meth:`observe` is called exactly once per :meth:`process_frame`, after
    tracking, in frame-stream order -- including for frames whose states are empty,
    so an implementation sees the true frame sequence and never has to infer a gap.
    Implementations must be deterministic (no wall-clock, no randomness) and must
    implement :meth:`reset` so a replayed stream reproduces an identical result;
    :meth:`CompositionPipeline.reset` resets the observer alongside the tracker.

    Optional by construction
    ------------------------
    ``frame_observer`` defaults to ``None``. When it is ``None`` this class behaves
    exactly as it did before P4-U2 -- no call, no branch cost, no ordering change --
    so the shipped wrong-way and illegal-stopping slices are byte-identical
    (asserted by ``tests/pipeline/test_frame_observer.py``).
    """

    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        """Inspect one frame's pixels + tracked states; accumulate internally.

        Called once per processed frame, after tracking, in stream order. Must not
        mutate its arguments or any pipeline state.
        """
        ...

    def reset(self) -> None:
        """Return the observer to its initial (pre-stream) state for replay."""
        ...


class FinalizeStrategy(Protocol[ReasonerT]):
    """The injected reasoning back half of :meth:`CompositionPipeline.finalize`.

    Encapsulates the only per-violation variation: which reasoner to build for a
    run, and how to derive observations + reason over one track's states. The base
    owns everything deterministic around it (provenance normalization, group
    iteration, per-track ordering, output ordering); the strategy owns nothing
    stateful across frames.
    """

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> ReasonerT:
        """Build the fresh reasoner for one :meth:`CompositionPipeline.finalize`.

        Called once per finalize; the returned reasoner is reused across every
        track. ``models`` is the run-level provenance the base already normalized.
        """
        ...

    def events_for_track(
        self, reasoner: ReasonerT, track: list[TrackState]
    ) -> Iterable[ConfirmedEvent]:
        """Derive observations for one already-sorted ``track`` and reason over them.

        ``track`` is a single ``(camera_id, track_id)`` group's states in
        ``(timestamp, frame_index)`` order. Returns the events confirmed for that
        track (the base sorts the union deterministically).
        """
        ...


class CompositionPipeline:
    """Deterministic offline detect -> track -> group -> reason orchestration.

    Holds an injected ``Detector`` and ``Tracker`` and a :class:`FinalizeStrategy`
    that supplies the reasoning back half. The ``detector_config`` configures the
    shared ``DetectionAdapter`` seam (label map + provenance). An optional
    :class:`FrameObserver` may be injected for pixel-dependent derivation; when it
    is ``None`` (the default) behaviour is exactly as before P4-U2. See the module
    docstring for the full contract.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        scene: SceneConfig,
        detector_config: DetectorConfig,
        finalize_strategy: FinalizeStrategy[Any],
        frame_observer: FrameObserver | None = None,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._scene = scene
        self._adapter = DetectionAdapter(detector_config)
        self._finalize_strategy = finalize_strategy
        # Optional pixel-dependent side-channel (P4-U2). ``None`` for every slice
        # whose observations are pure geometry over TrackState -- i.e. all of them
        # before Phase 4 -- so those pipelines are unchanged.
        self._frame_observer = frame_observer
        self._scene_hash = scene_config_hash(scene)
        self._history: dict[tuple[str, str], list[TrackState]] = {}
        # Run-level model provenance accumulated across frames (P2-U1): the
        # distinct truthful ``ModelRef``s the detector/tracker adapters stamp onto
        # ``Detection.source_model`` / ``TrackState.tracker``. Collected here at the
        # composition boundary (the only place that sees both), de-duplicated and
        # ordered in :meth:`finalize`, and stamped onto every minted event. Never
        # read by any reasoning predicate.
        self._model_refs: list[ModelRef] = []

    def reset(self) -> None:
        """Return the orchestration to a replayable initial state.

        Resets the injected (stateful) ``Tracker`` and the injected
        :class:`FrameObserver` (if any), and clears the accumulated per-track
        history + provenance. The detector needs no reset -- it holds no temporal
        state across frames -- and the reasoner/engine are ephemeral (rebuilt per
        :meth:`finalize`), so the same frame stream replays to an equal result.
        """

        self._tracker.reset()
        if self._frame_observer is not None:
            self._frame_observer.reset()
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
        # Pixel-dependent side-channel (P4-U2): the only point where the decoded
        # image and this frame's tracked identities coexist (finalize sees neither).
        # Called on every processed frame -- including zero-state frames, so an
        # observer sees the true frame sequence -- and it derives/accumulates only:
        # it decides nothing and mutates nothing here.
        if self._frame_observer is not None:
            self._frame_observer.observe(frame, states)
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

        Builds one reasoner via the injected strategy (with the run-level ``models``
        provenance -- the de-duplicated, sorted union of the ``ModelRef``s collected
        during :meth:`process_frame`), then for each ``(camera_id, track_id)`` group
        in sorted order feeds the track's states (sorted by ``(timestamp,
        frame_index)``) to the strategy and collects the confirmed events, sorted by
        ``(trigger_at, event_id)``. Idempotent: a pure function of the accumulated
        history + provenance (the reasoner is rebuilt here, not held across frames).
        """

        reasoner = self._finalize_strategy.build_reasoner(
            scene_config_hash=self._scene_hash,
            models=normalize_model_refs(self._model_refs),
        )
        events: list[ConfirmedEvent] = []
        for key in sorted(self._history):
            track = sorted(self._history[key], key=lambda s: (s.timestamp, s.frame_index or 0))
            events.extend(self._finalize_strategy.events_for_track(reasoner, track))
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

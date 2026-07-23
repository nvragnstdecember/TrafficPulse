"""The deterministic real-time inference engine (H6).

One engine runs the full flow the architecture fixes -- video -> RT-DETR
inference -> IoU tracking -> rule reasoning -> hypothesis lifecycle ->
``ConfirmedEvent`` -> ``EvidenceManifest`` -> persistence -- by **composing the
shipped implementations across their frozen seams**:

```
FrameSource (engine seam; file = P1-U5 ingestion)
  -> FrameScheduler (stride / target-FPS / bounded back-pressure)
  -> DetectorRunner (timed, batch-capable) over the injected Detector (P1-U6)
  -> InstrumentedTracker over the injected Tracker (P1-U8; IoU backend P1-U9)
  -> CompositionPipeline (P3-U2: adapt, track, observe, history, provenance)
  -> MultiRuleFinalize (P1-U3 RuleEngine lifecycle inside each shipped reasoner)
  -> ConfirmedEvent -> build_engine_manifest (real frame references)
  -> EventStore.persist_pairs (P1-U11 write-once JSON)
```

The engine adds **no** reasoning: every hypothesis, threshold, and confirmation
decision is the existing rules', reached through their public factories. What
it adds is the real-time envelope -- scheduling, batching, instrumentation,
structured logging, evidence frame picking -- as thin decorators around the
proven offline core, so a one-rule engine is event-identical to the
corresponding standalone pipeline (asserted by tests).

Live streams and incremental confirmation
-----------------------------------------
``run(source)`` is the self-contained whole-stream form. A live producer
instead drives ``submit`` / ``drain`` itself and may call :meth:`checkpoint`
periodically: ``finalize`` is idempotent over the accumulated history (it
rebuilds fresh reasoners each call -- the P3-U2 guarantee), every event field
is a pure function of the history **prefix up to its trigger**, and the P1-U11
store is write-once with idempotent identical replay -- so a later checkpoint
re-persists earlier events as byte-identical no-ops and appends only the new
ones. Frame *pixels* are never retained; per-frame memory cost is one
:class:`FrameStamp` (identity + PTS) for evidence picking, plus the
``TrackState`` history the offline pipelines already keep.

Determinism
-----------
No wall-clock and no randomness anywhere in the decision path: scheduling is
media-time (PTS) driven, log timestamps and wall-latency metrics exist only
under injected clock/probes, and event/manifest identity is content-derived.
An identical source + configuration + injected seams replays to byte-identical
events, manifests, logs (without a clock), and persisted files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..classifier.interface import HelmetClassifier
from ..contracts import ConfirmedEvent, EvidenceManifest, SceneConfig
from ..detector.config import DetectorConfig
from ..detector.interface import Detector
from ..ingestion.video import FrameRecord
from ..persistence import EventStore
from ..persistence.store import StoredEvent
from ..pipeline.base import CompositionPipeline, FrameObserver, frame_record_to_frame
from ..tracking.config import TrackerConfig
from ..tracking.interface import Tracker
from ..tracking.iou_tracker import IouTracker
from .config import EngineConfig
from .errors import RunCancelledError
from .evidence import FrameStamp, build_engine_manifest, media_seconds
from .logs import EngineLogEvent, EngineLogEventKind, EngineLogSink, EventData, NullLogSink
from .metrics import EngineMetrics, LatencyKind, MetricsRecorder
from .rules import CompositeFrameObserver, MultiRuleFinalize, build_rules
from .runner import DetectorRunner, InstrumentedTracker, build_detector, detector_adapter_config
from .scheduler import FrameScheduler, ScheduleDecision
from .sources import FrameSource


@dataclass(frozen=True)
class EngineRunResult:
    """The immutable outcome of one engine run.

    ``manifests[i]`` is the evidence manifest of ``events[i]`` (the pairing the
    store persists). ``metrics`` is the run's final snapshot.
    """

    source_id: str
    events: tuple[ConfirmedEvent, ...]
    manifests: tuple[EvidenceManifest, ...]
    metrics: EngineMetrics


class InferenceEngine:
    """Composes the shipped seams into one real-time run (see module docstring).

    The constructor takes **injected** ``Detector`` / ``Tracker`` /
    ``HelmetClassifier`` seams -- stubs in tests, real backends from
    :func:`build_engine` -- so this class never names a backend and importing it
    pulls in no ML framework. ``clock`` timestamps log events; ``perf`` /
    ``memory_probe`` / ``gpu_probe`` feed the environmental metrics; all four
    default to ``None`` = absent = nothing fabricated.
    """

    def __init__(
        self,
        *,
        scene: SceneConfig,
        detector: Detector,
        tracker: Tracker,
        detector_config: DetectorConfig,
        config: EngineConfig,
        classifier: HelmetClassifier | None = None,
        sink: EngineLogSink | None = None,
        clock: Callable[[], datetime] | None = None,
        perf: Callable[[], float] | None = None,
        memory_probe: Callable[[], int] | None = None,
        gpu_probe: Callable[[], int] | None = None,
        capture_overlay: bool = False,
    ) -> None:
        self._scene = scene
        self._config = config
        self._sink: EngineLogSink = sink if sink is not None else NullLogSink()
        self._clock = clock
        self._sequence = 0
        self._recorder = MetricsRecorder(
            perf=perf, memory_probe=memory_probe, gpu_probe=gpu_probe
        )

        rules = build_rules(
            config.rules, scene=scene, classifier=classifier, capture_overlay=capture_overlay
        )
        observers = tuple(rule.observer for rule in rules if rule.observer is not None)
        # Held so the composition root can retrieve per-rule pixel observers after a
        # run (e.g. the overlay framework reads the no-helmet observer's captured
        # metadata to redraw inference -- see frame_observers()).
        self._observers = observers
        self._runner = DetectorRunner(detector, self._recorder)
        self._core = CompositionPipeline(
            detector=self._runner,
            tracker=InstrumentedTracker(tracker, self._recorder),
            scene=scene,
            detector_config=detector_config,
            finalize_strategy=MultiRuleFinalize(rules),
            frame_observer=CompositeFrameObserver(observers) if observers else None,
        )
        self._scheduler = FrameScheduler(config.scheduler)
        self._stamps: list[FrameStamp] = []

    # --- read-only surface --------------------------------------------------------
    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def metrics(self) -> EngineMetrics:
        """The current metrics snapshot (cheap; safe to read mid-stream)."""

        return self._recorder.snapshot()

    def frame_observers(self) -> tuple[FrameObserver, ...]:
        """The per-rule pixel observers (P4-U2), for post-run inspection.

        The composition root uses this to reach an observer's accumulated
        capture -- e.g. the overlay framework reads the no-helmet observer's
        overlay metadata to redraw inference onto the source frames without
        re-running any model. Empty when no rule has a pixel observer.
        """

        return self._observers

    # --- lifecycle ------------------------------------------------------------------
    def reset(self) -> None:
        """Return every composed component to a replayable initial state."""

        self._core.reset()
        self._scheduler.reset()
        self._runner.clear()
        self._recorder.reset()
        self._stamps.clear()
        self._emit(EngineLogEventKind.ENGINE_RESET, {})

    def submit(self, record: FrameRecord) -> ScheduleDecision:
        """Offer one frame to the scheduler; return (and count) its fate."""

        self._recorder.frames_read += 1
        decision = self._scheduler.submit(record)
        if decision is ScheduleDecision.SKIPPED_STRIDE:
            self._recorder.frames_skipped_stride += 1
        elif decision is ScheduleDecision.SKIPPED_FPS:
            self._recorder.frames_skipped_fps += 1
        elif decision is ScheduleDecision.DROPPED_QUEUE_FULL:
            self._recorder.frames_dropped_backpressure += 1
            self._emit(
                EngineLogEventKind.FRAME_DROPPED,
                {"frame_id": record.frame_id, "frame_index": record.frame_index},
            )
        else:
            self._recorder.frames_admitted += 1
        self._recorder.observe_queue_depth(self._scheduler.queue_depth)
        return decision

    def drain(self) -> int:
        """Process every queued frame, in admitted batches; return the count.

        Each batch is offered to the runner's batch prefetch (one inference
        call when the backend is batch-capable), then fed frame-by-frame
        through the shared pipeline core -- detection adapt, tracking, pixel
        observers, history/provenance accumulation -- recording one
        :class:`FrameStamp` per processed frame for evidence picking.
        """

        processed = 0
        while True:
            batch = self._scheduler.take(self._config.batch_size)
            if not batch:
                break
            perf = self._recorder.perf
            self._runner.prefetch(
                [
                    frame_record_to_frame(record, camera_id=self._camera_id(record))
                    for record in batch
                ]
            )
            for record in batch:
                started = perf() if perf is not None else None
                self._core.process_frame(record)
                if perf is not None and started is not None:
                    self._recorder.observe_latency(LatencyKind.FRAME, perf() - started)
                self._recorder.frames_processed += 1
                self._recorder.observe_media_timestamp(record.timestamp_seconds)
                self._stamps.append(
                    FrameStamp(
                        camera_id=self._camera_id(record),
                        frame_id=record.frame_id,
                        frame_index=record.frame_index,
                        timestamp_seconds=record.timestamp_seconds,
                    )
                )
            self._recorder.batches_processed += 1
            self._recorder.sample_resources()
            self._emit(EngineLogEventKind.BATCH_PROCESSED, {"frames": len(batch)})
            processed += len(batch)
        return processed

    def finalize(self) -> tuple[ConfirmedEvent, ...]:
        """Reason over the accumulated history; return the confirmed events.

        Idempotent over the history (fresh reasoners each call), so a live
        caller may finalize periodically -- see the module docstring.
        """

        events = self._core.finalize()
        self._recorder.events_confirmed = len(events)
        self._emit(EngineLogEventKind.FINALIZED, {"events": len(events)})
        return events

    def manifests_for(
        self, events: Iterable[ConfirmedEvent]
    ) -> tuple[EvidenceManifest, ...]:
        """Build each event's manifest from the actually-processed frame record."""

        return tuple(
            build_engine_manifest(event, self._stamps, config=self._config.evidence)
            for event in events
        )

    def run(
        self,
        source: FrameSource,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> EngineRunResult:
        """One complete stream: reset -> schedule+process -> finalize -> evidence.

        Self-contained (resets first), so repeated calls with a replayable
        source produce equal results.

        ``should_cancel`` is an optional cooperative-cancellation predicate,
        checked once between source frames. When it becomes true the run stops
        immediately by raising :class:`RunCancelledError` -- *before* finalizing
        or building evidence -- so a cancelled run leaves no partial result and
        never touches persistence. Omitting it preserves the original behaviour
        exactly (no per-frame overhead, no new failure mode).
        """

        self.reset()
        self._emit(EngineLogEventKind.ENGINE_START, {"source_id": source.source_id})
        self._recorder.run_started()
        self._emit(EngineLogEventKind.SOURCE_OPENED, {"source_id": source.source_id})
        for record in source.frames():
            if should_cancel is not None and should_cancel():
                raise RunCancelledError(
                    f"run cancelled after {self._recorder.frames_processed} frame(s)"
                )
            self.submit(record)
            self.drain()
        events = self.finalize()
        manifests = self.manifests_for(events)
        self._recorder.run_ended()
        metrics = self._recorder.snapshot()
        self._emit(
            EngineLogEventKind.ENGINE_STOP,
            {"frames_processed": metrics.frames_processed, "events": len(events)},
        )
        return EngineRunResult(
            source_id=source.source_id,
            events=events,
            manifests=manifests,
            metrics=metrics,
        )

    # --- persistence -----------------------------------------------------------------
    def persist(
        self, result: EngineRunResult, *, store: EventStore, run_id: str
    ) -> tuple[StoredEvent, ...]:
        """Persist a result's event+manifest pairs (P1-U11 write-once semantics)."""

        stored = store.persist_pairs(
            run_id, zip(result.events, result.manifests, strict=True)
        )
        self._emit(
            EngineLogEventKind.PERSISTED, {"run_id": run_id, "events": len(stored)}
        )
        return stored

    def checkpoint(
        self, *, store: EventStore, run_id: str, final: bool = False
    ) -> tuple[StoredEvent, ...]:
        """Finalize now and persist incrementally (live-stream checkpointing).

        Safe to call repeatedly mid-stream: already-persisted events re-persist
        as byte-identical idempotent no-ops (see the module docstring), newly
        confirmed ones append.

        Evidence-window deferral (what makes repeat checkpoints conflict-free):
        an *event* is a pure function of the history prefix up to its trigger,
        but its manifest's **after-frame** keeps forming until
        ``after_seconds`` of media time have passed the trigger -- persisting it
        earlier would freeze a reference a later checkpoint could not
        reproduce, which the write-once store rightly refuses. A non-``final``
        checkpoint therefore defers events whose after-window is still open;
        pass ``final=True`` at end of stream to persist everything, clamping
        the after-frame to the last processed frame (truthful: the stream
        genuinely ended inside the margin).
        """

        events = self.finalize()
        if not final:
            horizon = (
                self._stamps[-1].timestamp_seconds if self._stamps else float("-inf")
            )
            events = tuple(
                event
                for event in events
                if media_seconds(event.trigger_at) + self._config.evidence.after_seconds
                <= horizon
            )
        stored = store.persist_pairs(
            run_id, zip(events, self.manifests_for(events), strict=True)
        )
        self._emit(
            EngineLogEventKind.PERSISTED, {"run_id": run_id, "events": len(stored)}
        )
        return stored

    # --- internals ---------------------------------------------------------------------
    def _camera_id(self, record: FrameRecord) -> str:
        return record.camera_id or self._scene.scene.camera_id

    def _emit(self, kind: EngineLogEventKind, data: EventData) -> None:
        event = EngineLogEvent(
            sequence=self._sequence,
            kind=kind,
            at=self._clock() if self._clock is not None else None,
            data=data,
        )
        self._sink.emit(event)
        self._sequence += 1


def build_engine(
    *,
    scene: SceneConfig,
    config: EngineConfig,
    classifier: HelmetClassifier | None = None,
    sink: EngineLogSink | None = None,
    clock: Callable[[], datetime] | None = None,
    perf: Callable[[], float] | None = None,
    memory_probe: Callable[[], int] | None = None,
    gpu_probe: Callable[[], int] | None = None,
    output_root: Path | str | None = None,
    capture_overlay: bool = False,
) -> tuple[InferenceEngine, EventStore]:
    """Composition root: realise the declared **real** backends and wire an engine.

    Builds the RT-DETR detector from ``config.inference`` (required here --
    injectors that want stubs construct :class:`InferenceEngine` directly) and
    the P1-U9 IoU tracker from ``config.tracker``, and returns the engine
    paired with an :class:`EventStore` rooted at ``output_root`` (the store's
    gitignored default when ``None``).

    Raises:
        EngineConfigurationError: ``config.inference`` is ``None``.
        DetectorError subclasses: the backend/checkpoint cannot be loaded
            (typed, from the P1-U7 backend; fail-fast).
    """

    from .errors import EngineConfigurationError

    if config.inference is None:
        raise EngineConfigurationError(
            "build_engine realises real backends and needs config.inference; "
            "construct InferenceEngine directly to inject a detector"
        )
    engine = InferenceEngine(
        scene=scene,
        detector=build_detector(config.inference),
        tracker=IouTracker(
            config=config.tracker.backend,
            tracker_config=TrackerConfig(tracker=config.tracker.tracker_ref),
        ),
        detector_config=detector_adapter_config(config.inference),
        config=config,
        classifier=classifier,
        sink=sink,
        clock=clock,
        perf=perf,
        memory_probe=memory_probe,
        gpu_probe=gpu_probe,
        capture_overlay=capture_overlay,
    )
    store = EventStore(output_root) if output_root is not None else EventStore()
    return engine, store

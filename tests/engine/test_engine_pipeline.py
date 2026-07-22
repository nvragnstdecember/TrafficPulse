"""Engine integration + end-to-end + deterministic replay tests (H6).

Covers the full composed flow -- source -> scheduler -> runner -> pipeline core
-> multi-rule finalize -> evidence -> persistence -- with scripted seams (the
same fixture scripts the pipeline e2e tests use; nothing here fabricates a
detection outside an injected stub) and real ingestion for the file path.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from _engine_helpers import (
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    frame_records,
    stub_engine,
)
from _helmet_fixtures import (
    helmet_detector_config,
    helmet_example_scene,
    scripted_helmet_classifier,
    scripted_rider_detector,
)
from _pipeline_helpers import moving_down_detector, moving_raw
from _slice_fixtures import scripted_down_detector, write_wrong_way_clip
from _stopping_fixtures import (
    illegal_stopping_test_scene,
    scripted_stopping_detector,
    stopping_detector_config,
)

from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector import RawDetection, StubDetector
from trafficpulse.detector.frame import Frame
from trafficpulse.engine import (
    EngineLogEventKind,
    FileFrameSource,
    IllegalStoppingRuleConfig,
    IterableFrameSource,
    NoHelmetRuleConfig,
    ScheduleDecision,
    SchedulerConfig,
    WrongWayRuleConfig,
)
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline import IllegalStoppingPipeline, WrongWayPipeline
from trafficpulse.pipeline.no_helmet import NoHelmetPipeline
from trafficpulse.tracking import IouTracker

_WRONG_WAY = (WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),)


# --- end-to-end: synthetic stream --------------------------------------------------
def test_wrong_way_end_to_end_over_a_synthetic_stream() -> None:
    engine, sink = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    result = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))

    assert [event.violation_type for event in result.events] == [ViolationType.WRONG_WAY]
    assert len(result.manifests) == 1
    manifest = result.manifests[0]
    assert manifest.event_id == result.events[0].event_id
    assert manifest.trigger_frame is not None
    assert manifest.before_frame is not None  # the 1 s margin fits the 1.5 s stream
    assert result.metrics.frames_processed == 45
    assert result.metrics.events_confirmed == 1
    kinds = [event.kind for event in sink.events]
    assert kinds[0] == EngineLogEventKind.ENGINE_RESET
    assert EngineLogEventKind.FINALIZED in kinds
    assert kinds[-1] == EngineLogEventKind.ENGINE_STOP


def test_engine_events_equal_the_standalone_pipeline() -> None:
    """The composition adds no behaviour: one-rule engine == WrongWayPipeline."""

    records = frame_records(45)
    standalone = WrongWayPipeline(
        detector=moving_down_detector(45),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    ).process(records)
    engine, _ = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    result = engine.run(IterableFrameSource(records, source_id="vsrc-test"))
    assert result.events == standalone


# --- end-to-end: real file ingestion ------------------------------------------------
def test_wrong_way_end_to_end_over_a_real_clip(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    engine, _ = stub_engine(detector=scripted_down_detector(), rules=_WRONG_WAY)
    result = engine.run(FileFrameSource(clip, camera_id=SCENE.scene.camera_id))

    assert [event.violation_type for event in result.events] == [ViolationType.WRONG_WAY]
    trigger = result.manifests[0].trigger_frame
    assert trigger is not None
    # The reference names a real decoded frame of this source (ingestion identity).
    assert trigger.locator.startswith(f"frames/{SCENE.scene.camera_id}/vfrm-")


# --- multi-rule -----------------------------------------------------------------------
def test_multi_rule_run_is_the_union_of_the_standalone_pipelines() -> None:
    scene = illegal_stopping_test_scene()
    records = frame_records(40, interval_seconds=0.1)
    config = stopping_detector_config()

    stopping = IllegalStoppingPipeline(
        detector=scripted_stopping_detector(),
        tracker=IouTracker(),
        scene=scene,
        detector_config=config,
    ).process(records)
    wrong_way = WrongWayPipeline(
        detector=scripted_stopping_detector(),
        tracker=IouTracker(),
        scene=scene,
        detector_config=config,
        direction_id=NORTH_DIRECTION_ID,
    ).process(records)
    assert stopping  # the scenario genuinely confirms illegal stopping

    engine, _ = stub_engine(
        detector=scripted_stopping_detector(),
        rules=(
            WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),
            IllegalStoppingRuleConfig(),
        ),
        scene=scene,
        detector_config=config,
    )
    result = engine.run(IterableFrameSource(records, source_id="vsrc-test"))
    expected = tuple(
        sorted([*wrong_way, *stopping], key=lambda e: (e.trigger_at, e.event_id))
    )
    assert result.events == expected
    assert {event.violation_type for event in result.events} >= {
        ViolationType.ILLEGAL_STOPPING
    }


def test_no_helmet_rule_runs_through_the_pixel_observer() -> None:
    scene = helmet_example_scene()
    records = frame_records(30, interval_seconds=0.1)
    standalone = NoHelmetPipeline(
        detector=scripted_rider_detector(),
        tracker=IouTracker(),
        classifier=scripted_helmet_classifier(),
        scene=scene,
        detector_config=helmet_detector_config(),
    ).process(records)
    assert standalone  # the scripted rider genuinely confirms no-helmet

    engine, _ = stub_engine(
        detector=scripted_rider_detector(),
        rules=(NoHelmetRuleConfig(),),
        scene=scene,
        detector_config=helmet_detector_config(),
        classifier=scripted_helmet_classifier(),
    )
    result = engine.run(IterableFrameSource(records, source_id="vsrc-test"))
    assert result.events == standalone
    assert result.events[0].violation_type is ViolationType.NO_HELMET


# --- deterministic replay --------------------------------------------------------------
def test_replay_is_byte_identical() -> None:
    def run_once() -> tuple[str, ...]:
        engine, sink = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
        result = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))
        return (
            *(event.model_dump_json() for event in result.events),
            *(manifest.model_dump_json() for manifest in result.manifests),
            result.metrics.model_dump_json(),
            *(event.model_dump_json() for event in sink.events),
        )

    assert run_once() == run_once()


def test_rerunning_one_engine_instance_replays_identically() -> None:
    engine, _ = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    first = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))
    second = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))
    assert first.events == second.events
    assert first.manifests == second.manifests


# --- scheduling integration --------------------------------------------------------------
def test_frame_stride_processes_fewer_frames_and_still_confirms() -> None:
    # 90 frames at 30 fps, stride 2 -> 45 processed frames spanning 3 s: still
    # comfortably past the 1.0 s persistence, with half the inference work.
    detector = StubDetector(
        per_frame={i: (moving_raw(i, step=2.5),) for i in range(90)}
    )
    engine, _ = stub_engine(
        detector=detector,
        rules=_WRONG_WAY,
        scheduler=SchedulerConfig(frame_stride=2),
    )
    result = engine.run(IterableFrameSource(frame_records(90), source_id="vsrc-test"))
    assert result.metrics.frames_read == 90
    assert result.metrics.frames_processed == 45
    assert result.metrics.frames_skipped_stride == 45
    assert [event.violation_type for event in result.events] == [ViolationType.WRONG_WAY]


def test_backpressure_drops_and_logs_when_submitting_without_draining() -> None:
    engine, sink = stub_engine(
        detector=moving_down_detector(5),
        rules=_WRONG_WAY,
        scheduler=SchedulerConfig(queue_capacity=2),
    )
    engine.reset()
    records = frame_records(5)
    decisions = [engine.submit(record) for record in records]
    assert decisions.count(ScheduleDecision.ADMITTED) == 2
    assert decisions.count(ScheduleDecision.DROPPED_QUEUE_FULL) == 3
    assert engine.drain() == 2
    metrics = engine.metrics
    assert metrics.frames_dropped_backpressure == 3
    assert metrics.frames_processed == 2
    assert metrics.queue_peak == 2
    dropped = [e for e in sink.events if e.kind is EngineLogEventKind.FRAME_DROPPED]
    assert len(dropped) == 3


# --- batching integration ------------------------------------------------------------------
class _BatchScriptedDetector(StubDetector):
    """The moving-car script with the batch capability, counting batch calls."""

    def __init__(self, frames: int) -> None:
        super().__init__(per_frame={i: (moving_raw(i),) for i in range(frames)})
        self.batch_calls = 0

    def detect_batch(self, frames: Sequence[Frame]) -> Sequence[Sequence[RawDetection]]:
        self.batch_calls += 1
        return [self.detect(frame) for frame in frames]


def test_batched_engine_confirms_identically() -> None:
    baseline_engine, _ = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    baseline = baseline_engine.run(
        IterableFrameSource(frame_records(45), source_id="vsrc-test")
    )

    detector = _BatchScriptedDetector(45)
    engine, _ = stub_engine(detector=detector, rules=_WRONG_WAY, batch_size=8)
    # Feed all frames before draining so full batches actually form.
    engine.reset()
    for record in frame_records(45):
        engine.submit(record)
    engine.drain()
    events = engine.finalize()

    assert events == baseline.events
    assert detector.batch_calls > 0  # the batch path genuinely ran


# --- persistence ------------------------------------------------------------------------------
def test_persist_and_reload_round_trips_events_and_manifests(tmp_path: Path) -> None:
    engine, sink = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    result = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))
    store = EventStore(tmp_path)

    stored = engine.persist(result, store=store, run_id="run-1")
    assert len(stored) == 1
    reloaded = store.load("run-1")
    assert [pair.event for pair in reloaded] == list(result.events)
    assert [pair.manifest for pair in reloaded] == list(result.manifests)
    assert reloaded[0].manifest.trigger_frame is not None  # real reference persisted
    assert any(e.kind is EngineLogEventKind.PERSISTED for e in sink.events)


def test_checkpointing_is_incremental_and_idempotent(tmp_path: Path) -> None:
    # 90 frames at 30 fps (3.0 s): the wrong-way event triggers at ~1.03 s, so
    # its 1.0 s after-window closes at ~2.03 s (~frame 61) -- well inside the
    # stream, letting a mid-stream checkpoint persist a *complete* manifest.
    detector = StubDetector(per_frame={i: (moving_raw(i, step=2.5),) for i in range(90)})
    engine, _ = stub_engine(detector=detector, rules=_WRONG_WAY)
    store = EventStore(tmp_path)
    engine.reset()
    records = frame_records(90)

    # Early checkpoint: before the 1.0 s persistence is reached -> nothing yet.
    for record in records[:20]:
        engine.submit(record)
        engine.drain()
    assert engine.checkpoint(store=store, run_id="run-live") == ()

    # Confirmed but the after-window is still open -> deferred, not half-frozen.
    for record in records[20:45]:
        engine.submit(record)
        engine.drain()
    assert engine.checkpoint(store=store, run_id="run-live") == ()

    # Window closed -> persisted with its final after-frame reference.
    for record in records[45:70]:
        engine.submit(record)
        engine.drain()
    mid = engine.checkpoint(store=store, run_id="run-live")
    assert len(mid) == 1
    assert mid[0].manifest.after_frame is not None

    # The final checkpoint re-persists byte-identically (write-once, no conflict)
    # and the reload matches.
    for record in records[70:]:
        engine.submit(record)
        engine.drain()
    final = engine.checkpoint(store=store, run_id="run-live", final=True)
    assert final == mid
    assert [pair.event for pair in store.load("run-live")] == [
        pair.event for pair in final
    ]


def test_final_checkpoint_clamps_an_open_after_window(tmp_path: Path) -> None:
    # A 45-frame stream ends ~0.43 s after the trigger -- inside the 1.0 s
    # margin. A non-final checkpoint defers; the final one persists with the
    # after-frame clamped to the last processed frame (the stream truly ended).
    engine, _ = stub_engine(detector=moving_down_detector(45), rules=_WRONG_WAY)
    store = EventStore(tmp_path)
    engine.reset()
    for record in frame_records(45):
        engine.submit(record)
        engine.drain()
    assert engine.checkpoint(store=store, run_id="run-live") == ()
    final = engine.checkpoint(store=store, run_id="run-live", final=True)
    assert len(final) == 1
    after = final[0].manifest.after_frame
    assert after is not None
    assert after.locator.endswith("vfrm-44")  # clamped to the last processed frame


# --- boundary ----------------------------------------------------------------------------------
def test_engine_import_pulls_in_no_ml_framework() -> None:
    code = (
        "import sys; import trafficpulse.engine; "
        "banned = {'torch', 'transformers'} & set(sys.modules); "
        "sys.exit(1 if banned else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr

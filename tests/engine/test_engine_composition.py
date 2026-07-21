"""build_engine composition root + engine read-only surface (H6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _engine_helpers import (
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    frame_records,
    scripted_perf,
    stub_engine,
)
from _pipeline_helpers import moving_down_detector
from _slice_fixtures import write_wrong_way_clip

from trafficpulse.contracts import ObjectClass
from trafficpulse.detector.errors import DetectorError
from trafficpulse.engine import (
    EngineConfig,
    EngineConfigurationError,
    FileFrameSource,
    FrameScheduler,
    InferenceConfig,
    IterableFrameSource,
    LatencyKind,
    SchedulerConfig,
    WrongWayRuleConfig,
    build_engine,
)
from trafficpulse.engine import engine as engine_module
from trafficpulse.ingestion.video import VideoSourceMetadata

_RULES = (WrongWayRuleConfig(direction_id=NORTH_DIRECTION_ID),)
_INFERENCE = InferenceConfig(
    checkpoint="no/such/checkpoint-dir", label_map={"car": ObjectClass.CAR}, device="cpu"
)


# --- build_engine ----------------------------------------------------------------
def test_build_engine_requires_an_inference_block() -> None:
    with pytest.raises(EngineConfigurationError, match="config.inference"):
        build_engine(scene=SCENE, config=EngineConfig(rules=_RULES))


def test_build_engine_fails_fast_on_an_unavailable_backend_or_checkpoint() -> None:
    # Offline + nonexistent checkpoint: the P1-U7 backend raises its typed error
    # (artifact unavailable with the extra installed; dependency error without).
    with pytest.raises(DetectorError):
        build_engine(
            scene=SCENE, config=EngineConfig(rules=_RULES, inference=_INFERENCE)
        )


def test_build_engine_wires_a_working_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Swap only the backend factory: everything else (adapter config from the
    # inference declaration, IoU tracker from config.tracker, store rooting) is
    # the real composition-root wiring under test.
    monkeypatch.setattr(
        engine_module, "build_detector", lambda config: moving_down_detector(45)
    )
    engine, store = build_engine(
        scene=SCENE,
        config=EngineConfig(rules=_RULES, inference=_INFERENCE),
        output_root=tmp_path,
    )
    assert store.root == tmp_path
    result = engine.run(IterableFrameSource(frame_records(45), source_id="vsrc-test"))
    assert len(result.events) == 1
    # The adapter stamped the declared checkpoint's provenance onto the event.
    assert any(ref.name == _INFERENCE.checkpoint for ref in result.events[0].models)
    engine.persist(result, store=store, run_id="run-1")
    assert len(store.load("run-1")) == 1


def test_build_engine_defaults_the_store_to_the_gitignored_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trafficpulse.persistence.store import DEFAULT_RUN_ROOT

    monkeypatch.setattr(
        engine_module, "build_detector", lambda config: moving_down_detector(1)
    )
    _, store = build_engine(
        scene=SCENE, config=EngineConfig(rules=_RULES, inference=_INFERENCE)
    )
    assert store.root == DEFAULT_RUN_ROOT


# --- engine read-only surface + probe-driven timing ---------------------------------
def test_engine_exposes_its_config_and_live_metrics() -> None:
    engine, _ = stub_engine(
        detector=moving_down_detector(30),
        rules=_RULES,
        scheduler=SchedulerConfig(target_fps=15.0),
    )
    assert engine.config.scheduler.target_fps == 15.0
    assert engine.metrics.frames_processed == 0  # readable before any stream


def test_engine_fps_gate_and_frame_latency_with_a_perf_probe() -> None:
    from trafficpulse.engine import InferenceEngine, MemoryLogSink
    from trafficpulse.tracking import IouTracker

    engine = InferenceEngine(
        scene=SCENE,
        detector=moving_down_detector(30),
        tracker=IouTracker(),
        detector_config=DETECTOR_CONFIG,
        config=EngineConfig(rules=_RULES, scheduler=SchedulerConfig(target_fps=15.0)),
        sink=MemoryLogSink(),
        perf=scripted_perf(0.01),
    )
    result = engine.run(IterableFrameSource(frame_records(30), source_id="vsrc-test"))
    # A 30 fps stream decimated to 15 fps: every other frame skipped by media time.
    assert result.metrics.frames_skipped_fps == 15
    assert result.metrics.frames_processed == 15
    assert result.metrics.latencies[LatencyKind.FRAME.value].count == 15
    assert result.metrics.wall_fps is not None


# --- small read-only surfaces -----------------------------------------------------------
def test_component_read_only_surfaces(tmp_path: Path) -> None:
    scheduler = FrameScheduler(SchedulerConfig(frame_stride=3))
    assert scheduler.config.frame_stride == 3

    source = FileFrameSource(write_wrong_way_clip(tmp_path / "clip.mp4"))
    assert isinstance(source.metadata, VideoSourceMetadata)
    assert source.metadata.source_id == source.source_id

    from trafficpulse.detector import StubDetector
    from trafficpulse.engine import DetectorRunner, MetricsRecorder

    inner = StubDetector()
    assert DetectorRunner(inner, MetricsRecorder()).inner is inner

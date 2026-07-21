"""Detector runner + instrumented tracker: timing, counting, batching (H6)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from _engine_helpers import scripted_perf
from _pipeline_helpers import moving_raw

from trafficpulse.contracts import ModelRef, ObjectClass
from trafficpulse.detector import RawDetection, StubDetector
from trafficpulse.detector.frame import Frame
from trafficpulse.engine import (
    DetectorRunner,
    InferenceConfig,
    InstrumentedTracker,
    LatencyKind,
    MetricsRecorder,
    SupportsBatchDetect,
    detector_adapter_config,
    detector_model_ref,
    resolve_device,
)
from trafficpulse.tracking import StubTracker


def _frame(index: int) -> Frame:
    return Frame(
        camera_id="cam-a",
        frame_index=index,
        timestamp=datetime(1970, 1, 1, tzinfo=UTC),
    )


class _BatchCapableStub(StubDetector):
    """A scripted detector that also implements the batch capability."""

    def __init__(self, per_frame: dict[int, tuple[RawDetection, ...]]) -> None:
        super().__init__(per_frame=per_frame)
        self.batch_calls: list[int] = []

    def detect_batch(self, frames: Sequence[Frame]) -> Sequence[Sequence[RawDetection]]:
        self.batch_calls.append(len(frames))
        return [self.detect(frame) for frame in frames]


def _script(count: int) -> dict[int, tuple[RawDetection, ...]]:
    return {i: (moving_raw(i),) for i in range(count)}


# --- pass-through + counting -----------------------------------------------------
def test_runner_is_transparent_to_the_seam() -> None:
    inner = StubDetector(per_frame=_script(3))
    runner = DetectorRunner(inner, MetricsRecorder())
    for index in range(3):
        assert runner.detect(_frame(index)) == inner.detect(_frame(index))


def test_runner_counts_raw_detections() -> None:
    recorder = MetricsRecorder()
    runner = DetectorRunner(StubDetector(per_frame=_script(2)), recorder)
    runner.detect(_frame(0))
    runner.detect(_frame(1))
    runner.detect(_frame(9))  # unscripted frame: empty result
    assert recorder.detections == 2


def test_runner_times_only_with_a_perf_probe() -> None:
    silent = MetricsRecorder()
    DetectorRunner(StubDetector(per_frame=_script(1)), silent).detect(_frame(0))
    assert silent.snapshot().latencies == {}

    timed = MetricsRecorder(perf=scripted_perf())
    DetectorRunner(StubDetector(per_frame=_script(1)), timed).detect(_frame(0))
    summary = timed.snapshot().latencies[LatencyKind.INFERENCE.value]
    assert summary.count == 1
    assert summary.total_seconds == pytest.approx(1.0)  # scripted step


# --- batching ----------------------------------------------------------------------
def test_prefetch_noop_without_the_capability() -> None:
    runner = DetectorRunner(StubDetector(per_frame=_script(2)), MetricsRecorder())
    assert runner.prefetch([_frame(0), _frame(1)]) is False


def test_prefetch_noop_for_single_frame_batches() -> None:
    inner = _BatchCapableStub(_script(2))
    runner = DetectorRunner(inner, MetricsRecorder())
    assert runner.prefetch([_frame(0)]) is False
    assert inner.batch_calls == []


def test_batched_and_per_frame_paths_are_equivalent() -> None:
    inner = _BatchCapableStub(_script(4))
    runner = DetectorRunner(inner, MetricsRecorder())
    assert runner.prefetch([_frame(i) for i in range(4)]) is True
    batched = [runner.detect(_frame(i)) for i in range(4)]

    plain = DetectorRunner(StubDetector(per_frame=_script(4)), MetricsRecorder())
    assert batched == [plain.detect(_frame(i)) for i in range(4)]
    assert inner.batch_calls == [4]  # one real inference call for the batch


def test_batch_call_is_one_latency_sample() -> None:
    recorder = MetricsRecorder(perf=scripted_perf())
    runner = DetectorRunner(_BatchCapableStub(_script(3)), recorder)
    runner.prefetch([_frame(i) for i in range(3)])
    for index in range(3):
        runner.detect(_frame(index))
    assert recorder.snapshot().latencies[LatencyKind.INFERENCE.value].count == 1
    assert recorder.detections == 3


def test_unprefetched_frames_fall_through_to_the_backend() -> None:
    inner = _BatchCapableStub(_script(3))
    runner = DetectorRunner(inner, MetricsRecorder())
    runner.prefetch([_frame(0), _frame(1)])
    assert runner.detect(_frame(2)) == inner.detect(_frame(2))  # never prefetched


def test_clear_drops_prefetched_results() -> None:
    inner = _BatchCapableStub(_script(2))
    runner = DetectorRunner(inner, MetricsRecorder())
    runner.prefetch([_frame(0), _frame(1)])
    runner.clear()
    assert runner.detect(_frame(0)) == inner.detect(_frame(0))  # recomputed, not stale


def test_capability_protocol_is_structural() -> None:
    assert isinstance(_BatchCapableStub({}), SupportsBatchDetect)
    assert not isinstance(StubDetector(), SupportsBatchDetect)


# --- instrumented tracker -------------------------------------------------------------
def test_instrumented_tracker_counts_and_times() -> None:
    recorder = MetricsRecorder(perf=scripted_perf())
    tracker = InstrumentedTracker(StubTracker(), recorder)
    assert tracker.update([]) == ()
    assert recorder.snapshot().latencies[LatencyKind.TRACKING.value].count == 1
    tracker.reset()  # delegates without error


# --- composition-root helpers ----------------------------------------------------------
def test_resolve_device_passes_explicit_values_through() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_yields_a_backend_valid_device() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_detector_model_ref_prefers_the_explicit_ref() -> None:
    labels = {"car": ObjectClass.CAR}
    explicit = InferenceConfig(
        checkpoint="ckpt",
        label_map=labels,
        source_model=ModelRef(name="custom", version="1.0"),
    )
    assert detector_model_ref(explicit).name == "custom"
    derived = detector_model_ref(InferenceConfig(checkpoint="ckpt", label_map=labels))
    assert (derived.name, derived.version) == ("ckpt", "provisional")
    assert derived.weights_hash is None  # never fabricated


def test_detector_adapter_config_carries_the_gate_and_provenance() -> None:
    config = InferenceConfig(
        checkpoint="ckpt", label_map={"car": ObjectClass.CAR}, score_threshold=0.4
    )
    adapter = detector_adapter_config(config)
    assert adapter.score_threshold == 0.4
    assert adapter.label_map == {"car": ObjectClass.CAR}
    assert adapter.source_model is not None
    assert adapter.source_model.name == "ckpt"

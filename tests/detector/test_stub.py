"""Stub-detector behavior and dependency-injection tests (P1-U6).

Covers the deterministic scripted replay, per-frame vs default output, pixel
independence, immutability of returned sequences, the ``Detector`` interface
contract, and the ``adapt_from`` dependency-injection seam that wires an injected
``Detector`` into the adapter.
"""

from datetime import UTC, datetime

import numpy as np

from trafficpulse.contracts import Detection, ObjectClass
from trafficpulse.detector import (
    DetectionAdapter,
    Detector,
    DetectorConfig,
    Frame,
    RawDetection,
    StubDetector,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CAR = RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))
PERSON = RawDetection("person", 0.8, (3.0, 3.0, 4.0, 4.0))


def _frame(index: int = 0) -> Frame:
    return Frame(camera_id="cam1", frame_index=index, timestamp=TS)


def test_stub_is_a_detector() -> None:
    assert isinstance(StubDetector(), Detector)


def test_default_script_returned_for_any_frame() -> None:
    stub = StubDetector([CAR])
    assert list(stub.detect(_frame(0))) == [CAR]
    assert list(stub.detect(_frame(99))) == [CAR]


def test_empty_stub_returns_empty() -> None:
    assert tuple(StubDetector().detect(_frame())) == ()


def test_per_frame_script_overrides_default() -> None:
    stub = StubDetector([CAR], per_frame={5: [PERSON]})
    assert list(stub.detect(_frame(5))) == [PERSON]
    assert list(stub.detect(_frame(0))) == [CAR]  # unlisted index falls back to default


def test_detect_is_deterministic_across_calls() -> None:
    stub = StubDetector([CAR, PERSON])
    assert tuple(stub.detect(_frame(3))) == tuple(stub.detect(_frame(3)))


def test_detect_ignores_pixels() -> None:
    stub = StubDetector([CAR])
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with_pixels = Frame(camera_id="cam1", frame_index=0, timestamp=TS, image=image)
    assert list(stub.detect(with_pixels)) == list(stub.detect(_frame(0)))


def test_returned_sequence_is_immutable_tuple() -> None:
    stub = StubDetector([CAR])
    assert isinstance(stub.detect(_frame()), tuple)


def test_adapt_from_wires_injected_detector() -> None:
    """The DI seam: adapter depends on the ``Detector`` abstraction, not a concrete one."""

    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    stub = StubDetector([CAR])
    frame = _frame()
    dets = adapter.adapt_from(stub, frame)
    assert len(dets) == 1
    assert isinstance(dets[0], Detection)
    assert dets[0].object_class is ObjectClass.CAR


def test_adapt_from_equals_manual_detect_then_adapt() -> None:
    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    stub = StubDetector([CAR])
    frame = _frame()
    assert adapter.adapt_from(stub, frame) == adapter.adapt(frame, stub.detect(frame))

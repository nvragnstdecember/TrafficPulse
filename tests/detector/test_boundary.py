"""Boundary tests: nothing detector-specific escapes the seam (P1-U6, ADR-001).

These guard the two invariants that make the permissive-only posture bounded:
(1) importing the detector package pulls in **no** ML/detector framework, and
(2) the adapter's public output is exclusively the frozen U2 ``Detection``
contract -- no ``RawDetection``, label string, or framework object leaks out.
"""

import importlib
import sys
from datetime import UTC, datetime

from trafficpulse.contracts import Detection, ObjectClass
from trafficpulse.detector import DetectionAdapter, DetectorConfig, Frame, RawDetection

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# Detector/ML frameworks the permissive-only foundation must NOT import (ADR-001).
_FORBIDDEN_MODULES = (
    "torch",
    "torchvision",
    "ultralytics",
    "onnxruntime",
    "transformers",
    "cv2",
    "tensorflow",
    "paddle",
)


def test_importing_detector_package_pulls_in_no_ml_framework() -> None:
    # Re-import fresh so the assertion reflects this package's own import graph.
    importlib.reload(importlib.import_module("trafficpulse.detector"))
    imported = set(sys.modules)
    assert not (imported & set(_FORBIDDEN_MODULES))


def test_adapter_output_is_only_the_frozen_contract() -> None:
    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    frame = Frame(camera_id="cam1", frame_index=0, timestamp=TS)
    dets = adapter.adapt(frame, [RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))])
    assert all(isinstance(d, Detection) for d in dets)
    assert all(type(d) is Detection for d in dets)  # exactly Detection, no subclass leakage

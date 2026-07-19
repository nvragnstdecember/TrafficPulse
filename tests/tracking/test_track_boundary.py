"""Boundary tests: nothing tracker-specific escapes the seam (P1-U8).

Guards the two invariants that make the permissive-only tracker posture bounded:
(1) importing the tracking package pulls in **no** ML / tracker framework, and
(2) the tracker's public output is exclusively the frozen U2 ``TrackState``
contract -- no ``TrackAssignment`` or framework object leaks out.
"""

import importlib
import sys

from _builders import make_detection

from trafficpulse.contracts import TrackState
from trafficpulse.tracking import ScriptedAssignment, StubTracker

# ML / tracker frameworks the permissive-only foundation must NOT import
# (ADR-001; architecture-review §10 excludes AGPL BoxMOT).
_FORBIDDEN_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
    "boxmot",
    "bytetrack",
    "yolox",
    "lap",
    "scipy",
    "filterpy",
    "cv2",
    "onnxruntime",
    "tensorflow",
)


def test_importing_tracking_package_pulls_in_no_ml_or_tracker_framework() -> None:
    # Evict-first so the assertion reflects this package's OWN import graph
    # (other suites legitimately import torch/scipy into the shared process);
    # the causal pattern from the classifier boundary test (P4-U2).
    evicted = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in _FORBIDDEN_MODULES
    }
    try:
        importlib.reload(importlib.import_module("trafficpulse.tracking"))
        leaked = [n for n in sys.modules if n.split(".")[0] in _FORBIDDEN_MODULES]
        assert leaked == [], f"tracking import pulled in: {leaked}"
    finally:
        sys.modules.update(evicted)


def test_tracker_output_is_only_the_frozen_contract() -> None:
    tracker = StubTracker({0: [ScriptedAssignment("T1")]})
    states = tracker.update([make_detection(0)])
    assert all(isinstance(s, TrackState) for s in states)
    assert all(type(s) is TrackState for s in states)  # exactly TrackState, no subclass leakage

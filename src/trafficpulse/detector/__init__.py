"""Detector-integration foundation for TrafficPulse (Phase 1, unit P1-U6).

The permissive-only detector seam mandated by ADR-001: a framework-neutral
``Detector`` interface (the dependency-injection boundary), a deterministic
``DetectionAdapter`` that converts detector-native ``RawDetection`` output into
the frozen U2 ``Detection`` contract, a pydantic ``DetectorConfig``, and a
scripted ``StubDetector`` for tests.

It contains **no** detector implementation, no inference, no model loading, no
weights, no image preprocessing, and no ML dependency -- only the stable boundary
a future RT-DETR detector plugs into without an API change (its class vocabulary
maps through ``DetectorConfig.label_map``; its pixels arrive through the existing
optional ``Frame.image`` slot). Nothing detector-specific -- tensors, framework
objects, or label strings -- escapes this package; downstream layers consume
``trafficpulse.contracts.Detection`` only.
"""

from .adapter import DetectionAdapter
from .config import DetectorConfig
from .errors import DetectorError, InvalidFrameError, MalformedDetectorOutputError
from .frame import Frame
from .interface import Detector
from .raw import RawDetection
from .stub import StubDetector

__all__ = [
    # interface + implementations
    "Detector",
    "StubDetector",
    # conversion
    "DetectionAdapter",
    # configuration
    "DetectorConfig",
    # boundary types
    "Frame",
    "RawDetection",
    # errors
    "DetectorError",
    "InvalidFrameError",
    "MalformedDetectorOutputError",
]

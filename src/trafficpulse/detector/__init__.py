"""Detector-integration foundation for TrafficPulse (Phase 1, unit P1-U6).

The permissive-only detector seam mandated by ADR-001: a framework-neutral
``Detector`` interface (the dependency-injection boundary), a deterministic
``DetectionAdapter`` that converts detector-native ``RawDetection`` output into
the frozen U2 ``Detection`` contract, a pydantic ``DetectorConfig``, and a
scripted ``StubDetector`` for tests.

The P1-U6 foundation (interface, adapter, config, stub) carries **no** ML
dependency. The P1-U7 :class:`RTDetrDetector` is the first *real* backend: it
integrates the Apache-2.0 HuggingFace Transformers RT-DETR port (ADR-001), but
imports torch/transformers **lazily**, so importing this package still pulls in no
ML framework. Its class vocabulary maps through ``DetectorConfig.label_map`` and
its pixels arrive through the existing ``Frame.image`` slot -- no API change to the
seam. Nothing detector-specific -- tensors, framework objects, or label strings --
escapes this package; downstream layers consume ``trafficpulse.contracts.Detection``
only, always via the ``DetectionAdapter``.
"""

from .adapter import DetectionAdapter
from .config import DetectorConfig
from .errors import DetectorError, InvalidFrameError, MalformedDetectorOutputError
from .frame import Frame
from .interface import Detector
from .raw import RawDetection
from .rtdetr import (
    BackendDependencyError,
    BackendInferenceError,
    InvalidDeviceError,
    MalformedBackendOutputError,
    MissingFrameImageError,
    ModelArtifactUnavailableError,
    RTDetrBackendError,
    RTDetrConfig,
    RTDetrDetector,
)
from .stub import StubDetector

__all__ = [
    # interface + implementations
    "Detector",
    "StubDetector",
    "RTDetrDetector",
    # conversion
    "DetectionAdapter",
    # configuration
    "DetectorConfig",
    "RTDetrConfig",
    # boundary types
    "Frame",
    "RawDetection",
    # errors
    "DetectorError",
    "InvalidFrameError",
    "MalformedDetectorOutputError",
    # RT-DETR backend errors
    "RTDetrBackendError",
    "BackendDependencyError",
    "ModelArtifactUnavailableError",
    "InvalidDeviceError",
    "MissingFrameImageError",
    "MalformedBackendOutputError",
    "BackendInferenceError",
]

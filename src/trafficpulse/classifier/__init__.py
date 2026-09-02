"""Classifier-integration foundation for TrafficPulse (Phase 4, unit P4-U2).

The perception seam for **crop-level classification**: a framework-neutral
``HelmetClassifier`` interface (the dependency-injection boundary), the ``Crop``
input and ``RawHelmetPrediction`` output boundary types, an error taxonomy, and a
scripted ``StubHelmetClassifier`` for tests.

This is the ``detector`` package's P1-U6 pattern one level down, and it exists for
the same ADR-001 reason: the model sits behind a seam so that choosing, swapping,
or removing it is a localized change. Downstream layers depend on this interface
and on the frozen U2 ``HelmetStateObservation`` contract -- never on an ML
framework.

The P4-U2 foundation (interface, boundary types, stub) carries **no** ML
dependency. The P4-U3 :class:`ZeroShotHelmetClassifier` is the first *real*
backend: it integrates the Apache-2.0 HuggingFace ``transformers`` CLIP-family
image-text port (ADR-001 permissive-only; **no Ultralytics / AGPL**), but imports
torch/transformers **lazily**, so importing this package still pulls in no ML
framework. Its prompt vocabulary travels in ``RawHelmetPrediction.label`` and its
pixels arrive through the existing ``Crop.image`` slot -- no API change to the
seam.

:class:`ResNetHelmetClassifier` (P4-U6) is the second real backend: the trained
P4-U5 ResNet-50, run through ``torchvision`` rather than ``timm``, behind the same
seam. It is **selectable, never default** -- ADR-005 adopts no backend, and gates
adoption on evaluating a candidate on the *derived head crops* the runtime actually
produces rather than on the whole-motorcycle crops P4-U5 measured.

Because that backend is **binary**, it cannot emit ``turban``, which the no-helmet
rule's exemption depends on. :mod:`~trafficpulse.classifier.capabilities` turns that
from a silent behaviour change into a composition-time error: a backend declares what
it can say via ``HelmetClassifier.supported_labels``, and a turban-dependent policy
refuses to build on a backend that has declared it cannot produce the evidence.

Scope boundary (what this package does NOT do)
----------------------------------------------
It defines no head-crop geometry, rider-slot attribution, quality gate, label map,
or observation. Cropping and ``HelmetStateObservation`` production are P4-U4;
temporal aggregation, abstention, and the no-helmet rule are P4-U5. Rules never
call a classifier: models produce observations, and the reasoning layer decides
violations.
"""

from .capabilities import (
    TURBAN_LABEL,
    ClassifierCapabilityError,
    missing_labels,
    require_turban_capability,
)
from .crop import Crop
from .errors import HelmetClassifierError
from .interface import HelmetClassifier
from .raw import RawHelmetPrediction
from .resnet import (
    CheckpointUnavailableError,
    MalformedCheckpointError,
    MalformedResNetOutputError,
    ResNetBackendError,
    ResNetDependencyError,
    ResNetHelmetClassifier,
    ResNetHelmetConfig,
    ResNetInferenceEngine,
    ResNetInferenceError,
    ResNetInvalidDeviceError,
    ResNetMissingCropImageError,
)
from .stub import UNCERTAIN, StubHelmetClassifier
from .zeroshot import (
    DEFAULT_HELMET_PROMPTS,
    BackendDependencyError,
    BackendInferenceError,
    InvalidDeviceError,
    MalformedBackendOutputError,
    MissingCropImageError,
    ModelArtifactUnavailableError,
    ZeroShotBackendError,
    ZeroShotHelmetClassifier,
    ZeroShotHelmetConfig,
    ZeroShotInferenceEngine,
)

__all__ = [
    # interface + implementations
    "HelmetClassifier",
    "StubHelmetClassifier",
    "ZeroShotHelmetClassifier",
    "ResNetHelmetClassifier",
    # boundary types
    "Crop",
    "RawHelmetPrediction",
    # configuration
    "ZeroShotHelmetConfig",
    "ResNetHelmetConfig",
    "DEFAULT_HELMET_PROMPTS",
    # internal engine seams (fakeable; no framework type crosses them)
    "ZeroShotInferenceEngine",
    "ResNetInferenceEngine",
    # stub conveniences
    "UNCERTAIN",
    # backend capability declaration (turban-blindness is loud, never silent)
    "ClassifierCapabilityError",
    "TURBAN_LABEL",
    "missing_labels",
    "require_turban_capability",
    # errors
    "HelmetClassifierError",
    # resnet backend errors
    "ResNetBackendError",
    "ResNetDependencyError",
    "CheckpointUnavailableError",
    "MalformedCheckpointError",
    "ResNetInvalidDeviceError",
    "ResNetMissingCropImageError",
    "MalformedResNetOutputError",
    "ResNetInferenceError",
    # zero-shot backend errors
    "ZeroShotBackendError",
    "BackendDependencyError",
    "ModelArtifactUnavailableError",
    "InvalidDeviceError",
    "MissingCropImageError",
    "MalformedBackendOutputError",
    "BackendInferenceError",
]

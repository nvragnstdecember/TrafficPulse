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

This foundation carries **no** ML dependency: importing this package pulls in no
torch, no transformers, and loads no weights. A real backend (P4-U3) will import
its framework lazily, exactly as ``detector/rtdetr.py`` does, so the base install
and every unit test stay ML-free and network-free.

Scope boundary (what this unit does NOT do)
-------------------------------------------
It performs no inference and defines no model, label vocabulary, head-crop
geometry, rider-slot attribution, quality gate, or observation. Cropping and
``HelmetStateObservation`` production are P4-U4; temporal aggregation and the
no-helmet rule are P4-U5; a real backend is P4-U3. Rules never call a classifier:
models produce observations, and the reasoning layer decides violations.
"""

from .crop import Crop
from .errors import HelmetClassifierError
from .interface import HelmetClassifier
from .raw import RawHelmetPrediction
from .stub import UNCERTAIN, StubHelmetClassifier

__all__ = [
    # interface + implementations
    "HelmetClassifier",
    "StubHelmetClassifier",
    # boundary types
    "Crop",
    "RawHelmetPrediction",
    # stub conveniences
    "UNCERTAIN",
    # errors
    "HelmetClassifierError",
]

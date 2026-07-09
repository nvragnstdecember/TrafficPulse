"""Detector-integration configuration (P1-U6).

``DetectorConfig`` is the *framework-neutral adapter configuration*: the settings
the adapter needs to convert any detector's ``RawDetection`` output into frozen
``Detection`` contracts. It is intentionally free of RT-DETR / ONNX / Torch
settings (device, weights path, input size, batching) -- those belong to a future
detector *implementation's* own configuration, not to this shared seam, so no
detector-specific assumption leaks into callers (ADR-001). The three fields here
are exactly what any detector adapter needs, which is what lets RT-DETR slot in
without an API change: its class vocabulary maps through ``label_map`` and its
weights version travels in ``source_model``.

Validation reuses pydantic (already a project runtime dependency); no new
dependency is added. The model is frozen + strict (``extra='forbid'``) like the
domain contracts, but it lives in the ``detector`` package rather than
``contracts`` because it is component configuration, not part of the typed
perception->reasoning data flow.
"""

from pydantic import BaseModel, ConfigDict, field_validator

from ..contracts import ModelRef, ObjectClass
from ..contracts.primitives import Confidence


class DetectorConfig(BaseModel):
    """Framework-neutral configuration for adapting detector output to ``Detection``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_map: dict[str, ObjectClass]
    """Detector-native class label -> frozen ``ObjectClass``. A label absent from
    this map is a class the project does not model and is dropped by the adapter."""

    score_threshold: Confidence = 0.0
    """Drop *valid* detections whose confidence is below this value. The default
    ``0.0`` keeps every detection with a valid score (pure pass-through)."""

    source_model: ModelRef | None = None
    """Stamped onto every produced ``Detection.source_model`` as provenance. Named
    to match the ``Detection`` field (and to avoid pydantic's ``model_``
    protected namespace)."""

    @field_validator("label_map")
    @classmethod
    def _non_empty_label_map(cls, value: dict[str, ObjectClass]) -> dict[str, ObjectClass]:
        if not value:
            raise ValueError("label_map must map at least one detector label to an ObjectClass")
        if any(not label for label in value):
            raise ValueError("label_map keys (detector labels) must be non-empty strings")
        return value

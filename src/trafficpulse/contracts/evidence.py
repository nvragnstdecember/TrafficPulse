"""Evidence-manifest contract (metadata for an evidence package).

Data only: references and metadata that *describe* an evidence package. It
performs no filesystem storage, hashing, clip generation, or artifact
creation (architecture-review §19 — those are Phase 1+ behaviour).
"""

from pydantic import AwareDatetime

from .enums import ArtifactKind
from .primitives import (
    Confidence,
    ContractModel,
    MeasuredValue,
    ModelRef,
    NonEmptyStr,
    NonNegativeInt,
    Sha256Hex,
)


class ArtifactReference(ContractModel):
    """A reference to one evidence artifact (metadata only, no file access).

    ``locator`` is an opaque content address or relative path; ``sha256``, if
    present, is the artifact's integrity hash (validated as hex, not computed).
    """

    kind: ArtifactKind
    locator: NonEmptyStr
    sha256: Sha256Hex | None = None
    media_type: str | None = None


class OcrResult(ContractModel):
    """A recognized plate string with confidence (data only, no OCR)."""

    text: str
    confidence: Confidence
    per_char_confidence: tuple[Confidence, ...] | None = None


class RuleTraceStep(ContractModel):
    """One ordered step of a rule's reasoning trace."""

    index: NonNegativeInt
    label: NonEmptyStr
    note: str | None = None
    measurements: tuple[MeasuredValue, ...] = ()


class EvidenceManifest(ContractModel):
    """Metadata describing the evidence package for a confirmed event.

    Distinct from the ``ConfirmedEvent`` itself: the event is the decision, the
    manifest is the reviewable evidence around it. Artifacts are referenced by
    ``ArtifactReference`` (locator + optional hash), never embedded or stored.
    """

    evidence_package_id: NonEmptyStr
    event_id: NonEmptyStr
    before_frame: ArtifactReference | None = None
    trigger_frame: ArtifactReference | None = None
    after_frame: ArtifactReference | None = None
    clip: ArtifactReference | None = None
    trajectory: ArtifactReference | None = None
    plate_crop: ArtifactReference | None = None
    ocr: OcrResult | None = None
    rule_trace: tuple[RuleTraceStep, ...] = ()
    additional_artifacts: tuple[ArtifactReference, ...] = ()
    models: tuple[ModelRef, ...] = ()
    code_version: str | None = None
    scene_config_hash: Sha256Hex | None = None
    created_at: AwareDatetime

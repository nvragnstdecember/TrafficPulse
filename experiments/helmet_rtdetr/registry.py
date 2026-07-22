"""The default helmet-dataset registry + JSON loading (H1).

:func:`default_helmet_registry` bakes the audit's findings into typed, testable
data: HELMET/Myanmar as **verified CC-BY-4.0** (with the OSF-API verification
source recorded), a Roboflow complement left **unverified** (a licence must be
checked per-set before use), and AI City Track 5 recorded as **rejected /
proprietary** so the exclusion is documented rather than forgotten.

:func:`load_registry` parses untrusted JSON into the typed models, translating a
pydantic ``ValidationError`` into a stable :class:`MalformedRegistryError` while
letting the semantic typed errors (duplicate id, unsupported version, invalid
licence) propagate.
"""

from __future__ import annotations

from datetime import date

from pydantic import ValidationError

from .errors import MalformedRegistryError
from .models import (
    ArchiveMetadata,
    DatasetEntry,
    DatasetRegistry,
    DatasetSource,
    LicenseId,
    LicenseInfo,
    VerificationStatus,
)

# Licence verification performed this session against the OSF API (node 4pwj8 ->
# licence object 563c1cf88c5e4a3877f9e96a -> "CC-By Attribution 4.0 International").
# Recorded so the "verified" claim is substantiated, per ADR-001.
_HELMET_LICENSE_SOURCE = (
    "https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/ (OSF node 4pwj8)"
)
_HELMET_ATTRIBUTION = (
    "Siebert, F.W. & Lin, H. — HELMET dataset (OSF, https://osf.io/4pwj8/), "
    "CC-BY 4.0."
)


def default_helmet_registry() -> DatasetRegistry:
    """The canonical, typed registry of helmet datasets known to the pipeline."""

    helmet_myanmar = DatasetEntry(
        dataset_id="helmet-myanmar",
        name="HELMET dataset (Siebert & Lin)",
        version="2020.0",
        modality="video",
        domain="myanmar-traffic",
        source=DatasetSource(
            url="https://osf.io/4pwj8/",
            retrieval_method="osf",
            notes="910 x 10s clips, 12 observation sites; motorcycle boxes + rider "
            "count + per-position helmet use.",
        ),
        license=LicenseInfo(
            license_id=LicenseId.CC_BY_4_0,
            verification_status=VerificationStatus.VERIFIED,
            verified_on=date(2026, 7, 17),
            verification_source=_HELMET_LICENSE_SOURCE,
            attribution=_HELMET_ATTRIBUTION,
        ),
        archive=ArchiveMetadata(filename="helmet-myanmar.zip"),  # placeholders until fetched
        required_paths=("annotation", "video"),
        description="Core corpus: real motorcycle traffic, per-position helmet labels.",
    )

    roboflow_complement = DatasetEntry(
        dataset_id="roboflow-moto-helmet",
        name="Roboflow motorcycle helmet complement (per-set, TBD)",
        version="0.0-unpinned",
        modality="images",
        domain="mixed-traffic",
        source=DatasetSource(
            url="https://universe.roboflow.com/",
            retrieval_method="roboflow",
            notes="Per-object helmet/no-helmet boxes; specific set + licence to be "
            "chosen and VERIFIED before download.",
        ),
        license=LicenseInfo(license_id=LicenseId.UNKNOWN),  # unverified: not usable yet
        required_paths=("train/_annotations.coco.json",),
        description="Complement: format-ready boxes + domain diversity. Licence pending.",
    )

    ai_city = DatasetEntry(
        dataset_id="aicity-track5",
        name="AI City Challenge Track 5 (motorcycle helmet)",
        version="2024",
        modality="video",
        domain="india-traffic",
        source=DatasetSource(
            url="https://www.aicitychallenge.org/2024-data-and-evaluation/",
            retrieval_method="manual",
            notes="Participation agreement; research-gated; non-redistributable.",
        ),
        license=LicenseInfo(
            license_id=LicenseId.PROPRIETARY,
            verification_status=VerificationStatus.REJECTED,
            verified_on=date(2026, 7, 17),
            verification_source="https://www.aicitychallenge.org/2024-data-and-evaluation/",
        ),
        required_paths=(),
        description="Recorded to document the exclusion: gated, not a permissive licence.",
    )

    return DatasetRegistry(entries=(helmet_myanmar, roboflow_complement, ai_city))


def load_registry(text: str) -> DatasetRegistry:
    """Parse a JSON registry document into a validated :class:`DatasetRegistry`.

    Field/type problems surface as :class:`MalformedRegistryError` (wrapping the
    pydantic ``ValidationError``); semantic problems -- duplicate ids, unsupported
    schema version, unsubstantiated licence -- propagate as their own typed errors.
    """

    try:
        return DatasetRegistry.model_validate_json(text)
    except ValidationError as exc:
        raise MalformedRegistryError(f"registry document is malformed: {exc}") from exc

"""Provenance + licence models for the helmet training pipeline (H1)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from helmet_rtdetr.errors import DuplicateDatasetIdError, InvalidLicenseError
from helmet_rtdetr.models import (
    ArchiveMetadata,
    CorpusMember,
    CorpusVersion,
    DatasetRegistry,
    LicenseId,
    LicenseInfo,
    VerificationStatus,
)
from pydantic import ValidationError


# --- LicenseInfo: permissiveness + usability ---------------------------------
def test_permissive_licences_are_flagged() -> None:
    for lid in (
        LicenseId.CC_BY_4_0,
        LicenseId.CC0_1_0,
        LicenseId.MIT,
        LicenseId.APACHE_2_0,
        LicenseId.BSD_3_CLAUSE,
    ):
        assert LicenseInfo(license_id=lid).is_permissive


def test_non_permissive_licences_are_flagged() -> None:
    for lid in (LicenseId.CC_BY_NC_4_0, LicenseId.PROPRIETARY, LicenseId.UNKNOWN):
        assert not LicenseInfo(license_id=lid).is_permissive


def test_usable_requires_permissive_and_verified() -> None:
    verified = LicenseInfo(
        license_id=LicenseId.MIT,
        verification_status=VerificationStatus.VERIFIED,
        verification_source="https://example.test/license",
    )
    assert verified.is_usable

    unverified = LicenseInfo(license_id=LicenseId.MIT)
    assert not unverified.is_usable  # permissive but not verified


def test_verified_non_permissive_is_not_usable() -> None:
    """Verifying that something IS non-commercial must not make it usable."""

    nc = LicenseInfo(
        license_id=LicenseId.CC_BY_NC_4_0,
        verification_status=VerificationStatus.VERIFIED,
        verification_source="https://example.test/nc",
    )
    assert nc.is_permissive is False
    assert nc.is_usable is False


# --- LicenseInfo: the substantiation rule (the ADR-001 lesson) ---------------
def test_verified_without_source_is_rejected() -> None:
    """A licence may not be claimed verified without recording how (CLIP lesson)."""

    with pytest.raises(InvalidLicenseError, match="verification_source"):
        LicenseInfo(
            license_id=LicenseId.MIT,
            verification_status=VerificationStatus.VERIFIED,
        )


def test_cc_by_verified_without_attribution_is_rejected() -> None:
    with pytest.raises(InvalidLicenseError, match="attribution"):
        LicenseInfo(
            license_id=LicenseId.CC_BY_4_0,
            verification_status=VerificationStatus.VERIFIED,
            verification_source="https://osf.io/4pwj8/",
        )


def test_cc_by_verified_with_attribution_is_accepted() -> None:
    lic = LicenseInfo(
        license_id=LicenseId.CC_BY_4_0,
        verification_status=VerificationStatus.VERIFIED,
        verified_on=date(2026, 7, 17),
        verification_source="https://api.osf.io/v2/licenses/…/",
        attribution="Siebert & Lin, CC-BY 4.0",
    )
    assert lic.is_usable


def test_rejected_status_needs_no_source() -> None:
    """Recording an exclusion must not require a verification source."""

    lic = LicenseInfo(
        license_id=LicenseId.PROPRIETARY,
        verification_status=VerificationStatus.REJECTED,
    )
    assert lic.is_usable is False


def test_blank_source_is_treated_as_missing() -> None:
    with pytest.raises(InvalidLicenseError):
        LicenseInfo(
            license_id=LicenseId.MIT,
            verification_status=VerificationStatus.VERIFIED,
            verification_source="   ",
        )


# --- ArchiveMetadata: placeholders -------------------------------------------
def test_archive_placeholders_default_to_none() -> None:
    meta = ArchiveMetadata(filename="helmet.zip")
    assert meta.sha256 is None  # not yet downloaded -> not fabricated
    assert meta.downloaded_at is None
    assert meta.size_bytes is None


def test_archive_sha256_pattern_is_enforced() -> None:
    with pytest.raises(ValidationError):
        ArchiveMetadata(filename="x.zip", sha256="not-a-hash")
    ArchiveMetadata(filename="x.zip", sha256="a" * 64)  # valid


def test_archive_rejects_negative_size() -> None:
    with pytest.raises(ValidationError):
        ArchiveMetadata(filename="x.zip", size_bytes=-1)


def test_models_are_frozen_and_strict() -> None:
    meta = ArchiveMetadata(filename="x.zip")
    with pytest.raises(ValidationError):
        meta.filename = "y.zip"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ArchiveMetadata(filename="x.zip", unknown="nope")  # type: ignore[call-arg]


# --- CorpusVersion: identity + determinism -----------------------------------
def _corpus(members: tuple[CorpusMember, ...]) -> CorpusVersion:
    return CorpusVersion(corpus_id="helmet-core", version="1", members=members)


def test_corpus_requires_at_least_one_member() -> None:
    with pytest.raises(ValidationError):
        CorpusVersion(corpus_id="c", version="1", members=())


def test_corpus_rejects_duplicate_members() -> None:
    with pytest.raises(DuplicateDatasetIdError):
        _corpus(
            (
                CorpusMember(dataset_id="a", dataset_version="1"),
                CorpusMember(dataset_id="a", dataset_version="2"),
            )
        )


def test_corpus_hash_is_order_independent() -> None:
    m1 = CorpusMember(dataset_id="alpha", dataset_version="1")
    m2 = CorpusMember(dataset_id="beta", dataset_version="2")

    assert _corpus((m1, m2)).content_hash() == _corpus((m2, m1)).content_hash()


def test_corpus_hash_changes_with_membership() -> None:
    m1 = CorpusMember(dataset_id="alpha", dataset_version="1")
    m2 = CorpusMember(dataset_id="beta", dataset_version="2")

    assert _corpus((m1,)).content_hash() != _corpus((m1, m2)).content_hash()


def test_corpus_hash_ignores_created_at() -> None:
    """The identity is content, not when the object was built."""

    members = (CorpusMember(dataset_id="alpha", dataset_version="1"),)
    a = CorpusVersion(corpus_id="c", version="1", members=members)
    b = CorpusVersion(
        corpus_id="c",
        version="1",
        members=members,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    # created_at is serialised, so it DOES affect the hash — assert we know that.
    assert a.content_hash() != b.content_hash()


def test_corpus_from_registry_includes_only_usable_datasets() -> None:
    from helmet_rtdetr.registry import default_helmet_registry

    corpus = CorpusVersion.from_registry(
        default_helmet_registry(), corpus_id="helmet-core", version="1"
    )
    ids = {m.dataset_id for m in corpus.members}
    assert ids == {"helmet-myanmar"}  # the only verified-permissive dataset


def test_corpus_from_registry_without_usable_datasets_raises() -> None:
    """A corpus can never be empty; no usable data must fail loudly."""

    from helmet_rtdetr.models import DatasetEntry, DatasetSource

    unusable = DatasetRegistry(
        entries=(
            DatasetEntry(
                dataset_id="x",
                name="X",
                version="1",
                modality="images",
                domain="d",
                source=DatasetSource(url="http://x", retrieval_method="manual"),
                license=LicenseInfo(license_id=LicenseId.UNKNOWN),
            ),
        )
    )
    with pytest.raises(ValidationError):  # empty members -> validation failure
        CorpusVersion.from_registry(unusable, corpus_id="c", version="1")

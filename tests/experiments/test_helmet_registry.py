"""Dataset registry: construction, validation, lookup, determinism (H1)."""

from __future__ import annotations

import pytest
from helmet_rtdetr.errors import (
    DatasetNotFoundError,
    DuplicateDatasetIdError,
    MalformedRegistryError,
    UnsupportedRegistryVersionError,
)
from helmet_rtdetr.models import (
    DatasetEntry,
    DatasetRegistry,
    DatasetSource,
    LicenseId,
    LicenseInfo,
    VerificationStatus,
)
from helmet_rtdetr.registry import default_helmet_registry, load_registry
from pydantic import ValidationError


def entry(dataset_id: str, license_info: LicenseInfo | None = None) -> DatasetEntry:
    return DatasetEntry(
        dataset_id=dataset_id,
        name=f"Dataset {dataset_id}",
        version="1.0",
        modality="images",
        domain="test",
        source=DatasetSource(url="https://example.test", retrieval_method="manual"),
        license=license_info or LicenseInfo(license_id=LicenseId.MIT),
        required_paths=("a.json",),
    )


# --- successful construction + lookup ----------------------------------------
def test_registry_registers_entries() -> None:
    reg = DatasetRegistry(entries=(entry("alpha"), entry("beta")))
    assert reg.ids() == ("alpha", "beta")
    assert reg.get("alpha").name == "Dataset alpha"


def test_empty_registry_is_valid() -> None:
    assert DatasetRegistry().entries == ()


def test_ids_are_sorted_regardless_of_input_order() -> None:
    reg = DatasetRegistry(entries=(entry("gamma"), entry("alpha"), entry("beta")))
    assert reg.ids() == ("alpha", "beta", "gamma")


# --- duplicate ids -----------------------------------------------------------
def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(DuplicateDatasetIdError, match="alpha"):
        DatasetRegistry(entries=(entry("alpha"), entry("alpha")))


# --- unsupported version -----------------------------------------------------
def test_unsupported_schema_version_is_rejected() -> None:
    with pytest.raises(UnsupportedRegistryVersionError, match="99.0.0"):
        DatasetRegistry(schema_version="99.0.0", entries=())


# --- lookup errors -----------------------------------------------------------
def test_get_unknown_dataset_raises() -> None:
    reg = DatasetRegistry(entries=(entry("alpha"),))
    with pytest.raises(DatasetNotFoundError, match="missing"):
        reg.get("missing")


# --- usable() gate -----------------------------------------------------------
def test_usable_returns_only_verified_permissive_entries() -> None:
    usable_lic = LicenseInfo(
        license_id=LicenseId.CC0_1_0,
        verification_status=VerificationStatus.VERIFIED,
        verification_source="https://example.test/cc0",
    )
    reg = DatasetRegistry(
        entries=(
            entry("verified", usable_lic),
            entry("unverified", LicenseInfo(license_id=LicenseId.MIT)),
        )
    )
    assert tuple(e.dataset_id for e in reg.usable()) == ("verified",)


# --- deterministic serialization ---------------------------------------------
def test_canonical_json_is_order_independent() -> None:
    a = DatasetRegistry(entries=(entry("alpha"), entry("beta")))
    b = DatasetRegistry(entries=(entry("beta"), entry("alpha")))
    assert a.canonical_json() == b.canonical_json()


def test_content_hash_is_stable_and_order_independent() -> None:
    a = DatasetRegistry(entries=(entry("alpha"), entry("beta")))
    b = DatasetRegistry(entries=(entry("beta"), entry("alpha")))
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() == a.content_hash()  # stable across calls


def test_content_hash_changes_with_content() -> None:
    a = DatasetRegistry(entries=(entry("alpha"),))
    b = DatasetRegistry(entries=(entry("alpha"), entry("beta")))
    assert a.content_hash() != b.content_hash()


def test_registry_round_trips_through_load_registry() -> None:
    original = DatasetRegistry(entries=(entry("alpha"), entry("beta")))
    reloaded = load_registry(original.model_dump_json())
    assert reloaded.content_hash() == original.content_hash()


# --- malformed registry (load path) ------------------------------------------
def test_malformed_json_raises_malformed_registry_error() -> None:
    with pytest.raises(MalformedRegistryError):
        load_registry('{"entries": [{"dataset_id": "x"}]}')  # missing required fields


def test_malformed_bad_type_raises_malformed_registry_error() -> None:
    with pytest.raises(MalformedRegistryError):
        load_registry('{"schema_version": 1.0, "entries": []}')  # wrong type


def test_load_registry_propagates_duplicate_id_error() -> None:
    """Semantic errors are NOT masked as malformed; they keep their own type."""

    reg = DatasetRegistry(entries=(entry("alpha"),))
    dup_json = reg.model_dump_json().replace(
        '"entries":[', '"entries":[' + reg.get("alpha").model_dump_json() + ","
    )
    with pytest.raises(DuplicateDatasetIdError):
        load_registry(dup_json)


def test_invalid_dataset_id_slug_is_rejected() -> None:
    with pytest.raises(ValidationError):
        entry("Not A Slug")


# --- the default registry (bakes the audit findings) -------------------------
def test_default_registry_is_valid_and_unique() -> None:
    reg = default_helmet_registry()
    assert reg.ids() == ("aicity-track5", "helmet-myanmar", "roboflow-moto-helmet")


def test_default_registry_marks_helmet_myanmar_usable() -> None:
    lic = default_helmet_registry().get("helmet-myanmar").license
    assert lic.license_id is LicenseId.CC_BY_4_0
    assert lic.verification_status is VerificationStatus.VERIFIED
    assert lic.is_usable
    assert lic.verification_source and "osf" in lic.verification_source.lower()
    assert lic.attribution  # CC-BY requires it, and it is present


def test_default_registry_leaves_roboflow_unverified() -> None:
    lic = default_helmet_registry().get("roboflow-moto-helmet").license
    assert not lic.is_usable  # licence must be checked per-set before use


def test_default_registry_records_aicity_exclusion() -> None:
    lic = default_helmet_registry().get("aicity-track5").license
    assert lic.license_id is LicenseId.PROPRIETARY
    assert lic.verification_status is VerificationStatus.REJECTED
    assert not lic.is_usable


def test_default_registry_has_exactly_one_usable_dataset() -> None:
    assert tuple(e.dataset_id for e in default_helmet_registry().usable()) == ("helmet-myanmar",)

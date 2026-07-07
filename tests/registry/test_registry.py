"""Validation and U2/U3-consistency tests for the dataset registry (U4).

Closed vocabularies are read from ``registry/schema.yaml`` and ontology/violation
references are validated against the live U2 contracts, so the registry, schema,
and contracts cannot silently drift apart. No dataset is downloaded or prepared.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from trafficpulse.contracts.enums import (
    HelmetState,
    ObjectClass,
    SignalState,
    ViolationType,
)
from trafficpulse.contracts.observations import OBSERVATION_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "registry" / "schema.yaml"
DATASETS_DIR = REPO_ROOT / "registry" / "datasets"

SCHEMA: dict[str, Any] = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
ENTRY_PATHS = sorted(DATASETS_DIR.glob("*.yaml"))

EXTERNAL_SOURCE_TYPES = {"academic_dataset", "challenge_dataset", "public_repository"}
AICITY_UNCONFIRMED_STATES = {"application_required", "restricted", "unconfirmed", "unavailable"}

VALID_ONTOLOGY_REFS: dict[str, set[str]] = {
    "violation_types": {v.value for v in ViolationType},
    "object_classes": {o.value for o in ObjectClass},
    "helmet_states": {h.value for h in HelmetState},
    "signal_states": {s.value for s in SignalState},
    "observation_types": {v.model_fields["obs_type"].default for v in OBSERVATION_VARIANTS},
}


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get(entry: dict[str, Any], dotted: str) -> Any:
    node: Any = entry
    for key in dotted.split("."):
        node = node[key]
    return node


def _iter_keys(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _iter_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_keys(value)


def _entry_id(path: Path) -> str:
    return path.stem


# --- Schema and load ---------------------------------------------------------
def test_schema_parses() -> None:
    assert isinstance(SCHEMA, dict)
    assert SCHEMA["schema"]["version"]
    assert "vocabularies" in SCHEMA
    assert "required_fields" in SCHEMA


def test_registry_non_empty() -> None:
    assert len(ENTRY_PATHS) >= 6


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_entry_parses(path: Path) -> None:
    assert isinstance(_load(path), dict)


# --- Required fields ---------------------------------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_required_fields_present(path: Path) -> None:
    entry = _load(path)
    nullable = set(SCHEMA["nullable_fields"])
    for section, fields in SCHEMA["required_fields"].items():
        assert section in entry, f"{path.stem}: missing section {section}"
        for field in fields:
            assert field in entry[section], f"{path.stem}: missing {section}.{field}"
            if field not in nullable:
                assert entry[section][field] not in (None, ""), f"empty {section}.{field}"


# --- Identity and uniqueness -------------------------------------------------
def test_ids_unique() -> None:
    ids = [_load(p)["identity"]["id"] for p in ENTRY_PATHS]
    assert len(ids) == len(set(ids)), f"duplicate dataset ids: {ids}"


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_task_categories_valid(path: Path) -> None:
    cats = _load(path)["identity"]["task_categories"]
    assert cats, f"{path.stem}: empty task_categories"
    assert set(cats) <= set(SCHEMA["vocabularies"]["task_category"])


# --- Closed status vocabularies ----------------------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_field_values_in_vocabularies(path: Path) -> None:
    entry = _load(path)
    for field, vocab_name in SCHEMA["field_vocabularies"].items():
        allowed = SCHEMA["vocabularies"][vocab_name]
        value = _get(entry, field)
        assert value in allowed, f"{path.stem}: {field}={value!r} not in {vocab_name}"


# --- Acquisition / access honesty --------------------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_not_downloaded(path: Path) -> None:
    entry = _load(path)
    assert entry["integrity"]["local_acquisition_status"] == "not_downloaded"
    assert entry["integrity"]["acquisition_date"] in (None, "pending")


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_access_confirmation_requires_evidence(path: Path) -> None:
    entry = _load(path)
    confirmed = entry["access"]["access_confirmed_by_project"]
    assert isinstance(confirmed, bool)
    if confirmed:
        assert entry["provenance"]["date_last_verified"], "confirmed access without date"
        assert entry["access"]["notes"], "confirmed access without notes"


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_no_local_paths(path: Path) -> None:
    entry = _load(path)
    forbidden = set(SCHEMA["forbidden_keys"])
    present = set(_iter_keys(entry))
    assert not (present & forbidden), f"{path.stem}: forbidden keys {present & forbidden}"


# --- Licensing honesty -------------------------------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_licence_consistency(path: Path) -> None:
    lic = _load(path)["licensing"]
    if lic["status"] == "verified":
        assert lic["identifier"], f"{path.stem}: verified licence without identifier"
        assert lic["source"], f"{path.stem}: verified licence without source"
    else:
        # Unknown/unclear/pending licences must never be treated as permissive.
        assert lic["commercial_use"] != "allowed", "unverified licence marked commercial-allowed"
        assert lic["redistribution"] != "allowed", "unverified licence marked redistributable"


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_canonical_source_present(path: Path) -> None:
    prov = _load(path)["provenance"]
    assert prov["canonical_source_name"]
    assert prov["canonical_source_url"]


@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_external_entries_have_verification_date(path: Path) -> None:
    prov = _load(path)["provenance"]
    if prov["source_type"] in EXTERNAL_SOURCE_TYPES:
        assert prov["date_last_verified"], f"{path.stem}: external entry missing date_last_verified"


# --- Split / leakage / privacy metadata --------------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_split_grouping_present(path: Path) -> None:
    assert _load(path)["split_leakage"]["source_grouping_unit"]


# --- U2/U3 consistency of ontology references --------------------------------
@pytest.mark.parametrize("path", ENTRY_PATHS, ids=_entry_id)
def test_ontology_references_valid(path: Path) -> None:
    refs = _load(path)["task_fit"]["ontology_references"]
    allowed_keys = set(SCHEMA["ontology_reference_keys"])
    for key, ids in refs.items():
        assert key in allowed_keys, f"{path.stem}: unknown ontology-reference key {key!r}"
        invalid = set(ids) - VALID_ONTOLOGY_REFS[key]
        assert not invalid, f"{path.stem}: invalid {key} ids {invalid}"


# --- First-class dependency + AI City honesty --------------------------------
def test_event_evaluation_dependency_present() -> None:
    represented = any(
        "event_evaluation" in _load(p)["identity"]["task_categories"] for p in ENTRY_PATHS
    )
    assert represented, "no entry represents the event-evaluation footage dependency"


def test_aicity_not_confirmed() -> None:
    for path in ENTRY_PATHS:
        if "aicity" not in path.stem:
            continue
        entry = _load(path)
        assert entry["access"]["access_confirmed_by_project"] is False
        assert entry["access"]["status"] in AICITY_UNCONFIRMED_STATES

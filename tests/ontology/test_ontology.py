"""Validation and U2-contract-consistency tests for the label ontology (U3).

Consistency with the frozen U2 enums is derived from the live enum/observation
types, never from a second manually maintained list, so the ontology and the
contracts cannot silently drift apart.
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
ONTOLOGY_PATH = REPO_ROOT / "configs" / "ontology.yaml"
DOC_PATH = REPO_ROOT / "docs" / "ontology.md"

ID_SECTIONS = (
    "helmet_states",
    "signal_states",
    "object_classes",
    "violation_types",
    "observation_types",
)


@pytest.fixture(scope="module")
def ontology() -> dict[str, Any]:
    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))


def _ids(section: list[dict[str, Any]]) -> list[str]:
    return [item["id"] for item in section]


def _helmet(ontology: dict[str, Any], state_id: str) -> dict[str, Any]:
    (entry,) = [h for h in ontology["helmet_states"] if h["id"] == state_id]
    return entry


def _iter_scalars(node: Any) -> Iterator[Any]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_scalars(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_scalars(value)
    else:
        yield node


# --- Structural validity -----------------------------------------------------
def test_yaml_parses(ontology: dict[str, Any]) -> None:
    assert isinstance(ontology, dict)


def test_required_sections_present(ontology: dict[str, Any]) -> None:
    required = {
        "ontology",
        "principles",
        "helmet_states",
        "signal_states",
        "object_classes",
        "violation_types",
        "observation_types",
        "annotation_policy",
    }
    assert required <= set(ontology)


def test_version_present_non_empty(ontology: dict[str, Any]) -> None:
    version = ontology["ontology"]["version"]
    assert isinstance(version, str)
    assert version.strip()


def test_ids_unique_per_section(ontology: dict[str, Any]) -> None:
    for section in ID_SECTIONS:
        ids = _ids(ontology[section])
        assert len(ids) == len(set(ids)), f"duplicate ids in {section}"


# --- Consistency with the frozen U2 contract enums ---------------------------
def test_helmet_states_match_enum(ontology: dict[str, Any]) -> None:
    assert set(_ids(ontology["helmet_states"])) == {s.value for s in HelmetState}


def test_signal_states_match_enum(ontology: dict[str, Any]) -> None:
    assert set(_ids(ontology["signal_states"])) == {s.value for s in SignalState}


def test_object_classes_match_enum(ontology: dict[str, Any]) -> None:
    assert set(_ids(ontology["object_classes"])) == {s.value for s in ObjectClass}


def test_violation_types_match_enum(ontology: dict[str, Any]) -> None:
    assert set(_ids(ontology["violation_types"])) == {s.value for s in ViolationType}


def test_observation_types_match_union(ontology: dict[str, Any]) -> None:
    actual = {v.model_fields["obs_type"].default for v in OBSERVATION_VARIANTS}
    assert set(_ids(ontology["observation_types"])) == actual


# --- Helmet semantics --------------------------------------------------------
def test_helmet_uncertain_abstains(ontology: dict[str, Any]) -> None:
    assert _helmet(ontology, "uncertain")["abstains"] is True


def test_helmet_decided_states_do_not_abstain(ontology: dict[str, Any]) -> None:
    for state_id in ("helmet", "no_helmet", "turban"):
        assert _helmet(ontology, state_id)["abstains"] is False


def test_helmet_turban_not_automatic_no_helmet(ontology: dict[str, Any]) -> None:
    turban = _helmet(ontology, "turban")
    assert turban["automatic_violation_semantics"] == "none"
    assert turban.get("distinct_from_no_helmet") is True


def test_no_helmet_state_auto_confirms_violation(ontology: dict[str, Any]) -> None:
    violation_ids = {v.value for v in ViolationType}
    for state in ontology["helmet_states"]:
        semantics = state["automatic_violation_semantics"]
        assert semantics == "none"
        assert semantics not in violation_ids


# --- No thresholds / no executable logic -------------------------------------
def test_no_numeric_thresholds(ontology: dict[str, Any]) -> None:
    for scalar in _iter_scalars(ontology):
        if isinstance(scalar, bool):
            continue
        assert not isinstance(scalar, int | float), f"numeric scalar not allowed: {scalar!r}"


# --- Internal cross-reference consistency ------------------------------------
def test_observation_consumed_by_are_valid_violations(ontology: dict[str, Any]) -> None:
    violation_ids = {v.value for v in ViolationType}
    for obs in ontology["observation_types"]:
        assert set(obs["consumed_by"]) <= violation_ids


def test_violation_upstream_are_valid_observations(ontology: dict[str, Any]) -> None:
    obs_ids = {o["id"] for o in ontology["observation_types"]}
    for viol in ontology["violation_types"]:
        assert set(viol["upstream_observations"]) <= obs_ids


def test_annotation_policy_flags(ontology: dict[str, Any]) -> None:
    policy = ontology["annotation_policy"]
    assert policy["prefer_abstention_over_guessing"] is True
    assert policy["use_future_frames_to_label_current_frame"] is False
    assert policy["preserve_source_grouping_metadata"] is True


# --- Documentation references the canonical source ---------------------------
def test_doc_references_yaml_and_version(ontology: dict[str, Any]) -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "configs/ontology.yaml" in doc
    assert ontology["ontology"]["version"] in doc

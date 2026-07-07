"""Typed scene-contract loading, round-trip, and deterministic-hash tests (U5).

Validates the authoritative plan's requirements: the example YAML loads into the
typed ``SceneConfig``; round-trips; rejects invalid structural data; and
``scene_config_hash`` is deterministic, formatting/key-order invariant, sensitive
to meaningful change, correctly formatted, and compatible with the
``scene_config_hash`` fields on ``ConfirmedEvent`` / ``EvidenceManifest``.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from trafficpulse.contracts import (
    ConfirmedEvent,
    EvidenceManifest,
    SceneConfig,
    SceneStatus,
    ViolationType,
    scene_config_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _raw() -> dict[str, Any]:
    return yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))


def _scene() -> SceneConfig:
    return SceneConfig.model_validate(_raw())


def _reorder(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _reorder(node[key]) for key in reversed(list(node.keys()))}
    if isinstance(node, list):
        return [_reorder(item) for item in node]
    return node


# --- loading and round-trip --------------------------------------------------
def test_example_yaml_loads_into_contract() -> None:
    scene = _scene()
    assert isinstance(scene, SceneConfig)
    assert scene.scene.status is SceneStatus.DRAFT
    assert len(scene.zones) == 6


def test_roundtrip_json() -> None:
    scene = _scene()
    restored = SceneConfig.model_validate_json(scene.model_dump_json())
    assert restored == scene


# --- invalid structural data is rejected -------------------------------------
def test_out_of_bounds_point_rejected() -> None:
    raw = _raw()
    raw["zones"][0]["polygon"][0] = [999999, 10]
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


def test_zero_length_vector_rejected() -> None:
    raw = _raw()
    raw["legal_directions"][0]["vector"] = {"dx": 0.0, "dy": 0.0}
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


def test_missing_zone_reference_rejected() -> None:
    raw = _raw()
    raw["stop_lines"][0]["zone_ids"] = ["zone-does-not-exist"]
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


def test_bad_matrix_shape_rejected() -> None:
    raw = _raw()
    raw["calibration"]["homography_matrix"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


def test_extra_field_rejected() -> None:
    raw = _raw()
    raw["unexpected_top_level"] = 1
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


def test_invalid_violation_reference_rejected() -> None:
    raw = _raw()
    raw["zones"][0]["applicable_violations"] = ["not_a_violation"]
    with pytest.raises(ValidationError):
        SceneConfig.model_validate(raw)


# --- deterministic hashing ---------------------------------------------------
def test_hash_deterministic_repeated_calls() -> None:
    scene = _scene()
    assert scene_config_hash(scene) == scene_config_hash(scene)


def test_hash_deterministic_across_fresh_instances() -> None:
    assert scene_config_hash(_scene()) == scene_config_hash(_scene())


def test_hash_stable_across_serialization_roundtrip() -> None:
    scene = _scene()
    restored = SceneConfig.model_validate_json(scene.model_dump_json())
    assert scene_config_hash(scene) == scene_config_hash(restored)


def test_hash_ignores_mapping_key_order() -> None:
    base = scene_config_hash(SceneConfig.model_validate(_raw()))
    reordered = scene_config_hash(SceneConfig.model_validate(_reorder(_raw())))
    assert base == reordered


def test_hash_ignores_yaml_formatting() -> None:
    raw = _raw()
    reformatted = yaml.safe_load(yaml.dump(raw, default_flow_style=True, sort_keys=True))
    assert scene_config_hash(SceneConfig.model_validate(raw)) == scene_config_hash(
        SceneConfig.model_validate(reformatted)
    )


def test_hash_changes_on_meaningful_change() -> None:
    base = scene_config_hash(_scene())
    mutated = _raw()
    mutated["frame"]["reference_width"] = mutated["frame"]["reference_width"] + 1
    assert base != scene_config_hash(SceneConfig.model_validate(mutated))


def test_hash_format_is_sha256_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", scene_config_hash(_scene())) is not None


# --- compatibility with U2 event/evidence hash fields ------------------------
def test_hash_compatible_with_confirmed_event() -> None:
    digest = scene_config_hash(_scene())
    event = ConfirmedEvent(
        event_id="e1",
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-synthetic-01",
        start_at=TS,
        trigger_at=TS,
        rule_id="wrong_way",
        scene_config_hash=digest,
        created_at=TS,
    )
    assert event.scene_config_hash == digest


def test_hash_compatible_with_evidence_manifest() -> None:
    digest = scene_config_hash(_scene())
    manifest = EvidenceManifest(
        evidence_package_id="ep1",
        event_id="e1",
        scene_config_hash=digest,
        created_at=TS,
    )
    assert manifest.scene_config_hash == digest

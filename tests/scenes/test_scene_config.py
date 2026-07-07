"""Structural, referential, and U2-consistency tests for scene configuration (U5).

Closed vocabularies and required structure are read from configs/scenes/schema.yaml;
violation/observation/signal references are validated against the live U2
contracts. Declarative data only — no geometry or rule behaviour is exercised.
"""

import copy
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from trafficpulse.contracts.enums import SignalState, ViolationType
from trafficpulse.contracts.observations import OBSERVATION_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "configs" / "scenes" / "schema.yaml"
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"

SCHEMA: dict[str, Any] = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
SCENE: dict[str, Any] = yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))

VIOLATIONS = {v.value for v in ViolationType}
SIGNALS = {s.value for s in SignalState}
OBSERVATIONS = {v.model_fields["obs_type"].default for v in OBSERVATION_VARIANTS}


# --- helpers -----------------------------------------------------------------
def _nested(node: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        node = node[key]
    return node


def _ids(items: list[dict[str, Any]], key: str) -> set[str]:
    return {item[key] for item in items}


def _frame_wh(cfg: dict[str, Any]) -> tuple[int, int]:
    return cfg["frame"]["reference_width"], cfg["frame"]["reference_height"]


def _image_points(cfg: dict[str, Any]) -> Iterator[tuple[str, list[float]]]:
    for zone in cfg["zones"]:
        for pt in zone["polygon"]:
            yield f"zone:{zone['zone_id']}", pt
    for sl in cfg["stop_lines"]:
        yield f"stopline:{sl['stop_line_id']}:a", sl["endpoints"]["a"]
        yield f"stopline:{sl['stop_line_id']}:b", sl["endpoints"]["b"]
    for sg in cfg["signal_groups"]:
        roi = sg["roi"]
        if roi["shape"] == "rectangle":
            br = [roi["x"] + roi["width"], roi["y"] + roi["height"]]
            yield f"roi:{sg['signal_group_id']}:tl", [roi["x"], roi["y"]]
            yield f"roi:{sg['signal_group_id']}:br", br
        else:
            for pt in roi.get("polygon", []):
                yield f"roi:{sg['signal_group_id']}", pt
    for corr in cfg["calibration"].get("correspondences") or []:
        yield "correspondence", corr["image_xy"]


def _all_points_in_bounds(cfg: dict[str, Any]) -> bool:
    w, h = _frame_wh(cfg)
    return all(0 <= pt[0] <= w and 0 <= pt[1] <= h for _, pt in _image_points(cfg))


def _vector_is_zero(vec: dict[str, float]) -> bool:
    return float(vec["dx"]) == 0.0 and float(vec["dy"]) == 0.0


def _vectors_nonzero(cfg: dict[str, Any]) -> bool:
    dirs_ok = all(not _vector_is_zero(d["vector"]) for d in cfg["legal_directions"])
    cross_ok = all(not _vector_is_zero(sl["crossing_direction"]) for sl in cfg["stop_lines"])
    return dirs_ok and cross_ok


def _references_resolve(cfg: dict[str, Any]) -> bool:
    zone_ids = _ids(cfg["zones"], "zone_id")
    sg_ids = _ids(cfg["signal_groups"], "signal_group_id")
    sl_ids = _ids(cfg["stop_lines"], "stop_line_id")
    dir_ids = _ids(cfg["legal_directions"], "direction_id")
    checks: list[bool] = []
    for sl in cfg["stop_lines"]:
        checks.append(sl["signal_group_id"] in sg_ids)
        checks.extend(z in zone_ids for z in sl["zone_ids"])
    for sg in cfg["signal_groups"]:
        checks.extend(s in sl_ids for s in sg["stop_line_ids"])
        checks.extend(z in zone_ids for z in sg["zone_ids"])
    for direction in cfg["legal_directions"]:
        checks.extend(z in zone_ids for z in direction["zone_ids"])
    for sp in cfg["speed_limits"]:
        checks.extend(z in zone_ids for z in sp["zone_ids"])
    for zone in cfg["zones"]:
        if zone.get("legal_direction_id") is not None:
            checks.append(zone["legal_direction_id"] in dir_ids)
        if zone.get("signal_group_id") is not None:
            checks.append(zone["signal_group_id"] in sg_ids)
    return all(checks)


def _row_ok(row: Any) -> bool:
    if not isinstance(row, list) or len(row) != 3:
        return False
    return all((not isinstance(v, bool)) and isinstance(v, int | float) for v in row)


def _calibration_matrix_valid(matrix: Any) -> bool:
    if not isinstance(matrix, list) or len(matrix) != 3:
        return False
    return all(_row_ok(r) for r in matrix)


def _iter_params(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for block in cfg["rule_parameters"]:
        yield from block["parameters"]


def _rule_params_provisional(cfg: dict[str, Any]) -> bool:
    return all(p["status"] in {"unset", "provisional"} for p in _iter_params(cfg))


def _iter_keys(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _iter_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_keys(value)


def _iter_strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_strings(value)
    elif isinstance(node, str):
        yield node


def _copy() -> dict[str, Any]:
    return copy.deepcopy(SCENE)


# --- structural validity -----------------------------------------------------
def test_schema_parses() -> None:
    assert isinstance(SCHEMA, dict)
    assert SCHEMA["meta"]["version"]


def test_scene_parses() -> None:
    assert isinstance(SCENE, dict)


def test_required_sections_present() -> None:
    assert set(SCHEMA["required_sections"]) <= set(SCENE)


def test_scene_identity_fields() -> None:
    for field in SCHEMA["required_fields"]["scene"]:
        assert field in SCENE["scene"], f"missing scene.{field}"


def test_item_required_fields() -> None:
    mapping = {
        "zones": "zone",
        "stop_lines": "stop_line",
        "legal_directions": "legal_direction",
        "signal_groups": "signal_group",
        "speed_limits": "speed_limit",
    }
    for section, spec in mapping.items():
        for item in SCENE[section]:
            for field in SCHEMA["required_fields"][spec]:
                assert field in item, f"{section} item missing {field}"
    for field in SCHEMA["required_fields"]["calibration"]:
        assert field in SCENE["calibration"], f"calibration missing {field}"
    for block in SCENE["rule_parameters"]:
        for field in SCHEMA["required_fields"]["rule_parameter_block"]:
            assert field in block
        for param in block["parameters"]:
            for field in SCHEMA["required_fields"]["rule_parameter"]:
                assert field in param, f"param missing {field}"


# --- coordinate convention and frame -----------------------------------------
def test_frame_dims_positive_int() -> None:
    w, h = _frame_wh(SCENE)
    assert isinstance(w, int) and not isinstance(w, bool) and w > 0
    assert isinstance(h, int) and not isinstance(h, bool) and h > 0


def test_coordinate_convention_frozen() -> None:
    for key, value in SCHEMA["coordinate_convention"].items():
        assert SCENE["frame"][key] == value, f"frame.{key} != frozen {value}"


# --- vocabularies ------------------------------------------------------------
def test_field_vocabularies() -> None:
    for path, vocab in SCHEMA["field_vocabularies"].items():
        section, field = path.split(".", 1)
        allowed = SCHEMA["vocabularies"][vocab]
        assert _nested(SCENE[section], field) in allowed, f"{path} invalid"


def test_item_field_vocabularies() -> None:
    for path, vocab in SCHEMA["item_field_vocabularies"].items():
        section, field = path.split(".", 1)
        allowed = SCHEMA["vocabularies"][vocab]
        for item in SCENE[section]:
            assert _nested(item, field) in allowed, f"{section}.{field} invalid"


def test_scene_status_in_vocab() -> None:
    assert SCENE["scene"]["status"] in SCHEMA["vocabularies"]["scene_status"]


# --- geometry (structure only) -----------------------------------------------
def test_polygons_have_min_points() -> None:
    for zone in SCENE["zones"]:
        assert len(zone["polygon"]) >= 3, f"{zone['zone_id']} polygon too small"


def test_all_image_points_within_bounds() -> None:
    assert _all_points_in_bounds(SCENE)


def test_stop_line_endpoints_within_bounds() -> None:
    w, h = _frame_wh(SCENE)
    for sl in SCENE["stop_lines"]:
        for key in ("a", "b"):
            x, y = sl["endpoints"][key]
            assert 0 <= x <= w and 0 <= y <= h


def test_roi_geometry_valid() -> None:
    w, h = _frame_wh(SCENE)
    for sg in SCENE["signal_groups"]:
        roi = sg["roi"]
        if roi["shape"] == "rectangle":
            assert roi["width"] > 0 and roi["height"] > 0
            assert roi["x"] >= 0 and roi["y"] >= 0
            assert roi["x"] + roi["width"] <= w
            assert roi["y"] + roi["height"] <= h


def test_legal_vectors_nonzero() -> None:
    assert _vectors_nonzero(SCENE)


# --- ids and references ------------------------------------------------------
def test_ids_unique_per_category() -> None:
    checks = {
        "zones": "zone_id",
        "stop_lines": "stop_line_id",
        "legal_directions": "direction_id",
        "signal_groups": "signal_group_id",
        "speed_limits": "speed_limit_id",
    }
    for section, key in checks.items():
        ids = [item[key] for item in SCENE[section]]
        assert len(ids) == len(set(ids)), f"duplicate ids in {section}"
    vtypes = [b["violation_type"] for b in SCENE["rule_parameters"]]
    assert len(vtypes) == len(set(vtypes))


def test_references_resolve() -> None:
    assert _references_resolve(SCENE)


# --- U2/U3 consistency -------------------------------------------------------
def test_violation_references_valid() -> None:
    for block in SCENE["rule_parameters"]:
        assert block["violation_type"] in VIOLATIONS
    for zone in SCENE["zones"]:
        assert set(zone.get("applicable_violations", [])) <= VIOLATIONS


def test_observation_references_valid() -> None:
    for zone in SCENE["zones"]:
        assert set(zone.get("observation_consumers", [])) <= OBSERVATIONS


def test_signal_state_references_valid() -> None:
    for sg in SCENE["signal_groups"]:
        assert set(sg.get("expected_states", [])) <= SIGNALS


def test_speed_limit_units_supported() -> None:
    for sl in SCENE["speed_limits"]:
        assert sl["unit"] in SCHEMA["vocabularies"]["speed_unit"]


# --- calibration honesty -----------------------------------------------------
def test_calibration_matrix_3x3() -> None:
    assert _calibration_matrix_valid(SCENE["calibration"]["homography_matrix"])


def test_calibration_status_valid() -> None:
    cal = SCENE["calibration"]
    assert cal["status"] in SCHEMA["vocabularies"]["calibration_status"]


def test_synthetic_calibration_not_verified() -> None:
    cal = SCENE["calibration"]
    assert cal["verification_status"] == "unverified"
    assert cal["status"] != "validated"
    assert cal["quality_metrics"]["reprojection_rmse_px"] is None


def test_example_scene_not_validated() -> None:
    assert SCENE["scene"]["status"] != "validated"


# --- provisional rule parameters ---------------------------------------------
def test_rule_params_provisional_only() -> None:
    assert _rule_params_provisional(SCENE)


def test_rule_param_units_explicit() -> None:
    allowed = set(SCHEMA["vocabularies"]["parameter_unit"])
    for param in _iter_params(SCENE):
        assert param["unit"] in allowed, f"{param['id']} bad unit"
        if param["status"] == "unset":
            assert param["value"] is None
        else:
            assert param["value"] is not None


def test_provisional_params_have_note() -> None:
    for param in _iter_params(SCENE):
        if param["status"] == "provisional":
            assert param["note"], f"{param['id']} provisional without note"


# --- security / privacy ------------------------------------------------------
def test_no_forbidden_keys() -> None:
    forbidden = {k.lower() for k in SCHEMA["forbidden_keys"]}
    present = {k.lower() for k in _iter_keys(SCENE)}
    assert not (present & forbidden), present & forbidden


def test_no_secret_value_patterns() -> None:
    patterns = [re.compile(p) for p in SCHEMA["secret_value_patterns"]]
    for text in _iter_strings(SCENE):
        for pat in patterns:
            assert not pat.search(text), f"secret-like value: {text!r}"


def test_no_behavioral_expressions() -> None:
    for text in _iter_strings(SCENE):
        for sub in SCHEMA["forbidden_value_substrings"]:
            assert sub not in text, f"expression-like value: {text!r}"


# --- negative tests (mutated copies) -----------------------------------------
def test_neg_out_of_bounds_point() -> None:
    cfg = _copy()
    cfg["zones"][0]["polygon"][0] = [999999, 10]
    assert not _all_points_in_bounds(cfg)


def test_neg_zero_length_vector() -> None:
    cfg = _copy()
    cfg["legal_directions"][0]["vector"] = {"dx": 0.0, "dy": 0.0}
    assert not _vectors_nonzero(cfg)


def test_neg_missing_zone_reference() -> None:
    cfg = _copy()
    cfg["stop_lines"][0]["zone_ids"] = ["zone-does-not-exist"]
    assert not _references_resolve(cfg)


def test_neg_missing_signal_group_reference() -> None:
    cfg = _copy()
    cfg["stop_lines"][0]["signal_group_id"] = "sg-does-not-exist"
    assert not _references_resolve(cfg)


def test_neg_bad_calibration_matrix_shape() -> None:
    cfg = _copy()
    cfg["calibration"]["homography_matrix"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert not _calibration_matrix_valid(cfg["calibration"]["homography_matrix"])


def test_neg_invalid_scene_status() -> None:
    cfg = _copy()
    cfg["scene"]["status"] = "definitely-not-a-status"
    assert cfg["scene"]["status"] not in SCHEMA["vocabularies"]["scene_status"]


def test_neg_param_marked_validated_flagged() -> None:
    cfg = _copy()
    cfg["rule_parameters"][0]["parameters"][0]["status"] = "validated"
    assert not _rule_params_provisional(cfg)

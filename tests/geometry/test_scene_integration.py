"""Integration tests: geometry primitives consuming the U5 example SceneConfig.

These prove the geometry layer answers real geometric questions against actual
configured scene data (a zone polygon, a legal direction, a stop line) while
embedding no behavioral rule threshold and importing no ML/CV framework or the
contract layer itself. No violation decision is made here -- only geometric
facts are computed.
"""

import ast
from pathlib import Path

import yaml

import trafficpulse.geometry as geometry
from trafficpulse.contracts import SceneConfig, Zone
from trafficpulse.geometry import (
    angle_between_degrees,
    direction,
    point_in_polygon,
    segments_intersect,
    stop_line_crossing,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"


def _scene() -> SceneConfig:
    return SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))


def _zone(scene: SceneConfig, zone_id: str) -> Zone:
    return next(z for z in scene.zones if z.zone_id == zone_id)


# --- 31: the example SceneConfig loads --------------------------------------
def test_example_scene_loads() -> None:
    scene = _scene()
    assert isinstance(scene, SceneConfig)
    assert len(scene.zones) == 6


# --- 32: a configured polygon drives point-in-polygon ------------------------
def test_configured_zone_polygon_membership() -> None:
    scene = _scene()
    lane = _zone(scene, "zone-lane-north")
    # The zone polygon is already a tuple of (x, y) tuples -> consumed directly.
    assert point_in_polygon((960.0, 800.0), lane.polygon) is True
    assert point_in_polygon((500.0, 500.0), lane.polygon) is False


# --- 33: a configured legal direction drives angular deviation ---------------
def test_configured_legal_direction_deviation() -> None:
    scene = _scene()
    north = next(d for d in scene.legal_directions if d.direction_id == "dir-north")
    legal_vec = (north.vector.dx, north.vector.dy)  # bridge DirectionVector -> tuple

    # Movement up the image (decreasing y) aligns with legal "north".
    aligned = direction((960.0, 1000.0), (960.0, 700.0))
    reversed_ = direction((960.0, 700.0), (960.0, 1000.0))

    # These are geometric facts only; no wrong-way threshold is applied.
    assert angle_between_degrees(aligned, legal_vec) == 0.0
    assert angle_between_degrees(reversed_, legal_vec) == 180.0


def test_configured_approach_direction_small_deviation() -> None:
    scene = _scene()
    approach = next(d for d in scene.legal_directions if d.direction_id == "dir-approach")
    legal_vec = (approach.vector.dx, approach.vector.dy)
    straight_up = direction((960.0, 1000.0), (960.0, 700.0))
    dev = angle_between_degrees(straight_up, legal_vec)
    assert 0.0 < dev < 10.0  # a fact about the angle, not a legality verdict


# --- 34: a configured stop line drives segment intersection / crossing -------
def test_configured_stop_line_crossing() -> None:
    scene = _scene()
    stop = next(s for s in scene.stop_lines if s.stop_line_id == "stopline-001")
    a = stop.endpoints.a
    b = stop.endpoints.b

    # A track moving up through y=700 crosses the configured stop line.
    crossing_move = ((960.0, 720.0), (960.0, 680.0))
    assert segments_intersect(*crossing_move, a, b) is True
    fact = stop_line_crossing(*crossing_move, a, b)
    assert fact.side_changed is True
    assert fact.intersects_segment is True

    # A track that stays below the line does not cross it.
    below_move = ((960.0, 720.0), (960.0, 710.0))
    assert segments_intersect(*below_move, a, b) is False
    assert stop_line_crossing(*below_move, a, b).side_changed is False


# --- 35: geometry embeds no rule threshold and no forbidden dependency -------
def _geometry_sources() -> list[Path]:
    pkg_dir = Path(geometry.__file__).resolve().parent
    return sorted(pkg_dir.glob("*.py"))


def test_geometry_embeds_no_scene_rule_parameter_ids() -> None:
    # Every rule-parameter id configured in the scene must be ABSENT from the
    # geometry source: geometry consumes scene data but hard-codes no behavioral
    # parameter the rule/temporal layers own.
    scene = _scene()
    param_ids = {p.id for block in scene.rule_parameters for p in block.parameters}
    assert param_ids  # guard: the scene actually declares rule parameters
    combined = "\n".join(p.read_text(encoding="utf-8") for p in _geometry_sources())
    leaked = {pid for pid in param_ids if pid in combined}
    assert not leaked, f"geometry source references rule-parameter ids: {sorted(leaked)}"


def test_geometry_imports_are_stdlib_or_intrapackage_only() -> None:
    # No ML/CV/GIS framework and no contract-layer coupling in geometry source.
    allowed_top_level = {"math", "collections", "typing", "__future__"}
    for path in _geometry_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_top_level, f"{path.name}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # relative intra-package import (e.g. .vectors)
                assert node.module is not None
                top = node.module.split(".")[0]
                assert top in allowed_top_level, f"{path.name}: from {node.module}"


def test_geometry_does_not_import_contracts() -> None:
    combined = "\n".join(p.read_text(encoding="utf-8") for p in _geometry_sources())
    assert "trafficpulse.contracts" not in combined
    assert "import pydantic" not in combined

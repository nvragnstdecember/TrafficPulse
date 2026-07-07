"""JSON Schema export: determinism and completeness."""

import json
from pathlib import Path

from trafficpulse.contracts.observations import OBSERVATION_VARIANTS
from trafficpulse.contracts.schema_export import (
    TOP_LEVEL_CONTRACTS,
    export_schemas,
)

EXPECTED_OBS_TYPES = {
    "in_zone",
    "signal_state",
    "heading_vs_lane",
    "stationary",
    "rider_count",
    "helmet_state",
    "speed",
}


def _expected_filenames() -> set[str]:
    names = {f"{m.__name__}.schema.json" for m in TOP_LEVEL_CONTRACTS}
    names |= {f"{m.__name__}.schema.json" for m in OBSERVATION_VARIANTS}
    names.add("Observation.schema.json")
    return names


def test_export_is_byte_identical_across_runs(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    paths_a = export_schemas(dir_a)
    paths_b = export_schemas(dir_b)

    names_a = {p.name for p in paths_a}
    names_b = {p.name for p in paths_b}
    assert names_a == names_b

    for name in names_a:
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_export_includes_all_public_schemas(tmp_path: Path) -> None:
    paths = export_schemas(tmp_path)
    produced = {p.name for p in paths}
    assert produced == _expected_filenames()
    assert len(produced) == len(TOP_LEVEL_CONTRACTS) + len(OBSERVATION_VARIANTS) + 1


def test_observation_union_schema_covers_all_variants(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    union_text = (tmp_path / "Observation.schema.json").read_text(encoding="utf-8")
    schema = json.loads(union_text)
    discriminator = schema.get("discriminator", {})
    mapping = discriminator.get("mapping", {})
    assert set(mapping) == EXPECTED_OBS_TYPES


def test_exported_schema_is_sorted_and_trailing_newline(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    raw = (tmp_path / "Detection.schema.json").read_bytes()
    assert raw.endswith(b"\n")
    # sort_keys=True guarantees a canonical, re-serializable form.
    reparsed = json.loads(raw)
    canonical = json.dumps(reparsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert raw == canonical.encode("utf-8")

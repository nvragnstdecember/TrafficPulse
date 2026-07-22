"""AppConfig + scene loading (H7A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _app_helpers import EXAMPLE_SCENE_PATH
from pydantic import ValidationError

from trafficpulse.app import AppConfig
from trafficpulse.app.config import load_scene
from trafficpulse.contracts import SceneConfig


def test_derived_directories(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=tmp_path)
    assert config.videos_dir == tmp_path / "videos"
    assert config.runs_dir == tmp_path / "runs"


def test_defaults(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=tmp_path)
    assert (config.host, config.port) == ("127.0.0.1", 8000)
    assert ".mp4" in config.allowed_extensions
    assert config.scene_path is None
    assert config.default_rules == ()
    assert config.inference is None


def test_config_is_frozen(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=tmp_path)
    with pytest.raises(ValidationError):
        config.port = 9000  # type: ignore[misc]


def test_extension_normalisation(tmp_path: Path) -> None:
    config = AppConfig(storage_dir=tmp_path, allowed_extensions={"MP4", ".AVI", "mkv"})
    assert config.allowed_extensions == {".mp4", ".avi", ".mkv"}
    assert config.is_supported_extension(".MP4")
    assert config.is_supported_extension("mkv")
    assert not config.is_supported_extension(".txt")


def test_port_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AppConfig(storage_dir=tmp_path, port=0)
    with pytest.raises(ValidationError):
        AppConfig(storage_dir=tmp_path, port=70000)


def test_extension_validator_rejects_non_collection(tmp_path: Path) -> None:
    # A non-collection passes through the normaliser unchanged for pydantic to
    # reject, rather than being silently coerced.
    with pytest.raises(ValidationError):
        AppConfig(storage_dir=tmp_path, allowed_extensions=123)  # type: ignore[arg-type]


# --- from_env ------------------------------------------------------------------
def test_from_env_defaults() -> None:
    config = AppConfig.from_env({})
    assert config.storage_dir == Path("trafficpulse-data")
    assert config.scene_path is None
    assert config.port == 8000


def test_from_env_overrides() -> None:
    config = AppConfig.from_env(
        {
            "TRAFFICPULSE_APP_STORAGE": "/data/tp",
            "TRAFFICPULSE_APP_SCENE": "scene.yaml",
            "TRAFFICPULSE_APP_HOST": "0.0.0.0",
            "TRAFFICPULSE_APP_PORT": "9100",
            "TRAFFICPULSE_APP_MAX_UPLOAD_BYTES": "1024",
        }
    )
    assert config.storage_dir == Path("/data/tp")
    assert config.scene_path == Path("scene.yaml")
    assert (config.host, config.port) == ("0.0.0.0", 9100)
    assert config.max_upload_bytes == 1024


def test_from_env_rejects_bad_port() -> None:
    with pytest.raises(ValueError):
        AppConfig.from_env({"TRAFFICPULSE_APP_PORT": "not-a-number"})


# --- scene loading -------------------------------------------------------------
def test_load_scene_from_yaml() -> None:
    scene = load_scene(EXAMPLE_SCENE_PATH)
    assert isinstance(scene, SceneConfig)


def test_load_scene_from_json(tmp_path: Path) -> None:
    scene = load_scene(EXAMPLE_SCENE_PATH)
    assert isinstance(scene, SceneConfig)
    json_path = tmp_path / "scene.json"
    json_path.write_text(json.dumps(scene.model_dump(mode="json")), encoding="utf-8")
    reloaded = load_scene(json_path)
    assert isinstance(reloaded, SceneConfig)
    assert reloaded == scene

"""The production launcher's composition (H16).

``serve.py`` is the documented production entrypoint and was previously untested:
a wrong key in ``LABEL_MAP`` silently disables a whole violation class, and nothing
would have caught it. These tests build the configuration only -- constructing
these configs imports no ML framework and loads no checkpoint.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from trafficpulse.contracts.enums import ObjectClass


@pytest.fixture
def serve_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    """Import ``serve`` against a temporary storage root, freshly each time."""

    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(tmp_path))
    module = importlib.import_module("serve")
    yield importlib.reload(module)


def test_the_launcher_configures_a_real_inference_backend(serve_module: Any) -> None:
    """The gap this module exists to close: `asgi:app` cannot process video."""

    config = serve_module.build_config()
    assert config.inference is not None
    assert config.helmet_classifier is not None
    assert config.default_rules, "a launcher with no rules can process nothing"


def test_checkpoints_are_resolved_offline(serve_module: Any) -> None:
    """No network access, and no weights vendored (ADR-001)."""

    config = serve_module.build_config()
    assert config.inference.local_files_only is True
    assert config.helmet_classifier.local_files_only is True


def test_the_label_map_covers_every_class_the_rules_need(serve_module: Any) -> None:
    """A wrong key here silently disables a violation class.

    ``motorbike`` is the VOC-style spelling this COCO-80 checkpoint actually uses;
    the intuitive ``motorcycle`` would map nothing, and every motorcycle-perception
    rule (no-helmet, triple-riding) would quietly find no vehicles.
    """

    label_map = serve_module.LABEL_MAP
    assert label_map["motorbike"] is ObjectClass.MOTORCYCLE
    assert label_map["person"] is ObjectClass.PERSON
    # The motorcycle rules need exactly these two present.
    assert {ObjectClass.MOTORCYCLE, ObjectClass.PERSON} <= set(label_map.values())
    assert "motorcycle" not in label_map, "this checkpoint does not use that label"


def test_deployment_knobs_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, serve_module: Any, tmp_path: Path
) -> None:
    """H16: the launcher starts from `AppConfig.from_env()`.

    Before this, every deployment value was a literal in the file and an operator
    could not change the storage root or the port without editing source.
    """

    storage = tmp_path / "elsewhere"
    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(storage))
    monkeypatch.setenv("TRAFFICPULSE_APP_PORT", "9123")
    monkeypatch.setenv("TRAFFICPULSE_APP_LOG_LEVEL", "debug")
    monkeypatch.setenv("TRAFFICPULSE_APP_CORS_ORIGINS", "https://example.org")
    monkeypatch.setenv("TRAFFICPULSE_APP_MAX_UPLOAD_BYTES", "1024")

    config = serve_module.build_config()
    assert config.storage_dir == storage
    assert config.port == 9123
    assert config.log_level == "DEBUG"
    assert config.cors_allow_origins == ("https://example.org",)
    assert config.max_upload_bytes == 1024
    # ... while the model composition still comes from code.
    assert config.inference is not None


def test_an_operator_supplied_scene_wins_over_the_shipped_example(
    monkeypatch: pytest.MonkeyPatch, serve_module: Any, tmp_path: Path
) -> None:
    scene = tmp_path / "my-camera.yaml"
    scene.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRAFFICPULSE_APP_SCENE", str(scene))
    assert serve_module.build_config().scene_path == scene


def test_the_shipped_example_scene_is_the_fallback(serve_module: Any) -> None:
    """A fresh deployment is demonstrable without setting anything."""

    config = serve_module.build_config()
    assert config.scene_path == serve_module.DEFAULT_SCENE_PATH
    assert config.scene_path.is_file()


def test_checkpoint_overrides_are_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator who has reviewed a different artifact's licence can use it."""

    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(tmp_path))
    monkeypatch.setenv("TRAFFICPULSE_APP_DETECTOR_CHECKPOINT", "local/other-rtdetr")
    monkeypatch.setenv("TRAFFICPULSE_APP_DEVICE", "cpu")
    module = importlib.reload(importlib.import_module("serve"))

    config = module.build_config()
    assert config.inference.checkpoint == "local/other-rtdetr"
    assert config.inference.device == "cpu"

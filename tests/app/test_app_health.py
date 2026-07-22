"""Health endpoint + engine-readiness reporting (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import StubEngineProvider, make_client, make_config

from trafficpulse import __version__
from trafficpulse.app import create_app
from trafficpulse.app.config import AppConfig


def test_health_reports_ok_and_version(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "version": __version__, "engine": "ready"}


def test_health_reports_engine_readiness_from_provider(tmp_path: Path) -> None:
    client = make_client(tmp_path, provider=StubEngineProvider(readiness="degraded"))
    assert client.get("/api/health").json()["engine"] == "degraded"


def test_default_provider_reports_unconfigured_without_inference(tmp_path: Path) -> None:
    # No engine_provider injected -> the production RealEngineProvider, which
    # honestly reports 'unconfigured' when no inference backend is set.
    from fastapi.testclient import TestClient

    app = create_app(AppConfig(storage_dir=tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["engine"] == "unconfigured"


def test_make_config_wires_example_scene(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.scene_path is not None and config.scene_path.exists()

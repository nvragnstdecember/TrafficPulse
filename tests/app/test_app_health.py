"""Health endpoint + engine/repository readiness reporting (H7A, extended in H16)."""

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
    # The three original fields are unchanged, so pre-H16 clients keep working.
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["engine"] == "ready"


def test_health_distinguishes_liveness_from_readiness(tmp_path: Path) -> None:
    """H16: 'the process is alive' and 'it can do its job' are separate answers."""

    client = make_client(tmp_path)
    body = client.get("/api/health").json()
    assert body == {
        "status": "ok",
        "version": __version__,
        "engine": "ready",
        "repository": "ready",
        "inference_available": True,
        "scene_configured": True,
    }


def test_health_reports_unavailable_inference_without_a_backend(tmp_path: Path) -> None:
    """The default env-configured server: serving, but unable to process."""

    from fastapi.testclient import TestClient

    app = create_app(AppConfig(storage_dir=tmp_path))
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"  # alive
    assert body["inference_available"] is False  # but not able to work
    assert body["scene_configured"] is False
    assert body["repository"] == "ready"


def test_health_reports_an_unwritable_repository(tmp_path: Path) -> None:
    """A read-only repository is reported, not turned into a failed health check."""

    from trafficpulse.app.routers.health import repository_status

    blocked = tmp_path / "file-not-a-directory"
    blocked.write_bytes(b"")
    assert repository_status(blocked) == "unavailable"
    assert repository_status(tmp_path / "fresh") == "ready"


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

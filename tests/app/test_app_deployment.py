"""Deployment-readiness wiring: CORS + SPA static serving (H8).

Both are opt-in and additive: the default app carries no CORS surface and serves
only the JSON API, so these tests assert the *enabled* behaviour explicitly while
the rest of the suite continues to exercise the default posture.
"""

from __future__ import annotations

from pathlib import Path

from _app_helpers import StubEngineProvider, make_config
from fastapi.testclient import TestClient

from trafficpulse.app import AppConfig, SynchronousJobExecutor, create_app


def _client(config: AppConfig) -> TestClient:
    return TestClient(
        create_app(
            config,
            engine_provider=StubEngineProvider(),
            executor=SynchronousJobExecutor(),
        ),
        raise_server_exceptions=False,
    )


# --- CORS ----------------------------------------------------------------------
def test_no_cors_headers_by_default(tmp_path: Path) -> None:
    client = _client(make_config(tmp_path))
    response = client.get("/api/health", headers={"Origin": "http://example.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_cors_allows_a_configured_origin(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(
        update={"cors_allow_origins": ("http://localhost:5173",)}
    )
    client = _client(config)
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_origins_parsed_from_env() -> None:
    config = AppConfig.from_env(
        {
            "TRAFFICPULSE_APP_STORAGE": "data",
            "TRAFFICPULSE_APP_CORS_ORIGINS": "http://a.test, http://b.test ,",
        }
    )
    assert config.cors_allow_origins == ("http://a.test", "http://b.test")


# --- SPA static serving --------------------------------------------------------
def _write_spa(root: Path) -> Path:
    static = root / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>TrafficPulse</title>", "utf-8")
    (static / "assets" / "app.js").write_text("console.log('app')", "utf-8")
    return static


def test_api_only_by_default_serves_no_spa(tmp_path: Path) -> None:
    client = _client(make_config(tmp_path))
    # With no static dir, the root path is unrouted.
    assert client.get("/").status_code == 404


def test_static_dir_serves_the_spa_and_falls_back_for_client_routes(tmp_path: Path) -> None:
    static = _write_spa(tmp_path)
    config = make_config(tmp_path).model_copy(update={"static_dir": static})
    client = _client(config)

    index = client.get("/")
    assert index.status_code == 200
    assert "TrafficPulse" in index.text

    # A real built asset is served as-is.
    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    # A client-side route (no such file) falls back to the SPA shell.
    deep_link = client.get("/videos")
    assert deep_link.status_code == 200
    assert "TrafficPulse" in deep_link.text


def test_static_serving_never_shadows_the_api(tmp_path: Path) -> None:
    static = _write_spa(tmp_path)
    config = make_config(tmp_path).model_copy(update={"static_dir": static})
    client = _client(config)
    # /api still returns JSON, not the SPA shell.
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_static_dir_parsed_from_env() -> None:
    config = AppConfig.from_env(
        {"TRAFFICPULSE_APP_STORAGE": "data", "TRAFFICPULSE_APP_STATIC_DIR": "frontend/dist"}
    )
    assert config.static_dir == Path("frontend/dist")


def test_missing_static_dir_is_ignored(tmp_path: Path) -> None:
    # A configured-but-absent directory does not break startup; the API still serves.
    config = make_config(tmp_path).model_copy(update={"static_dir": tmp_path / "nope"})
    client = _client(config)
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404

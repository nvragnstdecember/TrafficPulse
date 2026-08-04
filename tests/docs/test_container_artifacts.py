"""The container build artifacts (H16).

Validates the Dockerfile / compose.yaml / .dockerignore **statically**. A real
`docker build` needs a daemon and several minutes of network I/O, which CI here
deliberately does not have; these assertions catch the failures that actually
happen -- a stale entrypoint, storage baked into an image layer, weights vendored
against ADR-001 -- without either.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "compose.yaml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_the_container_files_exist() -> None:
    for path in (DOCKERFILE, COMPOSE, DOCKERIGNORE):
        assert path.is_file(), f"{path.name} is missing"


def test_the_build_is_multi_stage_with_a_frontend_stage() -> None:
    text = _dockerfile()
    assert "AS frontend" in text
    assert "npm ci" in text and "npm run build" in text
    # The built SPA is copied into the runtime, not rebuilt there.
    assert "--from=frontend" in text


def test_the_entrypoint_is_the_documented_production_launcher() -> None:
    """`asgi:app` has no inference backend; shipping it would be the H16 defect."""

    text = _dockerfile()
    assert "serve:app" in text
    assert "trafficpulse.app.asgi" not in text


def test_storage_lives_on_a_volume_not_in_a_layer() -> None:
    text = _dockerfile()
    assert "TRAFFICPULSE_APP_STORAGE=/data" in text
    assert 'VOLUME ["/data"]' in text
    assert "trafficpulse-data:/data" in COMPOSE.read_text(encoding="utf-8")


def test_the_spa_is_served_by_the_api() -> None:
    assert "TRAFFICPULSE_APP_STATIC_DIR=/app/frontend/dist" in _dockerfile()


def test_no_model_weights_are_baked_into_the_image() -> None:
    """ADR-001: checkpoint acquisition is an operator decision, per artifact."""

    text = _dockerfile()
    assert "huggingface-cli download" not in text
    assert "from_pretrained" not in text
    # The cache is a mount point, and compose mounts it read-only.
    assert "HF_HOME=/models" in text
    assert "/models:ro" in COMPOSE.read_text(encoding="utf-8")


def test_the_image_installs_the_extras_the_app_needs() -> None:
    """`api` for the HTTP layer, `overlay` for evidence rendering."""

    text = _dockerfile()
    assert "INSTALL_EXTRAS=api,overlay" in text
    assert "${INSTALL_EXTRAS}" in text


def test_the_container_runs_unprivileged() -> None:
    text = _dockerfile()
    assert "useradd" in text
    assert "USER trafficpulse" in text


def test_a_healthcheck_is_declared() -> None:
    text = _dockerfile()
    assert "HEALTHCHECK" in text
    assert "/api/health" in text


def test_the_log_level_is_configurable_in_the_image() -> None:
    assert "TRAFFICPULSE_APP_LOG_LEVEL" in _dockerfile()
    assert "TRAFFICPULSE_APP_LOG_LEVEL" in COMPOSE.read_text(encoding="utf-8")


def test_repository_state_is_excluded_from_the_build_context() -> None:
    """State belongs on the volume; shipping it would bloat and leak the image."""

    ignored = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for entry in ("trafficpulse-data", "runs", ".venv", "frontend/node_modules", "tests"):
        assert entry in ignored, f"{entry} should not enter the build context"


def test_the_python_base_satisfies_the_declared_floor() -> None:
    """requires-python >= 3.11; the image must not undercut it."""

    text = _dockerfile()
    assert "FROM python:3." in text
    version = text.split("FROM python:3.")[1].split("-")[0]
    assert int(version) >= 11

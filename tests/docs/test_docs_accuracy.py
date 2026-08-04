"""Documented facts must match the repository (H16).

The H16 investigation found `README.md` claiming three different backend test
counts -- 1840 in one place, 2016 in another, against an actual 2262 -- and
describing the project as complete through H8, eight milestones after the fact.
Stale documentation is not cosmetic: it erodes trust in every other claim the
repository makes.

These tests keep the load-bearing numbers honest by deriving them, so the docs
cannot drift again without CI saying so.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DEPLOYMENT = REPO_ROOT / "docs" / "deployment.md"

#: How far a documented count may drift before it is misleading. A handful of
#: tests added since the last docs pass is normal; hundreds is the drift this
#: guard exists to catch.
COUNT_TOLERANCE = 40


def _collected_backend_tests() -> int:
    """The real number of collected backend tests."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(\d+) tests collected", result.stdout)
    if match is None:  # pragma: no cover - collection failure is its own signal
        pytest.skip("could not collect the backend suite")
    return int(match.group(1))


def _documented_counts(text: str, unit: str) -> list[int]:
    """Every ``N passing <unit> tests`` / ``currently N passing`` figure."""

    counts = [int(m) for m in re.findall(rf"([\d,]+) passing {unit} tests", text.replace(",", ""))]
    if unit == "backend":
        counts += [int(m) for m in re.findall(r"currently ([\d,]+) passing", text.replace(",", ""))]
    return counts


def test_the_readme_states_one_backend_test_count() -> None:
    """It previously stated two different ones."""

    counts = _documented_counts(README.read_text(encoding="utf-8"), "backend")
    assert counts, "README should state a backend test count"
    assert len(set(counts)) == 1, f"README states conflicting backend counts: {counts}"


def test_the_documented_backend_count_is_current() -> None:
    documented = _documented_counts(README.read_text(encoding="utf-8"), "backend")[0]
    actual = _collected_backend_tests()
    assert abs(documented - actual) <= COUNT_TOLERANCE, (
        f"README claims {documented} backend tests; the suite collects {actual}"
    )


def test_the_readme_records_the_current_milestone() -> None:
    """The README described the project as complete through H8 for eight milestones."""

    text = README.read_text(encoding="utf-8")
    assert "H16" in text, "the README should record the latest completed milestone"


def test_the_deployment_guide_names_the_production_entrypoint() -> None:
    """`asgi:app` cannot process video; documenting it as the deployment path was the bug."""

    text = DEPLOYMENT.read_text(encoding="utf-8")
    assert "serve:app" in text
    assert "serve.py" in text


def test_the_deployment_guide_documents_every_environment_variable() -> None:
    """A knob nobody can discover is a knob that does not exist."""

    from trafficpulse.app.config import AppConfig

    text = DEPLOYMENT.read_text(encoding="utf-8")
    recognised = {
        "TRAFFICPULSE_APP_STORAGE",
        "TRAFFICPULSE_APP_SCENE",
        "TRAFFICPULSE_APP_HOST",
        "TRAFFICPULSE_APP_PORT",
        "TRAFFICPULSE_APP_MAX_UPLOAD_BYTES",
        "TRAFFICPULSE_APP_CORS_ORIGINS",
        "TRAFFICPULSE_APP_STATIC_DIR",
        "TRAFFICPULSE_APP_LOG_LEVEL",
    }
    for name in recognised:
        assert name in text, f"{name} is unread by the deployment guide"
    # And each one is actually honoured by the config loader.
    assert AppConfig.from_env({"TRAFFICPULSE_APP_LOG_LEVEL": "DEBUG"}).log_level == "DEBUG"


def test_the_deployment_guide_documents_docker() -> None:
    text = DEPLOYMENT.read_text(encoding="utf-8")
    assert "docker compose" in text
    assert "trafficpulse-data" in text


def test_the_version_is_consistent_across_the_project() -> None:
    """The backend and the SPA are released together."""

    import json

    from trafficpulse import __version__

    package_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text("utf-8"))
    assert package_json["version"] == __version__

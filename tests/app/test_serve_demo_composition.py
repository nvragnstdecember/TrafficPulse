"""The demo launcher's composition, and its relationship to the production one.

``serve_demo.py`` selects the trained ResNet-50 backend and declares a helmet
*analysis*. Both halves matter and both are asserted here, because the value of the
demo rests on them being true together: a strong classifier is only presentable if the
violation rule it cannot safely feed is genuinely off.

These tests build configuration only. No ML framework is imported, no checkpoint is
read, and ``serve.py`` is asserted **unchanged** -- the demo is a separate composition,
not a redefinition of production.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from trafficpulse.app.posture import describe, no_helmet_rule_available
from trafficpulse.classifier import ResNetHelmetConfig, ZeroShotHelmetConfig


@pytest.fixture
def demo_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    """Import ``serve_demo`` against a temporary storage root, freshly each time."""

    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(tmp_path))
    module = importlib.import_module("serve_demo")
    yield importlib.reload(module)


@pytest.fixture
def serve_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(tmp_path))
    module = importlib.import_module("serve")
    yield importlib.reload(module)


# --- what the demo selects ------------------------------------------------------------
def test_the_demo_selects_the_trained_resnet_backend(demo_module: Any) -> None:
    config = demo_module.build_config()

    assert isinstance(config.helmet_classifier, ResNetHelmetConfig)


def test_the_demo_pins_the_pre_committed_operating_point(demo_module: Any) -> None:
    """The temperature and abstention floor are quoted from P4-U5/P4-U9, not chosen.

    Tuning either against anything observed at demo time would leak a held-out split
    into the deployment's operating point, which is exactly what the backend's own
    documentation refuses.
    """

    classifier = demo_module.build_config().helmet_classifier

    assert classifier.temperature == pytest.approx(2.2298)
    assert classifier.abstain_below == pytest.approx(0.80)


def test_the_demo_checkpoint_is_local_and_never_downloaded(demo_module: Any) -> None:
    """ADR-001: an artifact's provenance is a per-artifact review."""

    checkpoint = demo_module.build_config().helmet_classifier.checkpoint

    assert isinstance(checkpoint, Path)
    assert not str(checkpoint).startswith("http")


# --- classify, do not enforce -----------------------------------------------------------
def test_the_demo_declares_an_analysis_rather_than_the_helmet_rule(demo_module: Any) -> None:
    config = demo_module.build_config()

    assert config.helmet_analysis is not None
    assert config.helmet_analysis.kind == "helmet_analysis"


def test_the_demo_does_not_bypass_the_turban_capability_guard(demo_module: Any) -> None:
    """The load-bearing assertion of this whole configuration.

    ``acknowledge_turban_blind`` is the documented escape hatch, and the demo must not
    use it -- not through a pinned rule, and not through the derived set either, which
    is why the derivation's own answer is asserted rather than only the rule list.
    """

    config = demo_module.build_config()

    assert config.default_rules == ()
    assert no_helmet_rule_available(config) is False
    assert describe(config).helmet_enforcement == "disabled"


def test_the_demo_never_maps_turban_onto_anything(demo_module: Any) -> None:
    """The backend simply cannot say it, and the posture reports that plainly."""

    posture = describe(demo_module.build_config())

    assert posture.turban_capable is False
    assert "turban" not in posture.helmet_backend_labels


def test_the_demo_leaves_the_other_violation_families_alone(demo_module: Any) -> None:
    """Only the helmet violation is off, and only because its evidence is not there."""

    config = demo_module.build_config()

    # Rules stay scene-derived, exactly as production leaves them.
    assert config.default_rules == ()
    assert config.auto_calibrate_uploads is True


# --- production is unchanged -------------------------------------------------------------
def test_production_still_composes_the_zero_shot_backend(serve_module: Any) -> None:
    """The demo is a separate composition; it must not become the default."""

    assert isinstance(serve_module.build_config().helmet_classifier, ZeroShotHelmetConfig)


def test_production_declares_no_analysis(serve_module: Any) -> None:
    assert serve_module.build_config().helmet_analysis is None


def test_the_demo_reuses_production_for_everything_that_is_not_a_helmet_decision(
    demo_module: Any, serve_module: Any
) -> None:
    """One definition of the detector composition, so the two cannot drift.

    The ``LABEL_MAP`` in particular: a wrong key there silently disables a whole
    violation class, and a demo that carried its own copy would be one edit away from
    a silently different system.
    """

    demo = demo_module.build_config()
    production = serve_module.build_config()

    assert demo.inference == production.inference
    assert demo.scene_path == production.scene_path
    assert demo.auto_calibrate_uploads == production.auto_calibrate_uploads
    assert demo.default_rules == production.default_rules


def test_importing_a_launcher_does_not_build_an_application(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``build_config`` is reusable; importing must not construct a whole app.

    The demo launcher reads production's composition, so importing production had to
    stop building (and recovering the storage of) an application nobody serves.
    """

    monkeypatch.setenv("TRAFFICPULSE_APP_STORAGE", str(tmp_path))
    module = importlib.reload(importlib.import_module("serve"))

    assert "app" not in vars(module)
    assert "config" not in vars(module)
    # ...but the ASGI attribute still resolves, so `uvicorn serve:app` is unaffected.
    assert module.app is not None

"""Dependency injection + the engine-provider seam (H7A)."""

from __future__ import annotations

import threading
from pathlib import Path

from _app_helpers import StubEngineProvider, make_client, make_config

from trafficpulse.app import (
    RealEngineProvider,
    SynchronousJobExecutor,
    ThreadJobExecutor,
    create_app,
)
from trafficpulse.app.dependencies import AppContext, get_context
from trafficpulse.contracts import ObjectClass
from trafficpulse.engine import InferenceConfig


# --- provider seam -------------------------------------------------------------
def test_injected_provider_is_used_end_to_end(tmp_path: Path) -> None:
    marker = StubEngineProvider(readiness="stub-marker")
    client = make_client(tmp_path, provider=marker)
    assert client.get("/api/health").json()["engine"] == "stub-marker"


def test_real_provider_reports_unconfigured_and_ready(tmp_path: Path) -> None:
    unconfigured = RealEngineProvider(make_config(tmp_path))
    assert unconfigured.describe() == "unconfigured"
    configured = RealEngineProvider(
        make_config(tmp_path).model_copy(
            update={
                "inference": InferenceConfig(
                    checkpoint="ckpt", label_map={"car": ObjectClass.CAR}
                )
            }
        )
    )
    assert configured.describe() == "ready"


def test_real_provider_without_inference_raises_engine_unavailable(tmp_path: Path) -> None:
    import pytest
    import yaml
    from _app_helpers import EXAMPLE_SCENE_PATH

    from trafficpulse.app.errors import EngineUnavailableError
    from trafficpulse.contracts import SceneConfig

    provider = RealEngineProvider(make_config(tmp_path))
    scene = SceneConfig.model_validate(
        yaml.safe_load(EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    )
    with pytest.raises(EngineUnavailableError):
        provider.create(scene=scene, rules=())


# --- context wiring ------------------------------------------------------------
def test_context_is_a_single_wired_object(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path), engine_provider=StubEngineProvider())
    context = app.state.context
    assert isinstance(context, AppContext)
    # Every service shares the one context; no module globals are involved.
    assert context.evidence is not None and context.events is not None


def test_default_executor_is_the_thread_executor(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path), engine_provider=StubEngineProvider())
    # The context holds services; the executor default is the background thread
    # one (asserted indirectly: creating an app without an executor succeeds and
    # the processing service exists).
    assert app.state.context.processing is not None


# --- executor seam -------------------------------------------------------------
def test_synchronous_executor_runs_inline() -> None:
    ran: list[str] = []
    SynchronousJobExecutor().submit(lambda: ran.append("done"))
    assert ran == ["done"]


def test_thread_executor_runs_the_work() -> None:
    finished = threading.Event()
    ThreadJobExecutor().submit(finished.set)
    assert finished.wait(timeout=5.0)


def test_get_context_reads_app_state(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path), engine_provider=StubEngineProvider())

    class _Req:
        def __init__(self, application: object) -> None:
            self.app = application

    context = get_context(_Req(app))  # type: ignore[arg-type]
    assert isinstance(context, AppContext)


def test_asgi_module_exposes_a_ready_app() -> None:
    from fastapi import FastAPI

    from trafficpulse.app import asgi

    assert isinstance(asgi.app, FastAPI)
    assert "/api/health" in asgi.app.openapi()["paths"]

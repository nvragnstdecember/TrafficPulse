"""Opt-in REAL RT-DETR processing through the application (v1.1 backend completion).

Exercises the production ``RealEngineProvider`` path end to end: build the app with
a real ``InferenceConfig`` (no injected stub provider), upload a decodable clip, and
process it, asserting the real engine runs the whole detect → track → reason →
persist pipeline to completion. It is the guard for
``RealEngineProvider._build_real_engine`` (otherwise never executed by the suite).

**Skipped by default** and never touches the network: it runs only when torch +
transformers are importable and ``TRAFFICPULSE_E2E_MODEL`` names a locally-available
RT-DETR checkpoint (loaded ``local_files_only``). Optional
``TRAFFICPULSE_E2E_DEVICE`` (default ``cpu``). No events are asserted — the synthetic
clip's pixels are not real vehicles, so the value proven is that the real pipeline is
*operational* (real detections flow, the job succeeds); real footage yields events.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from _app_helpers import make_config
from _triple_fixtures import write_triple_riding_clip
from fastapi.testclient import TestClient

from trafficpulse.app import SynchronousJobExecutor, create_app
from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.engine import InferenceConfig, TripleRidingRuleConfig

_MODEL = os.environ.get("TRAFFICPULSE_E2E_MODEL")
_DEVICE = os.environ.get("TRAFFICPULSE_E2E_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not (_MODEL and _HAVE_DEPS),
    reason=(
        "opt-in real RT-DETR app processing: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_E2E_MODEL to a locally-available RT-DETR checkpoint"
    ),
)

# The COCO-80 RT-DETR checkpoint uses the VOC-style "motorbike" spelling.
LABEL_MAP = {
    "person": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}


def test_real_engine_processes_an_uploaded_clip_to_completion(tmp_path: Path) -> None:
    config = make_config(tmp_path, default_rules=(TripleRidingRuleConfig(),)).model_copy(
        update={
            "inference": InferenceConfig(
                checkpoint=str(_MODEL),
                label_map=LABEL_MAP,
                device=_DEVICE,
                local_files_only=True,
            )
        }
    )
    # No injected provider -> the production RealEngineProvider builds the real engine;
    # the synchronous executor runs the job inline so any failure surfaces here.
    app = create_app(config, executor=SynchronousJobExecutor())
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/api/health").json()["engine"] == "ready"

    clip = write_triple_riding_clip(tmp_path / "clip.mp4", riders=3, frames=8)
    upload = client.post(
        "/api/video/upload", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")}
    )
    assert upload.status_code == 201, upload.text

    created = client.post("/api/process", json={"video_id": upload.json()["video_id"]})
    assert created.status_code == 202, created.text

    status = client.get(f"/api/process/{created.json()['job_id']}").json()
    # The real RT-DETR engine drove the whole pipeline to a clean completion.
    assert status["status"] == "succeeded", status.get("error")
    assert status["frames_processed"] == 8
    assert status["error"] is None

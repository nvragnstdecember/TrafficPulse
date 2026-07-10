"""Opt-in REAL RT-DETR end-to-end slice verification (P1-U12).

The only slice test that runs genuine ``torch`` + ``transformers`` RT-DETR
inference through the whole offline path (ingestion -> real detection -> IoU
tracking -> wrong-way reasoning -> event -> evidence). **Skipped by default** and
never touches the network. It runs only when BOTH

* the optional ``rtdetr`` dependencies are importable, and
* ``TRAFFICPULSE_E2E_MODEL`` points at a **locally available** RT-DETR checkpoint
  (a HuggingFace cache id or a local directory), reviewed per ADR-001.

Two honest modes:

* default (no clip provided): runs on a **generated synthetic** clip and asserts
  only that real inference *integrates end to end* -- the run completes and returns
  a valid report with an integer ``event_count`` (>= 0). A COCO RT-DETR is not
  expected to fire the vehicle class on synthetic pixels, so this proves the *seam*,
  not a detection;
* ``TRAFFICPULSE_E2E_CLIP`` set to a real wrong-way clip: additionally asserts
  ``event_count >= 1`` -- the genuine real-detector-to-event proof, pending an
  approved clip.

Device override: ``TRAFFICPULSE_E2E_DEVICE`` (default ``cpu``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from _slice_fixtures import write_wrong_way_clip

from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.detector import DetectorConfig
from trafficpulse.pipeline.runner import run_wrong_way_slice
from trafficpulse.tracking import IouTracker

_MODEL = os.environ.get("TRAFFICPULSE_E2E_MODEL")
_CLIP = os.environ.get("TRAFFICPULSE_E2E_CLIP")
_DEVICE = os.environ.get("TRAFFICPULSE_E2E_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not (_MODEL and _HAVE_DEPS),
    reason=(
        "opt-in real RT-DETR end-to-end slice: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_E2E_MODEL to a locally-available checkpoint (optionally "
        "TRAFFICPULSE_E2E_CLIP to a real wrong-way clip)"
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"


def _scene() -> SceneConfig:
    import yaml

    return SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text("utf-8")))


def test_real_rtdetr_slice_runs_end_to_end(tmp_path: Path) -> None:
    from trafficpulse.detector import RTDetrConfig, RTDetrDetector

    clip = Path(_CLIP) if _CLIP else write_wrong_way_clip(tmp_path / "clip.mp4")
    detector = RTDetrDetector(
        RTDetrConfig(checkpoint=str(_MODEL), device=_DEVICE, local_files_only=True, threshold=0.5)
    )
    report = run_wrong_way_slice(
        clip=clip,
        scene=_scene(),
        detector=detector,
        tracker=IouTracker(),
        detector_config=DetectorConfig(label_map={"car": ObjectClass.CAR}, score_threshold=0.5),
        output_dir=tmp_path / "runs",
        run_id="e2e-rtdetr",
        direction_id="dir-north",
        checkpoint=str(_MODEL),
        device=_DEVICE,
    )

    # Real inference integrated through the whole slice without leaking a backend type.
    assert report.detector_kind == "RTDetrDetector"
    assert report.frames_processed >= 1
    assert isinstance(report.event_count, int) and report.event_count >= 0
    assert report.manifest_count == report.event_count

    if _CLIP:  # an approved real wrong-way clip must yield the genuine event
        assert report.event_count >= 1

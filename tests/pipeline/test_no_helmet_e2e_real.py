"""Opt-in REAL end-to-end no-helmet slice on real footage (P4-U6).

The only test that runs the complete pipeline with **both** real perception
backends -- real RT-DETR detection and real zero-shot helmet classification -- on a
real video. It is **skipped by default** and never touches the network: it runs
only when ALL of

* the optional ``rtdetr`` dependencies are importable (``pip install
  'trafficpulse[rtdetr]'``);
* ``TRAFFICPULSE_E2E_MODEL`` names a **locally available** RT-DETR checkpoint;
* ``TRAFFICPULSE_HELMET_E2E_MODEL`` names a **locally available** CLIP-family
  checkpoint (weight provenance reviewed per ADR-001, U4 registry);
* ``TRAFFICPULSE_HELMET_E2E_CLIP`` points at a real local video file.

This is the P1-U7 / P1-U12 posture applied to the helmet slice: the default suite
proves the *wiring* offline against scripted perception, and this test is the
operator-triggered step that proves the *perception* on real pixels. Both are
needed; neither substitutes for the other.

It asserts **structure, not accuracy**: that the whole chain runs on real pixels
and persists well-formed, replayable records through the unmodified EventStore.
Asserting that a particular clip yields N helmet violations would be asserting an
accuracy claim this project has not earned -- zero-shot performance on small CCTV
head crops is unvalidated (P4-U1 measured a median rider head region of ~30px, and
P4-U4's crop geometry is deliberately containment-biased). Whether the perception
is any *good* is the pre-registered CNN-vs-ViT experiment's question, on
licence-cleared data.

Example:
    TRAFFICPULSE_E2E_MODEL=PekingU/rtdetr_r50vd \\
    TRAFFICPULSE_HELMET_E2E_MODEL=openai/clip-vit-base-patch32 \\
    TRAFFICPULSE_HELMET_E2E_CLIP=runs/viewer/_uploads/<clip>.webm \\
    ./.venv/Scripts/python.exe -m pytest tests/pipeline/test_no_helmet_e2e_real.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_DET_MODEL = os.environ.get("TRAFFICPULSE_E2E_MODEL")
_HELMET_MODEL = os.environ.get("TRAFFICPULSE_HELMET_E2E_MODEL")
_CLIP = os.environ.get("TRAFFICPULSE_HELMET_E2E_CLIP")
_DEVICE = os.environ.get("TRAFFICPULSE_HELMET_E2E_DEVICE", "cpu")
_HAVE_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not (_DET_MODEL and _HELMET_MODEL and _CLIP and _HAVE_DEPS),
    reason=(
        "opt-in real end-to-end no-helmet slice: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_E2E_MODEL, TRAFFICPULSE_HELMET_E2E_MODEL (locally-available "
        "checkpoints) and TRAFFICPULSE_HELMET_E2E_CLIP (a real local video)"
    ),
)


def test_real_end_to_end_no_helmet_slice(tmp_path: Path) -> None:
    from _helmet_fixtures import helmet_detector_config, helmet_example_scene

    from trafficpulse.classifier.zeroshot import (
        ZeroShotHelmetClassifier,
        ZeroShotHelmetConfig,
    )
    from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector
    from trafficpulse.persistence import EventStore
    from trafficpulse.pipeline.no_helmet_runner import run_no_helmet_slice
    from trafficpulse.tracking import IouTracker

    detector = RTDetrDetector(
        RTDetrConfig(
            checkpoint=str(_DET_MODEL), device=_DEVICE, local_files_only=True, threshold=0.5
        )
    )
    classifier = ZeroShotHelmetClassifier(
        ZeroShotHelmetConfig(
            checkpoint=str(_HELMET_MODEL), device=_DEVICE, local_files_only=True
        )
    )

    report = run_no_helmet_slice(
        clip=Path(str(_CLIP)),
        scene=helmet_example_scene(),
        detector=detector,
        tracker=IouTracker(),
        classifier=classifier,
        detector_config=helmet_detector_config(),
        output_dir=tmp_path / "out",
        run_id="real-1",
        checkpoint=str(_DET_MODEL),
        helmet_checkpoint=str(_HELMET_MODEL),
        device=_DEVICE,
    )

    # Structural only -- no accuracy claim (see module docstring).
    assert report.frames_processed > 0
    assert report.detector_kind == "RTDetrDetector"
    assert report.classifier_kind == "ZeroShotHelmetClassifier"
    assert report.checkpoint == str(_DET_MODEL)
    assert report.helmet_checkpoint == str(_HELMET_MODEL)
    assert report.event_count == report.manifest_count

    stored = EventStore(tmp_path / "out").load("real-1") if report.event_count else ()
    for record in stored:
        assert record.manifest.event_id == record.event.event_id
        assert record.event.violation_type.value == "no_helmet"
        # Real perception must stamp truthful provenance onto the event.
        assert record.event.models

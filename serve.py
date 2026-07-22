"""Runnable composition module that wires the REAL RT-DETR + helmet backend.

This is the operator composition root the architecture intends (see
``AppConfig`` docstring): it constructs a typed ``AppConfig`` with the real
inference detector, the real zero-shot helmet classifier, and a calibration-free
default rule set, then exposes ``app`` for any ASGI server:

    uvicorn serve:app --port 8000

Rule choice: ``triple_riding`` and ``no_helmet`` are motorcycle-perception rules
that need no per-camera geometry, so they work on arbitrary uploaded footage.
``wrong_way`` and ``illegal_stopping`` are deliberately NOT enabled here because
they require a ``SceneConfig`` calibrated to the uploaded video's camera (legal
lane directions / no-stopping zone); enabling them against the synthetic example
scene would produce meaningless geometry. Add them only with a scene calibrated
to your camera.

Checkpoints load offline from the local HuggingFace cache (local_files_only).
"""

from __future__ import annotations

from pathlib import Path

from trafficpulse.app import AppConfig, create_app
from trafficpulse.classifier import ZeroShotHelmetConfig
from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.engine import (
    InferenceConfig,
    NoHelmetRuleConfig,
    TripleRidingRuleConfig,
)

# The RT-DETR (COCO-80) checkpoint's native labels -> TrafficPulse classes. This
# checkpoint uses the VOC-style "motorbike" spelling (verified against the cached
# model's id2label), which is what association/perception need for motorcycles.
LABEL_MAP: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}

config = AppConfig(
    storage_dir=Path("trafficpulse-data"),
    scene_path=Path("configs/scenes/example-scene.yaml"),
    inference=InferenceConfig(
        checkpoint="PekingU/rtdetr_r50vd",
        label_map=LABEL_MAP,
        device="auto",  # uses CUDA when available, else CPU (CPU is ~2-3 s/frame)
        score_threshold=0.5,
        local_files_only=True,
    ),
    helmet_classifier=ZeroShotHelmetConfig(
        checkpoint="openai/clip-vit-base-patch32",
        device="cpu",
        local_files_only=True,
    ),
    default_rules=(TripleRidingRuleConfig(), NoHelmetRuleConfig()),
)

app = create_app(config)

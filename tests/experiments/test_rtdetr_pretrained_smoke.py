"""Opt-in REAL pretrained RT-DETR training smoke (H4B). Skipped by default.

Runs only when ``TRAFFICPULSE_HELMET_TRAIN_SMOKE_MODEL`` names a locally cached
RT-DETR checkpoint (e.g. ``PekingU/rtdetr_r50vd``). Offline
(``local_files_only=True``); one forward+backward step at full resolution; no
accuracy assertion — a single step on synthetic pixels proves the pretrained
weights load re-headed and the training graph runs, nothing more. The real
fine-tune remains gated on the CUDA install and the licence-cleared HELMET data.
"""

from __future__ import annotations

import os

import pytest
from _rtdetr_helpers import HAVE_TORCH

_MODEL = os.environ.get("TRAFFICPULSE_HELMET_TRAIN_SMOKE_MODEL")

pytestmark = pytest.mark.skipif(
    not (HAVE_TORCH and _MODEL),
    reason=(
        "opt-in real pretrained training smoke: install trafficpulse[rtdetr] and set "
        "TRAFFICPULSE_HELMET_TRAIN_SMOKE_MODEL to a locally cached RT-DETR checkpoint"
    ),
)


def test_pretrained_reheaded_training_step() -> None:
    import math

    import torch
    from helmet_rtdetr.rtdetr import RTDETRModel, build_optimizer
    from helmet_rtdetr.training import AdamWConfig

    model = RTDETRModel.from_pretrained(str(_MODEL), num_labels=2, local_files_only=True)
    model.train()
    pixel = torch.rand(1, 3, 640, 640)
    labels = [
        {"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])}
    ]
    out = model.forward(pixel_values=pixel, labels=labels)
    assert math.isfinite(float(out.loss))
    optimizer = build_optimizer(AdamWConfig(lr=1e-5), model.parameters())
    out.loss.backward()
    optimizer.step()  # one real optimisation step on the re-headed pretrained model

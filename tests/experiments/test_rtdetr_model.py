"""RT-DETR model wrapper + optimizer/scheduler builders (H4B). Tiny real torch."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from _rtdetr_helpers import HAVE_TORCH, TORCH_SKIP_REASON
from helmet_rtdetr.errors import ModelIOError
from helmet_rtdetr.training import (
    AdamWConfig,
    CosineSchedulerConfig,
    OneCycleSchedulerConfig,
    SgdConfig,
    StepSchedulerConfig,
)

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_SKIP_REASON)


@pytest.fixture(scope="module")
def tiny_model():
    from helmet_rtdetr.rtdetr import RTDETRModel

    return RTDETRModel.tiny()


def _inputs():
    import torch

    torch.manual_seed(0)
    pixel = torch.rand(1, 3, 64, 64)
    labels = [
        {"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    ]
    return pixel, labels


# --- wrapper ------------------------------------------------------------------
def test_forward_with_labels_yields_finite_loss(tiny_model) -> None:
    pixel, labels = _inputs()
    tiny_model.train()
    out = tiny_model.forward(pixel_values=pixel, labels=labels)
    assert math.isfinite(float(out.loss))
    assert out.loss_dict


def test_eval_forward_without_labels(tiny_model) -> None:
    import torch

    pixel, _ = _inputs()
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model.forward(pixel_values=pixel)
    assert out.loss is None
    assert tuple(out.logits.shape) == (1, 10, 2)  # (batch, queries, num_labels)
    assert tuple(out.pred_boxes.shape) == (1, 10, 4)


def test_decode_returns_per_image_detections(tiny_model) -> None:
    import torch

    pixel, _ = _inputs()
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model.forward(pixel_values=pixel)
    decoded = tiny_model.decode(
        out, target_sizes=torch.tensor([[64, 64]]), threshold=0.0
    )
    assert len(decoded) == 1
    assert set(decoded[0]) == {"scores", "labels", "boxes"}


def test_state_round_trip(tmp_path: Path, tiny_model) -> None:
    import torch
    from helmet_rtdetr.rtdetr import RTDETRModel

    path = tiny_model.save_state(tmp_path / "weights" / "model.pt")
    other = RTDETRModel.tiny()
    other.load_state(path)

    ours = tiny_model.state_dict()
    theirs = other.state_dict()
    assert all(torch.equal(ours[key], theirs[key]) for key in ours)


def test_load_state_missing_file_raises(tiny_model) -> None:
    with pytest.raises(ModelIOError, match="not found"):
        tiny_model.load_state(Path("definitely/absent.pt"))


def test_save_pretrained_exports_hf_layout(tmp_path: Path, tiny_model) -> None:
    out = tiny_model.save_pretrained(tmp_path / "export")
    assert (out / "config.json").is_file()


def test_build_with_no_checkpoint_gives_tiny(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import RTDETRModel, RTDETRModelConfig

    model = RTDETRModel.build(RTDETRModelConfig(checkpoint=None, num_labels=3))
    assert model.module.config.num_labels == 3


def test_build_with_unavailable_checkpoint_raises_typed_error(tmp_path: Path) -> None:
    """Offline-by-default: a non-cached checkpoint fails loudly, never downloads."""

    from helmet_rtdetr.rtdetr import RTDETRModel, RTDETRModelConfig

    with pytest.raises(ModelIOError, match="not available"):
        RTDETRModel.build(
            RTDETRModelConfig(
                checkpoint=str(tmp_path / "no-such-checkpoint"), local_files_only=True
            )
        )


def test_dataset_image_path_accessor(tmp_path: Path) -> None:
    from _rtdetr_helpers import unified, write_image
    from helmet_rtdetr.rtdetr import RTDETRDataset
    from helmet_rtdetr.unified import UnifiedClass

    write_image(tmp_path / "images" / "a.png", seed=1)
    split = tmp_path / "train.jsonl"
    split.write_text(
        unified("a.png", UnifiedClass.HELMET).model_dump_json() + "\n", encoding="utf-8"
    )
    data = RTDETRDataset(
        split, image_root=tmp_path / "images", image_height=64, image_width=64
    )
    assert data.image_path(0) == "a.png"


# --- optimizer construction ---------------------------------------------------
def test_adamw_construction_maps_config(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_optimizer

    optimizer = build_optimizer(
        AdamWConfig(lr=3e-4, weight_decay=0.05, betas=(0.8, 0.9), eps=1e-6),
        tiny_model.parameters(),
    )
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"], group["betas"], group["eps"]) == (
        3e-4,
        0.05,
        (0.8, 0.9),
        1e-6,
    )


def test_sgd_construction_maps_config(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_optimizer

    optimizer = build_optimizer(
        SgdConfig(lr=0.01, momentum=0.9, nesterov=True), tiny_model.parameters()
    )
    group = optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["nesterov"]) == (0.01, 0.9, True)


# --- scheduler construction ---------------------------------------------------
def _optimizer(tiny_model, lr: float = 0.1):
    from helmet_rtdetr.rtdetr import build_optimizer

    return build_optimizer(AdamWConfig(lr=lr), tiny_model.parameters())


def test_cosine_with_warmup_ramps_then_decays(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_scheduler

    optimizer = _optimizer(tiny_model)
    scheduler, granularity = build_scheduler(
        CosineSchedulerConfig(warmup_steps=4), optimizer, total_steps=20
    )
    assert granularity == "step"
    assert optimizer.param_groups[0]["lr"] < 1e-6  # warmup starts near zero
    for _ in range(4):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-3)  # ramped
    for _ in range(16):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.1  # decayed past the peak


def test_cosine_without_warmup_is_plain_cosine(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_scheduler

    optimizer = _optimizer(tiny_model)
    scheduler, granularity = build_scheduler(
        CosineSchedulerConfig(warmup_steps=0, min_lr_fraction=0.5), optimizer, total_steps=10
    )
    assert granularity == "step"
    for _ in range(10):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05, rel=1e-3)  # lr * fraction


def test_step_scheduler_decays_per_epoch(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_scheduler

    optimizer = _optimizer(tiny_model)
    scheduler, granularity = build_scheduler(
        StepSchedulerConfig(step_size=2, gamma=0.1), optimizer, total_steps=100
    )
    assert granularity == "epoch"
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_one_cycle_is_per_step_over_total_steps(tiny_model) -> None:
    from helmet_rtdetr.rtdetr import build_scheduler

    optimizer = _optimizer(tiny_model)
    scheduler, granularity = build_scheduler(
        OneCycleSchedulerConfig(pct_start=0.5), optimizer, total_steps=10
    )
    assert granularity == "step"
    start_lr = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > start_lr  # rose toward max_lr mid-cycle

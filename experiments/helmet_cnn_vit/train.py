"""Training loop for the CNN-vs-ViT comparison (P4-U5).

**One loop, both families.** Nothing in this module branches on architecture: the
model comes from :func:`helmet_cnn_vit.models.create_model`, the augmentation
recipe from :data:`helmet_cnn_vit.datasets.RECIPES`, and everything else --
optimiser, schedule, loss, epoch count, batch size, AMP, checkpoint selection -- is
literally shared code. That is what makes architecture the independent variable.

Reused from H4A
---------------
:func:`~helmet_rtdetr.training.seeding.derive_seed_plan` /
:func:`~helmet_rtdetr.training.seeding.apply_seed_plan` fan one base seed into
decorrelated per-component seeds and seed ``random``/NumPy, deferring torch to the
caller exactly as their contract specifies; this module applies the deferred torch
and cuDNN seeding. :class:`~helmet_rtdetr.training.run_layout.RunLayout` gives the
run-directory structure under the gitignored top-level ``runs/``.

H4A's full :class:`~helmet_rtdetr.training.trainer.Trainer` is not driven here: its
:class:`ExperimentConfig` models a detection run (optimiser declarations, resume
fingerprinting) and adapting it would add indirection without adding a guarantee
this loop needs. The two pieces that carry real reproducibility value -- the seed
plan and the run layout -- are reused directly.

Model selection
---------------
The best epoch is chosen by **validation macro-F1**, never by test. The test split
is touched exactly once per run, after training ends, and its predictions are
written out for :mod:`helmet_cnn_vit.stats` to consume.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from helmet_rtdetr.models import _Model
from helmet_rtdetr.training.run_layout import RunLayout
from helmet_rtdetr.training.seeding import apply_seed_plan, derive_seed_plan
from pydantic import Field

from .datasets import (
    CLASS_INDEX,
    INDEX_CLASS,
    RECIPES,
    build_dataset,
    build_loader,
    build_transform,
    class_weights,
    load_rows,
)
from .metrics import compute_metrics, predictions_from_scores
from .models import IMAGE_SIZE, create_model, normalisation_for, parameter_count, spec_for

#: Where runs land by default (top-level runs/ is gitignored; see RunLayout docs).
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs" / "helmet_cnn_vit"


class TrainConfig(_Model):
    """Everything that determines a run. Serialised verbatim into ``config.json``."""

    model: str
    seed: int = Field(default=0, ge=0)
    epochs: int = Field(default=12, ge=1)
    batch_size: int = Field(default=64, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    warmup_epochs: int = Field(default=1, ge=0)
    min_lr_factor: float = Field(default=0.01, gt=0.0, le=1.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    class_weighted: bool = True
    amp: bool = True
    num_workers: int = Field(default=4, ge=0)
    image_size: int = Field(default=IMAGE_SIZE, ge=32)

    def run_name(self) -> str:
        """Stable, human-readable run id: model, learning rate, seed."""

        return f"{self.model}_lr{self.lr:g}_s{self.seed}"


class EpochRecord(_Model):
    """One epoch's training loss and validation metrics."""

    epoch: int
    train_loss: float
    val_macro_f1: float | None
    val_accuracy: float
    learning_rate: float
    seconds: float


class SplitPredictions(_Model):
    """Per-crop predictions for one split -- the input to the statistics module."""

    split: str
    crop_ids: tuple[str, ...]
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    no_helmet_scores: tuple[float, ...]
    confidences: tuple[float, ...]


class TrainResult(_Model):
    """The full record of one training run."""

    config: TrainConfig
    run_dir: str
    timm_id: str
    family: str
    parameters: int
    checkpoint_bytes: int
    best_epoch: int
    best_val_macro_f1: float | None
    epochs: tuple[EpochRecord, ...]
    train_seconds: float
    device: str
    torch_version: str


def _cosine_lr(config: TrainConfig, epoch: int) -> float:
    """Linear warm-up then cosine decay, in epoch units. Shared by both families."""

    if config.warmup_epochs and epoch < config.warmup_epochs:
        return config.lr * (epoch + 1) / config.warmup_epochs
    span = max(1, config.epochs - config.warmup_epochs)
    progress = (epoch - config.warmup_epochs) / span
    floor = config.lr * config.min_lr_factor
    return floor + 0.5 * (config.lr - floor) * (1.0 + math.cos(math.pi * progress))


def infer(model: Any, loader: Any, device: Any, *, amp: bool) -> tuple[list[int], list[float]]:
    """Run the model over a loader, returning true class indices and no_helmet scores."""

    import torch

    model.eval()
    truth: list[int] = []
    scores: list[float] = []
    positive = CLASS_INDEX["no_helmet"]
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1)[:, positive]
            scores.extend(probabilities.cpu().tolist())
            truth.extend(int(t) for t in targets)
    return truth, scores


def _predictions(
    split: str, crop_ids: Sequence[str], truth: Sequence[int], scores: Sequence[float]
) -> SplitPredictions:
    labels, confidences = predictions_from_scores(scores)
    return SplitPredictions(
        split=split,
        crop_ids=tuple(crop_ids),
        truth=tuple(INDEX_CLASS[t] for t in truth),
        predicted=labels,
        no_helmet_scores=tuple(scores),
        confidences=confidences,
    )


def train_one(
    config: TrainConfig,
    *,
    crop_dir: Path,
    runs_root: Path | None = None,
    pretrained: bool = True,
    progress: int = 50,
) -> TrainResult:
    """Train one model at one seed, and write its run directory.

    Writes ``config.json``, ``seed_plan.json``, ``metrics/metrics.json``,
    ``checkpoints/best.pt``, and ``artifacts/predictions_{val,test}.json``.

    ``progress`` prints a throughput line every N batches (0 disables). A long GPU
    job with no output is indistinguishable from a deadlocked one -- which is not
    hypothetical here: a silent dataloader stall cost real time on this unit.
    """

    import torch
    from torch import nn

    spec = spec_for(config.model)
    layout = RunLayout(runs_root or DEFAULT_RUNS_ROOT, config.run_name())
    layout.create()

    # --- deterministic seeding (H4A plan; torch applied here, as its contract says)
    plan = derive_seed_plan(config.seed)
    apply_seed_plan(plan)
    torch.manual_seed(plan.torch_seed)
    torch.cuda.manual_seed_all(plan.torch_seed)
    torch.backends.cudnn.deterministic = plan.cudnn_deterministic
    torch.backends.cudnn.benchmark = not plan.cudnn_deterministic
    layout.seed_plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    layout.config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp and device.type == "cuda"

    model = create_model(spec, pretrained=pretrained).to(device)
    normalisation = normalisation_for(model)
    recipe = RECIPES[spec.family]

    train_rows = load_rows(crop_dir, "train")
    val_rows = load_rows(crop_dir, "val")
    test_rows = load_rows(crop_dir, "test")

    train_transform = build_transform(
        recipe, training=True, normalisation=normalisation, image_size=config.image_size
    )
    eval_transform = build_transform(
        recipe, training=False, normalisation=normalisation, image_size=config.image_size
    )
    train_loader = build_loader(
        build_dataset(crop_dir, train_rows, train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        seed=plan.torch_seed,
        num_workers=config.num_workers,
    )
    val_loader = build_loader(
        build_dataset(crop_dir, val_rows, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        seed=plan.torch_seed,
        num_workers=config.num_workers,
    )
    test_loader = build_loader(
        build_dataset(crop_dir, test_rows, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        seed=plan.torch_seed,
        num_workers=config.num_workers,
    )

    weights = (
        torch.tensor(class_weights(train_rows), dtype=torch.float32, device=device)
        if config.class_weighted
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_macro_f1: float | None = None
    best_epoch = -1
    records: list[EpochRecord] = []
    checkpoint_path = layout.checkpoints / "best.pt"
    started = time.perf_counter()

    for epoch in range(config.epochs):
        learning_rate = _cosine_lr(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        epoch_started = time.perf_counter()
        running = 0.0
        seen = 0
        window_started = time.perf_counter()
        window_seen = 0
        for batches, (images, targets) in enumerate(train_loader, start=1):
            if progress and batches % progress == 0:
                # Windowed, not cumulative: on Windows the DataLoader's spawn
                # workers take tens of seconds to start, and a cumulative average
                # folds that startup into every later reading, understating the
                # steady-state rate badly enough to look like a stall.
                now = time.perf_counter()
                rate = window_seen / max(1e-9, now - window_started)
                print(
                    f"    epoch {epoch} batch {batches} seen={seen} {rate:.0f} img/s",
                    flush=True,
                )
                window_started, window_seen = now, 0
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
            window_seen += images.size(0)

        truth, scores = infer(model, val_loader, device, amp=use_amp)
        labels, _ = predictions_from_scores(scores)
        metrics = compute_metrics([INDEX_CLASS[t] for t in truth], labels)
        if progress:
            print(
                f"  epoch {epoch}: loss={running / max(1, seen):.4f} "
                f"val_macroF1={metrics.macro_f1:.4f} "
                f"({time.perf_counter() - epoch_started:.0f}s)",
                flush=True,
            )
        record = EpochRecord(
            epoch=epoch,
            train_loss=running / max(1, seen),
            val_macro_f1=metrics.macro_f1,
            val_accuracy=metrics.accuracy,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - epoch_started,
        )
        records.append(record)

        current = metrics.macro_f1
        if current is not None and (best_macro_f1 is None or current > best_macro_f1):
            best_macro_f1 = current
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": config.model_dump()},
                checkpoint_path,
            )

    train_seconds = time.perf_counter() - started

    # Restore the validation-selected weights before the single test pass.
    if checkpoint_path.is_file():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])

    for split, rows, loader in (
        ("val", val_rows, val_loader),
        ("test", test_rows, test_loader),
    ):
        truth, scores = infer(model, loader, device, amp=use_amp)
        predictions = _predictions(split, [r.crop_id for r in rows], truth, scores)
        (layout.artifacts / f"predictions_{split}.json").write_text(
            predictions.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    result = TrainResult(
        config=config,
        run_dir=str(layout.run_dir),
        timm_id=spec.timm_id,
        family=spec.family,
        parameters=parameter_count(model),
        checkpoint_bytes=checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0,
        best_epoch=best_epoch,
        best_val_macro_f1=best_macro_f1,
        epochs=tuple(records),
        train_seconds=train_seconds,
        device=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        torch_version=torch.__version__,
    )
    (layout.metrics_dir / "metrics.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result


def load_predictions(run_dir: Path, split: str) -> SplitPredictions:
    """Read back one split's predictions from a finished run."""

    path = Path(run_dir) / "artifacts" / f"predictions_{split}.json"
    return SplitPredictions.model_validate(json.loads(path.read_text(encoding="utf-8")))

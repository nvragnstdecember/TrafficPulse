"""The RT-DETR training + validation loop, driving the H4A Trainer (H4B).

H4A owns lifecycle, state, callbacks, checkpoint metadata, logging, metrics, and
seeding; this module supplies **only** the model-specific work between the
lifecycle calls — exactly the division H4A was designed for. No lifecycle logic
is duplicated: the H4A ``Trainer`` is driven as intended
(``begin`` → per epoch ``begin_epoch`` / ``record_batch``... / ``end_epoch`` →
``end``), and every H4A guarantee (guards, resume fingerprints, duplicate-name
refusal, checkpoint policy, event ordering) applies unchanged.

Seeding: H4A's ``begin()`` applies the python/numpy components of the seed plan;
this loop applies the **deferred torch component** (``torch.manual_seed`` +
cuDNN determinism) immediately after — completing the H4A design — and only then
builds the model, so even a random-init model is deterministic per seed.

Resume: H4A restores ``TrainingState`` from the checkpoint metadata and validates
the config fingerprint; this loop then loads the matching weight payload —
model / optimizer / scheduler / scaler state_dicts — located by reconstructing
the checkpoint id from the restored state (the id is a pure function of
``(epoch, global_step)``, the same formula the manager uses). A resumed run whose
metadata exists but whose payload is missing fails loudly rather than silently
retraining from scratch.

AMP: autocast + GradScaler are enabled only when AMP is requested **and** CUDA is
present; on CPU both degrade gracefully to no-ops (verified in Step 0), so one
code path serves both devices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..errors import InvalidTrainingConfigError
from ..models import _Model
from ..training.callbacks import Callback
from ..training.config import ExperimentConfig
from ..training.events import LogSink
from ..training.metrics import MetricsStore
from ..training.seeding import derive_seed_plan
from ..training.state import CheckpointRecord, TrainingState
from ..training.trainer import Trainer
from .data import DataConfig, RTDETRDataset, build_dataloader
from .model import RTDETRModel, RTDETRModelConfig, require_torch
from .optim import SchedulerGranularity, build_optimizer, build_scheduler
from .payload import PAYLOAD_KEYS, PayloadStore


def checkpoint_id_for(state: TrainingState) -> str:
    """The manager's checkpoint id for a state — a pure function of (epoch, step)."""

    return f"e{state.epoch:04d}-s{state.global_step:08d}"


class RTDETRTrainerConfig(_Model):
    """Everything one RT-DETR training run needs: experiment + model + data."""

    experiment: ExperimentConfig
    model: RTDETRModelConfig
    data: DataConfig
    amp: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"


class _PayloadSaver(Callback):
    """Writes the weight payload whenever H4A checkpoints, then prunes orphans."""

    def __init__(self, owner: RTDETRTrainer) -> None:
        self._owner = owner

    def on_checkpoint(self, state: TrainingState, record: CheckpointRecord) -> None:
        self._owner._write_payload(record)


class RTDETRTrainer:
    """Composes the H4A Trainer with the RT-DETR model, data, and optimisation."""

    def __init__(
        self,
        config: RTDETRTrainerConfig,
        *,
        callbacks: Sequence[Callback] = (),
        sink: LogSink | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._trainer = Trainer(
            config.experiment,
            callbacks=(*callbacks, _PayloadSaver(self)),
            sink=sink,
            clock=clock,
        )
        self._payloads = PayloadStore(self._trainer.layout.checkpoints)
        self._model: RTDETRModel | None = None
        self._optimizer: Any = None
        self._scheduler: Any = None
        self._scheduler_granularity: SchedulerGranularity = "step"
        self._scaler: Any = None
        self._amp_active = False

    # --- read-only surface ------------------------------------------------------
    @property
    def state(self) -> TrainingState:
        return self._trainer.state

    @property
    def metrics(self) -> MetricsStore:
        return self._trainer.metrics

    @property
    def trainer(self) -> Trainer:
        """The underlying H4A trainer (layout, checkpoints, resumed flag)."""

        return self._trainer

    @property
    def amp_active(self) -> bool:
        """Whether mixed precision was actually enabled in the last ``fit`` (CUDA only)."""

        return self._amp_active

    # --- the run -----------------------------------------------------------------
    def fit(self) -> TrainingState:
        """Run the configured training to completion (or continuation); return state."""

        torch = require_torch()
        experiment = self._config.experiment

        state = self._trainer.begin()  # python/numpy seeded; resume resolved

        # The deferred torch component of the H4A seed plan, applied by the loop
        # exactly as seeding.py designed — before the model is built, so random
        # init is deterministic per seed.
        plan = derive_seed_plan(experiment.seed)
        torch.manual_seed(plan.torch_seed)
        if torch.cuda.is_available() and plan.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._resolve_device(torch)
        self._model = RTDETRModel.build(self._config.model).to(device)

        splits_dir = Path(self._config.data.splits_dir)
        image_root = Path(self._config.data.image_root)
        train_data = RTDETRDataset(
            splits_dir / "train.jsonl",
            image_root=image_root,
            image_height=self._config.data.image_height,
            image_width=self._config.data.image_width,
        )
        if len(train_data) == 0:
            raise InvalidTrainingConfigError("the train split contains no images")
        val_path = splits_dir / "val.jsonl"
        val_data = (
            RTDETRDataset(
                val_path,
                image_root=image_root,
                image_height=self._config.data.image_height,
                image_width=self._config.data.image_width,
            )
            if val_path.is_file()
            else None
        )

        train_loader = build_dataloader(
            train_data,
            batch_size=experiment.batch_size,
            shuffle=True,
            seed=plan.torch_seed,
            num_workers=experiment.num_workers,
        )
        val_loader = (
            build_dataloader(
                val_data,
                batch_size=experiment.batch_size,
                shuffle=False,
                seed=plan.torch_seed,
                num_workers=experiment.num_workers,
            )
            if val_data is not None and len(val_data) > 0
            else None
        )

        total_steps = max(1, len(train_loader)) * experiment.epochs
        self._optimizer = build_optimizer(experiment.optimizer, self._model.parameters())
        self._scheduler, self._scheduler_granularity = build_scheduler(
            experiment.scheduler, self._optimizer, total_steps=total_steps
        )

        from torch.amp import GradScaler

        self._amp_active = bool(self._config.amp and device.type == "cuda")
        self._scaler = GradScaler("cuda", enabled=self._amp_active)

        if self._trainer.resumed and state.epoch > 0:
            self._restore_payload(state)

        while self._trainer.state.epoch < experiment.epochs:
            self._trainer.begin_epoch()
            train_metrics = self._train_epoch(train_loader, device, torch)
            val_metrics = (
                self._validate(val_loader, device, torch) if val_loader is not None else {}
            )
            lr = float(self._optimizer.param_groups[0]["lr"])
            self._trainer.end_epoch({**train_metrics, **val_metrics, "train/lr": lr})

        return self._trainer.end()

    # --- internals ----------------------------------------------------------------
    def _resolve_device(self, torch: Any) -> Any:
        if self._config.device == "cpu":
            return torch.device("cpu")
        if self._config.device == "cuda":
            if not torch.cuda.is_available():
                raise InvalidTrainingConfigError(
                    "device='cuda' was requested but CUDA is not available"
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_epoch(self, loader: Any, device: Any, torch: Any) -> dict[str, float]:
        assert self._model is not None
        from torch.amp import autocast

        self._model.train()
        losses: list[float] = []
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [
                {key: value.to(device) for key, value in label.items()}
                for label in batch["labels"]
            ]
            with autocast(device.type, enabled=self._amp_active):
                outputs = self._model.forward(pixel_values=pixel_values, labels=labels)
            self._optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(outputs.loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            if self._scheduler_granularity == "step":
                self._scheduler.step()
            losses.append(float(outputs.loss.detach().cpu()))
            self._trainer.record_batch()
        if self._scheduler_granularity == "epoch":
            self._scheduler.step()
        return {"train/loss": sum(losses) / len(losses)} if losses else {}

    def _validate(self, loader: Any, device: Any, torch: Any) -> dict[str, float]:
        assert self._model is not None
        self._model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {key: value.to(device) for key, value in label.items()}
                    for label in batch["labels"]
                ]
                outputs = self._model.forward(pixel_values=pixel_values, labels=labels)
                losses.append(float(outputs.loss.detach().cpu()))
        return {"val/loss": sum(losses) / len(losses)} if losses else {}

    def _write_payload(self, record: CheckpointRecord) -> None:
        if self._model is None:  # checkpoint fired outside fit(): nothing to persist
            return
        self._payloads.save(
            record.checkpoint_id,
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
                "scaler": self._scaler.state_dict(),
            },
        )
        self._payloads.prune(self._trainer.checkpoints.checkpoint_ids())

    def _restore_payload(self, state: TrainingState) -> None:
        assert self._model is not None
        payload = self._payloads.load(checkpoint_id_for(state))
        missing = [key for key in PAYLOAD_KEYS if key not in payload]
        if missing:
            raise InvalidTrainingConfigError(
                f"checkpoint payload is missing components: {missing}"
            )
        self._model.load_state_dict(payload["model"])
        self._optimizer.load_state_dict(payload["optimizer"])
        self._scheduler.load_state_dict(payload["scheduler"])
        self._scaler.load_state_dict(payload["scaler"])

"""RT-DETR training integration (H4B) — plugs the model into the H4A seams.

Public API of the RT-DETR training subsystem, exposed as
``helmet_rtdetr.rtdetr`` (mirroring ``helmet_rtdetr.training``): the data
pipeline (H1–H3), the training infrastructure (H4A), and the RT-DETR integration
(H4B) stay separable namespaces.

torch/transformers/scipy are imported lazily inside the modules: importing this
package — and mypy over it — pulls in no ML framework. Every entry point that
actually needs the framework raises the typed ``BackendUnavailableError`` when
the optional ``rtdetr`` extra is absent.
"""

from __future__ import annotations

from ..errors import (
    BackendUnavailableError,
    DatasetIOError,
    ModelIOError,
    PayloadNotFoundError,
)
from .data import (
    LABEL_IDS,
    DataConfig,
    RTDETRDataset,
    build_dataloader,
    collate_batch,
    seed_worker,
)
from .loop import RTDETRTrainer, RTDETRTrainerConfig, checkpoint_id_for
from .model import DEFAULT_NUM_LABELS, RTDETRModel, RTDETRModelConfig, require_torch
from .optim import SchedulerGranularity, build_optimizer, build_scheduler
from .payload import PAYLOAD_KEYS, PayloadStore

__all__ = [
    # model
    "RTDETRModel",
    "RTDETRModelConfig",
    "DEFAULT_NUM_LABELS",
    "require_torch",
    # data
    "RTDETRDataset",
    "DataConfig",
    "LABEL_IDS",
    "build_dataloader",
    "collate_batch",
    "seed_worker",
    # optimisation
    "build_optimizer",
    "build_scheduler",
    "SchedulerGranularity",
    # payloads
    "PayloadStore",
    "PAYLOAD_KEYS",
    # loop
    "RTDETRTrainer",
    "RTDETRTrainerConfig",
    "checkpoint_id_for",
    # errors
    "BackendUnavailableError",
    "ModelIOError",
    "DatasetIOError",
    "PayloadNotFoundError",
]

"""The RT-DETR model wrapper (H4B).

Step-0 verification (performed against the installed libraries, not recalled)
-----------------------------------------------------------------------------
* **Source implementation:** the HuggingFace ``transformers`` RT-DETR port
  (Apache-2.0) — ``RTDetrForObjectDetection`` / ``RTDetrConfig`` /
  ``RTDetrImageProcessor`` — the same permissive port the runtime inference
  backend (``src/trafficpulse/detector/rtdetr.py``, P1-U7) uses, per ADR-001.
  Verified importable from transformers 5.13.0 in this environment.
* **Checkpoint format:** a HuggingFace id or local directory via
  ``from_pretrained`` (``PekingU/rtdetr_r50vd`` is locally cached); weights
  re-headed to our label count with ``num_labels=... , ignore_mismatched_sizes=True``.
* **Input format:** ``pixel_values`` float tensor ``(B, 3, H, W)``; training
  ``labels`` = one dict per image with ``class_labels`` (int64) and ``boxes``
  (float, **normalized cxcywh**) — exactly what ``RTDetrImageProcessor`` emits
  from COCO-format annotations (verified: bbox ``[10,10,20,20]`` in a 64px image
  → ``[0.312, 0.312, 0.312, 0.312]``).
* **Output tensors:** ``loss`` (scalar), ``loss_dict``, ``logits``
  ``(B, num_queries, num_labels)``, ``pred_boxes`` ``(B, num_queries, 4)``;
  decoded by ``post_process_object_detection``.
* **Training-only dependency:** the loss's Hungarian matcher requires **scipy**
  (BSD-3-Clause); pure inference does not — found empirically when the first
  training forward raised ``ImportError``.
* **Tiny-config path:** a small random-init ``RTDetrConfig`` (217K params,
  multi-scale backbone with explicit ``out_features``) runs forward+loss in
  ~0.04 s on CPU — what makes the H4B test suite fast and real.

torch/transformers are imported **lazily** (the repo's P1-U7 discipline): importing
this module — and mypy over the package — touches no ML framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from ..errors import BackendUnavailableError, ModelIOError
from ..models import _Model

# The binary helmet label space the approved training design fixes (turban is a
# rule-layer exemption, deliberately not a detector class).
DEFAULT_NUM_LABELS = 2


def require_torch() -> Any:
    """Import and return torch, or raise the typed backend error."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch present in this env
        raise BackendUnavailableError(
            "the RT-DETR training backend needs the optional 'rtdetr' extra "
            "(torch, transformers, scipy); install with: pip install 'trafficpulse[rtdetr]'"
        ) from exc
    return torch


class RTDETRModelConfig(_Model):
    """Which RT-DETR to build.

    ``checkpoint=None`` builds the **tiny random-init dev model** (the verified
    small config) — for tests and pipeline verification only; it carries no
    pretrained knowledge and no accuracy claim. A real run names the cached
    ``PekingU/rtdetr_r50vd`` (or a local directory), re-headed to ``num_labels``.
    """

    checkpoint: str | None = None
    num_labels: int = Field(default=DEFAULT_NUM_LABELS, ge=1)
    local_files_only: bool = True


class RTDETRModel:
    """A thin, trainer-isolated wrapper around ``RTDetrForObjectDetection``.

    Owns model construction (pretrained or tiny), checkpoint state I/O, mode
    switching, forward, and prediction decoding. It knows nothing about the
    Trainer, dataloaders, or optimizers — those compose it from outside.
    """

    def __init__(self, module: Any) -> None:
        self._module = module
        self._decode_processor: Any = None

    # --- construction ---------------------------------------------------------
    @classmethod
    def build(cls, config: RTDETRModelConfig) -> RTDETRModel:
        """Build per config: pretrained when a checkpoint is named, else tiny."""

        if config.checkpoint is None:
            return cls.tiny(num_labels=config.num_labels)
        return cls.from_pretrained(
            config.checkpoint,
            num_labels=config.num_labels,
            local_files_only=config.local_files_only,
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, *, num_labels: int, local_files_only: bool = True
    ) -> RTDETRModel:
        """Load a real RT-DETR checkpoint, re-headed to ``num_labels`` classes."""

        require_torch()
        from transformers import RTDetrForObjectDetection

        try:
            module = RTDetrForObjectDetection.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,  # COCO head (80) -> helmet head (2)
                local_files_only=local_files_only,
            )
        except OSError as exc:
            raise ModelIOError(
                f"checkpoint {checkpoint!r} is not available "
                f"(local_files_only={local_files_only}): {exc}"
            ) from exc
        return cls(module)

    @classmethod
    def tiny(cls, *, num_labels: int = DEFAULT_NUM_LABELS) -> RTDETRModel:
        """The verified tiny random-init RT-DETR (~217K params) for fast real tests.

        The exact configuration Step 0 validated: a 4-stage mini backbone with
        explicit multi-scale ``out_features`` (the default single-stage output
        breaks the encoder's feature-level indexing), 32-d model, 10 queries,
        denoising off. Random-init: useful for exercising the pipeline, useless
        for detection — and never presented otherwise.
        """

        require_torch()
        from transformers import RTDetrConfig, RTDetrForObjectDetection
        from transformers.models.rt_detr.configuration_rt_detr_resnet import (
            RTDetrResNetConfig,
        )

        backbone = RTDetrResNetConfig(
            embedding_size=16,
            hidden_sizes=[16, 32, 48, 64],
            depths=[1, 1, 1, 1],
            out_features=["stage2", "stage3", "stage4"],
        )
        config = RTDetrConfig(
            backbone_config=backbone,
            d_model=32,
            encoder_hidden_dim=32,
            num_queries=10,
            decoder_layers=1,
            encoder_layers=1,
            num_denoising=0,
            num_labels=num_labels,
            decoder_ffn_dim=32,
            encoder_ffn_dim=32,
            encoder_in_channels=[32, 48, 64],
            decoder_in_channels=[32, 32, 32],
            feat_channels=[32, 32, 32],
        )
        return cls(RTDetrForObjectDetection(config))

    # --- module surface --------------------------------------------------------
    @property
    def module(self) -> Any:
        """The underlying framework module (for optimizers and device moves)."""

        return self._module

    def forward(self, *, pixel_values: Any, labels: Any = None) -> Any:
        """One forward pass; with ``labels`` the output carries ``loss``/``loss_dict``."""

        return self._module(pixel_values=pixel_values, labels=labels)

    def train(self) -> None:
        self._module.train()

    def eval(self) -> None:
        self._module.eval()

    def to(self, device: Any) -> RTDETRModel:
        self._module.to(device)
        return self

    def parameters(self) -> Any:
        return self._module.parameters()

    def state_dict(self) -> Any:
        return self._module.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self._module.load_state_dict(state)

    # --- checkpoint state I/O ---------------------------------------------------
    def save_state(self, path: Path) -> Path:
        """Write the model state_dict to ``path`` (torch format); return ``path``."""

        torch = require_torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), path)
        return path

    def load_state(self, path: Path) -> None:
        """Load a state_dict written by :meth:`save_state`."""

        torch = require_torch()
        if not path.is_file():
            raise ModelIOError(f"model state file not found: {path}")
        self._module.load_state_dict(torch.load(path, map_location="cpu"))

    def save_pretrained(self, directory: Path) -> Path:
        """Export the model in HuggingFace layout (config + safetensors)."""

        directory.mkdir(parents=True, exist_ok=True)
        self._module.save_pretrained(str(directory))
        return directory

    # --- prediction decoding ----------------------------------------------------
    def decode(
        self, outputs: Any, *, target_sizes: Any, threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Decode raw outputs into per-image ``{scores, labels, boxes}`` (pixel xyxy).

        Uses the processor's ``post_process_object_detection`` — the same inverse
        transform the runtime inference backend relies on.
        """

        if self._decode_processor is None:
            from transformers import RTDetrImageProcessor

            self._decode_processor = RTDetrImageProcessor()
        result: list[dict[str, Any]] = self._decode_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )
        return result

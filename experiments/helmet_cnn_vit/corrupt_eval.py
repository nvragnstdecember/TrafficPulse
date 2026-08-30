"""Corruption-robustness evaluation (P4-U5).

The metadata slices in :mod:`helmet_cnn_vit.evaluate` are free -- they re-index
predictions that already exist. Corruption slices are not: each of the twelve
(corruption, severity) variants requires a fresh forward pass over the whole test
split, so this pass is separate and opt-in.

Discipline it preserves
------------------------
* Corruptions are applied to **test crops only**, at evaluation time. Nothing
  corrupted is ever trained on, and the corruption never touches train or val.
* The *same* corrupted image is shown to both families: the corruption is applied
  to the decoded crop before either model's normalisation, so any difference in the
  result is the model, not the input.
* The checkpoint evaluated is the validation-selected one each run already saved --
  this pass never retrains or reselects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from helmet_rtdetr.models import _Model

from .datasets import CLASS_INDEX, INDEX_CLASS, RECIPES, CropRow, build_transform, load_rows
from .metrics import compute_metrics, predictions_from_scores
from .models import create_model, normalisation_for, require_backend, spec_for
from .robustness import apply_corruption, corruption_variants


class CorruptionResult(_Model):
    """One model's metrics under one corruption at one severity."""

    corruption: str
    severity: int
    samples: int
    macro_f1: float | None
    balanced_accuracy: float | None
    accuracy: float


class CorruptionDataset:
    """Test crops with one corruption applied, then the model's eval transform.

    Module-level and picklable, for the same Windows ``spawn`` reason as
    :class:`helmet_cnn_vit.datasets.CropDataset`.
    """

    def __init__(
        self,
        crop_dir: Path,
        rows: Sequence[CropRow],
        transform: Any,
        corruption: str | None,
        severity: int,
    ) -> None:
        self._crop_dir = crop_dir
        self._rows = tuple(rows)
        self._transform = transform
        self._corruption = corruption
        self._severity = severity

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        row = self._rows[index]
        with Image.open(self._crop_dir / row.path) as handle:
            image = handle.convert("RGB")
        if self._corruption is not None:
            image = apply_corruption(image, self._corruption, self._severity)
        return self._transform(image), CLASS_INDEX[row.label]


def evaluate_corruptions(
    model_name: str,
    checkpoint: Path,
    *,
    crop_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[CorruptionResult]:
    """Evaluate one trained checkpoint across every corruption variant."""

    require_backend()
    import torch
    from torch.utils.data import DataLoader

    spec = spec_for(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = create_model(spec, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    model.eval()

    normalisation = normalisation_for(model)
    transform = build_transform(
        RECIPES[spec.family], training=False, normalisation=normalisation
    )
    rows = load_rows(crop_dir, "test")
    positive = CLASS_INDEX["no_helmet"]

    results: list[CorruptionResult] = []
    for corruption, severity in corruption_variants():
        dataset = CorruptionDataset(crop_dir, rows, transform, corruption, severity)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        truth: list[int] = []
        scores: list[float] = []
        with torch.inference_mode():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                scores.extend(torch.softmax(logits.float(), dim=1)[:, positive].cpu().tolist())
                truth.extend(int(t) for t in targets)

        labels, _ = predictions_from_scores(scores)
        metrics = compute_metrics([INDEX_CLASS[t] for t in truth], list(labels))
        results.append(
            CorruptionResult(
                corruption=corruption,
                severity=severity,
                samples=len(truth),
                macro_f1=metrics.macro_f1,
                balanced_accuracy=metrics.balanced_accuracy,
                accuracy=metrics.accuracy,
            )
        )
        print(
            f"    [{model_name}] {corruption} s{severity}: "
            f"macro-F1 {metrics.macro_f1:.4f}",
            flush=True,
        )
    return results

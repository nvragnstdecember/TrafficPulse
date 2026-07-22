"""Dataset + dataloader layer over the H3 split manifests (H4B).

Reads the JSONL split files that H3's ``export_splits`` writes (one
``UnifiedObject`` per line), groups objects by image, and turns each image into
the exact tensors Step 0 verified the model consumes: processor-resized
``pixel_values`` plus ``class_labels`` / normalized-cxcywh ``boxes``. Splitting is
**not** rebuilt here — the leakage-safe assignment happened in H3; this layer
only consumes its manifests.

Label space: ``helmet -> 0``, ``no_helmet -> 1`` (the approved binary detector).
``motorcycle`` objects and ``ignore``-flagged objects contribute **no** training
box; an image whose objects are all skipped is kept as a genuine negative sample
with an empty target (the matcher handles zero ground truth), constructed
explicitly rather than routed through the processor's empty-annotation edge case.

Determinism: dataset order is the H3 manifest order (already content-sorted);
shuffling happens only in the DataLoader via an injected, seeded generator, and
worker processes re-seed python/numpy from their torch seed via
:func:`seed_worker` — so the batch sequence is a pure function of the seed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from ..errors import DatasetIOError, InvalidTrainingConfigError
from ..models import NonEmptyStr, _Model
from ..unified import UnifiedClass, UnifiedObject
from .model import require_torch

# UnifiedClass -> contiguous model label ids (the detector's binary head).
LABEL_IDS: dict[UnifiedClass, int] = {
    UnifiedClass.HELMET: 0,
    UnifiedClass.NO_HELMET: 1,
}


class DataConfig(_Model):
    """Where the split manifests and images live, and the training resolution."""

    splits_dir: NonEmptyStr
    image_root: NonEmptyStr
    image_height: int = Field(default=640, ge=32)
    image_width: int = Field(default=640, ge=32)

    @model_validator(mode="after")
    def _stride_divisible(self) -> Self:
        # RT-DETR's multi-scale features use strides 8/16/32; a non-divisible
        # input would silently change the effective resolution.
        if self.image_height % 32 or self.image_width % 32:
            raise InvalidTrainingConfigError(
                f"image size {self.image_width}x{self.image_height} must be "
                "divisible by 32 (RT-DETR feature strides)"
            )
        return self


class RTDETRDataset:
    """One H3 split manifest as a torch-style map dataset (index -> tensors)."""

    def __init__(
        self,
        split_path: Path,
        *,
        image_root: Path,
        image_height: int,
        image_width: int,
    ) -> None:
        require_torch()
        from transformers import RTDetrImageProcessor

        if not split_path.is_file():
            raise DatasetIOError(f"split manifest not found: {split_path}")
        objects = [
            UnifiedObject.model_validate_json(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_image: dict[str, list[UnifiedObject]] = {}
        for obj in objects:  # manifest order is already deterministic (H3)
            by_image.setdefault(obj.image_path, []).append(obj)
        self._samples: list[tuple[str, list[UnifiedObject]]] = list(by_image.items())
        self._image_root = image_root

        missing = [path for path, _ in self._samples if not (image_root / path).is_file()]
        if missing:
            raise DatasetIOError(
                f"{len(missing)} image(s) referenced by {split_path.name} are missing "
                f"under {image_root}: {missing[:5]}"
            )
        self._processor = RTDetrImageProcessor(
            do_resize=True, size={"height": image_height, "width": image_width}
        )

    def __len__(self) -> int:
        return len(self._samples)

    def image_path(self, index: int) -> str:
        """The manifest-relative image path of one sample (test/diagnostic aid)."""

        return self._samples[index][0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = require_torch()
        from PIL import Image

        image_path, objects = self._samples[index]
        with Image.open(self._image_root / image_path) as handle:
            image = handle.convert("RGB")

        annotations = [
            {
                "image_id": index,
                "category_id": LABEL_IDS[obj.label],
                "bbox": [obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h],
                "area": obj.bbox.area,
                "iscrowd": 0,
            }
            for obj in objects
            if not obj.ignore and obj.label in LABEL_IDS
        ]
        if annotations:
            encoded = self._processor(
                images=image,
                annotations={"image_id": index, "annotations": annotations},
                return_tensors="pt",
            )
            labels = {key: value for key, value in encoded["labels"][0].items()}
        else:
            # A genuine negative: explicit empty target, not a processor edge case.
            encoded = self._processor(images=image, return_tensors="pt")
            labels = {
                "class_labels": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }
        return {"pixel_values": encoded["pixel_values"][0], "labels": labels}


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size images; keep per-image variable-length labels as a list."""

    torch = require_torch()
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "labels": [sample["labels"] for sample in samples],
    }


def seed_worker(worker_id: int) -> None:
    """Re-seed python/numpy inside a DataLoader worker from its torch seed.

    torch hands each worker a distinct deterministic seed derived from the
    loader's generator; propagating it keeps any python/numpy randomness in
    augmentation code deterministic too.
    """

    torch = require_torch()
    worker_seed = torch.initial_seed() % 2**32

    import numpy

    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: RTDETRDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> Any:
    """A deterministic DataLoader: seeded generator + per-worker re-seeding."""

    torch = require_torch()
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        generator=generator,
        worker_init_fn=seed_worker,
    )

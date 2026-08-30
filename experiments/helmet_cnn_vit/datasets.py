"""Crop dataset and the per-family augmentation policy (P4-U5).

Augmentation fairness (evaluation-protocol §8)
----------------------------------------------
The protocol is explicit that identical recipes are *not* required and can
themselves be unfair -- ViTs are known to need stronger augmentation than CNNs at
this data scale, so forcing the CNN's recipe on DeiT would handicap it and forcing
DeiT's on ResNet would be an equally arbitrary choice. What must be equal is the
*tuning budget*, which :mod:`helmet_cnn_vit.train` enforces.

So: both families share a base recipe (random resized crop, horizontal flip, colour
jitter), and the ViT additionally gets RandAugment, which is part of DeiT's
published recipe. Both recipes are declared as data below and recorded in the run
config, so the difference is visible in the artifact rather than buried in code.

Deviation recorded honestly: DeiT's published recipe also uses mixup and CutMix.
Neither is applied here, to **either** family. Mixup needs soft targets, which
would change the loss and metric path for one model only unless implemented for
both; within this unit's budget the symmetric choice is to omit it from both and
say so. This is a deviation from "each family's best-known recipe" and is listed in
the pre-registration.

Determinism
-----------
The evaluation transform has no randomness at all. Training randomness comes from
the seeded ``torch`` global RNG (see ``helmet_rtdetr.training.seeding``) and from a
seeded loader generator, so a given seed reproduces a given run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from helmet_rtdetr.models import _Model

from .errors import CorpusBuildError
from .labels import HelmetState
from .models import IMAGE_SIZE, require_backend

#: Class index order. Fixed here so the confusion matrix, the loss weights, and the
#: reported label names can never drift apart.
CLASS_INDEX: dict[str, int] = {HelmetState.HELMET.value: 0, HelmetState.NO_HELMET.value: 1}
INDEX_CLASS: dict[int, str] = {v: k for k, v in CLASS_INDEX.items()}


class AugmentationRecipe(_Model):
    """A declarative augmentation policy, serialised into the run config."""

    name: str
    scale_min: float = 0.7
    scale_max: float = 1.0
    hflip: float = 0.5
    colour_jitter: float = 0.2
    #: timm's ``auto_augment`` string, or ``None``. RandAugment for the ViT only.
    auto_augment: str | None = None


#: Shared base -- the CNN family's recipe.
BASE_RECIPE = AugmentationRecipe(name="shared-base")

#: Base plus RandAugment (magnitude 7, 2 ops), from DeiT's published recipe.
VIT_RECIPE = AugmentationRecipe(name="shared-base+randaugment", auto_augment="rand-m7-n2-mstd0.5")

RECIPES: dict[str, AugmentationRecipe] = {
    "cnn": BASE_RECIPE,
    "vit": VIT_RECIPE,
}


class CropRow(_Model):
    """One row of the extraction index, as the dataset consumes it."""

    crop_id: str
    split: str
    label: str
    video_id: str
    site_id: str
    track_id: str
    frame_index: int
    rider_count: int
    any_no_helmet: bool
    source_label: str
    box_w: float
    box_h: float
    path: str


def load_rows(crop_dir: Path, split: str) -> tuple[CropRow, ...]:
    """Read the extraction index and return one split's rows, in index order."""

    path = crop_dir / "crops.jsonl"
    if not path.is_file():
        raise CorpusBuildError(f"no crop index at {path}; run extraction first")
    rows = [
        CropRow.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = tuple(row for row in rows if row.split == split)
    if not selected:
        raise CorpusBuildError(f"no crops for split {split!r} in {path}")
    return selected


def class_weights(rows: Sequence[CropRow]) -> tuple[float, float]:
    """Inverse-frequency weights for the loss, in :data:`CLASS_INDEX` order.

    §12 requires class-weighted cross-entropy. Weights are computed from the
    **training** rows only and are the same for both families, so the imbalance
    handling cannot advantage either one.
    """

    counts = [0, 0]
    for row in rows:
        counts[CLASS_INDEX[row.label]] += 1
    if min(counts) == 0:
        raise CorpusBuildError(f"a class is absent from the training split: {counts}")
    total = sum(counts)
    return (total / (2 * counts[0]), total / (2 * counts[1]))


def build_transform(
    recipe: AugmentationRecipe,
    *,
    training: bool,
    normalisation: dict[str, Any],
    image_size: int = IMAGE_SIZE,
) -> Any:
    """Compose the torchvision transform for one family and one phase.

    Built through ``timm.data.create_transform`` so each family gets the same
    battle-tested implementation of its recipe rather than a hand-rolled near-copy.
    """

    require_backend()
    from timm.data import create_transform

    mean = normalisation.get("mean")
    std = normalisation.get("std")
    interpolation = normalisation.get("interpolation", "bicubic")

    if not training:
        # No randomness whatsoever on val/test: crops are already square 224s, so
        # this is a tensor conversion plus the checkpoint's own normalisation.
        return create_transform(
            input_size=(3, image_size, image_size),
            is_training=False,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=1.0,
        )

    return create_transform(
        input_size=(3, image_size, image_size),
        is_training=True,
        scale=(recipe.scale_min, recipe.scale_max),
        hflip=recipe.hflip,
        color_jitter=recipe.colour_jitter,
        auto_augment=recipe.auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        re_prob=0.0,  # no random erasing: not part of either declared recipe
    )


class CropDataset:
    """A map-style dataset over the extracted crops.

    Deliberately a plain module-level class rather than a ``torch.utils.data.Dataset``
    subclass defined inside a factory. ``DataLoader`` accepts any object with
    ``__len__``/``__getitem__``, so inheritance buys nothing -- and on Windows the
    loader's workers use the ``spawn`` start method, which pickles the dataset. A
    class defined inside a function is unpicklable, so that version crashed with
    ``Can't get local object`` the moment ``num_workers > 0``. Module level keeps the
    class picklable *and* keeps torch out of this module's import graph.
    """

    def __init__(self, crop_dir: Path, rows: Sequence[CropRow], transform: Any) -> None:
        self._crop_dir = crop_dir
        self._rows = tuple(rows)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        row = self._rows[index]
        with Image.open(self._crop_dir / row.path) as handle:
            image = handle.convert("RGB")
        return self._transform(image), CLASS_INDEX[row.label]


def build_dataset(crop_dir: Path, rows: Sequence[CropRow], transform: Any) -> CropDataset:
    """A dataset over ``rows``, reading crops from ``crop_dir``."""

    return CropDataset(crop_dir, rows, transform)


def build_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 4,
) -> Any:
    """A seeded ``DataLoader``. Shuffling is driven by an explicit generator.

    Worker budget, learned the hard way
    -----------------------------------
    On Windows the loader spawns workers as fresh processes, each of which imports
    torch and timm and costs on the order of a gigabyte of RSS. A run builds three
    loaders (train/val/test); giving all three four *persistent* workers meant twelve
    such processes, which exhausted the 16 GB machine -- the paging file gave out and
    even ``bash`` could no longer fork.

    So workers are spent only where they pay for themselves: the training loader,
    which is read every epoch and is JPEG-decode bound. The evaluation loaders run
    once per epoch over a fraction of the data and use the main process. Persistent
    workers are likewise limited to the training loader, where the respawn cost would
    otherwise be paid every epoch.
    """

    require_backend()
    import torch
    from torch.utils.data import DataLoader

    # Only the shuffled (training) loader gets workers; see the docstring.
    workers = num_workers if shuffle else 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=generator if shuffle else None,
        persistent_workers=workers > 0,
    )

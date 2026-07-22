"""RT-DETR dataset + dataloader layer (H4B). Real (tiny) torch path."""

from __future__ import annotations

from pathlib import Path

import pytest
from _rtdetr_helpers import HAVE_TORCH, TORCH_SKIP_REASON, unified, write_image
from helmet_rtdetr.errors import DatasetIOError, InvalidTrainingConfigError
from helmet_rtdetr.unified import UnifiedClass

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_SKIP_REASON)


def write_split(tmp_path: Path, objects: list) -> tuple[Path, Path]:
    """Write a JSONL manifest (H3 export format) + its images; return (split, root)."""

    image_root = tmp_path / "images"
    for obj in objects:
        write_image(image_root / obj.image_path, seed=hash(obj.image_path) % 1000)
    split = tmp_path / "train.jsonl"
    split.write_text(
        "\n".join(o.model_dump_json() for o in objects) + "\n", encoding="utf-8"
    )
    return split, image_root


def dataset(split: Path, image_root: Path):
    from helmet_rtdetr.rtdetr import RTDETRDataset

    return RTDETRDataset(split, image_root=image_root, image_height=64, image_width=64)


# --- grouping + label mapping -------------------------------------------------
def test_objects_group_into_one_sample_per_image(tmp_path: Path) -> None:
    split, root = write_split(
        tmp_path,
        [
            unified("a.png", UnifiedClass.HELMET, box=(4.0, 4.0, 16.0, 16.0)),
            unified("a.png", UnifiedClass.NO_HELMET, box=(30.0, 30.0, 20.0, 20.0)),
            unified("b.png", UnifiedClass.HELMET),
        ],
    )
    data = dataset(split, root)
    assert len(data) == 2  # two images, not three objects


def test_label_ids_map_the_binary_head(tmp_path: Path) -> None:
    split, root = write_split(
        tmp_path,
        [
            unified("a.png", UnifiedClass.HELMET, box=(4.0, 4.0, 16.0, 16.0)),
            unified("a.png", UnifiedClass.NO_HELMET, box=(30.0, 30.0, 20.0, 20.0)),
        ],
    )
    labels = dataset(split, root)[0]["labels"]
    assert sorted(int(v) for v in labels["class_labels"]) == [0, 1]  # helmet=0, no_helmet=1


def test_motorcycle_and_ignored_objects_contribute_no_boxes(tmp_path: Path) -> None:
    split, root = write_split(
        tmp_path,
        [
            unified("moto.png", UnifiedClass.MOTORCYCLE),
            unified("ign.png", UnifiedClass.NO_HELMET, ignore=True),
        ],
    )
    data = dataset(split, root)
    assert len(data) == 2  # kept as genuine negatives...
    for index in range(2):
        labels = data[index]["labels"]
        assert labels["class_labels"].numel() == 0  # ...with explicit empty targets
        assert tuple(labels["boxes"].shape) == (0, 4)


def test_boxes_are_normalized(tmp_path: Path) -> None:
    split, root = write_split(tmp_path, [unified("a.png", UnifiedClass.HELMET)])
    boxes = dataset(split, root)[0]["labels"]["boxes"]
    assert bool(((boxes >= 0.0) & (boxes <= 1.0)).all())


def test_sample_tensor_shapes(tmp_path: Path) -> None:
    split, root = write_split(tmp_path, [unified("a.png", UnifiedClass.HELMET)])
    sample = dataset(split, root)[0]
    assert tuple(sample["pixel_values"].shape) == (3, 64, 64)


# --- collate ------------------------------------------------------------------
def test_collate_stacks_pixels_and_lists_labels(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import collate_batch

    split, root = write_split(
        tmp_path, [unified("a.png", UnifiedClass.HELMET), unified("b.png", UnifiedClass.NO_HELMET)]
    )
    data = dataset(split, root)
    batch = collate_batch([data[0], data[1]])
    assert tuple(batch["pixel_values"].shape) == (2, 3, 64, 64)
    assert isinstance(batch["labels"], list) and len(batch["labels"]) == 2


# --- deterministic loading ----------------------------------------------------
def _epoch_order(split: Path, root: Path, seed: int) -> list[int]:
    import torch
    from helmet_rtdetr.rtdetr import build_dataloader

    data = dataset(split, root)
    loader = build_dataloader(data, batch_size=1, shuffle=True, seed=seed, num_workers=0)
    order = []
    for batch in loader:
        # batch of one image: identify it by its pixel checksum
        order.append(int(torch.sum(batch["pixel_values"] * 1000)) % 100003)
    return order


def test_same_seed_yields_identical_shuffle_order(tmp_path: Path) -> None:
    objects = [unified(f"i{k}.png", UnifiedClass.HELMET) for k in range(8)]
    split, root = write_split(tmp_path, objects)
    assert _epoch_order(split, root, seed=5) == _epoch_order(split, root, seed=5)


def test_different_seed_yields_different_shuffle_order(tmp_path: Path) -> None:
    objects = [unified(f"i{k}.png", UnifiedClass.HELMET) for k in range(8)]
    split, root = write_split(tmp_path, objects)
    assert _epoch_order(split, root, seed=5) != _epoch_order(split, root, seed=6)


def test_seed_worker_reseeds_python_and_numpy(tmp_path: Path) -> None:
    import random

    import numpy
    import torch
    from helmet_rtdetr.rtdetr import seed_worker

    torch.manual_seed(1234)
    seed_worker(0)
    first = (random.random(), float(numpy.random.random()))
    torch.manual_seed(1234)
    seed_worker(0)
    assert (random.random(), float(numpy.random.random())) == first


# --- failure modes ------------------------------------------------------------
def test_missing_split_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetIOError, match="split manifest"):
        dataset(tmp_path / "absent.jsonl", tmp_path)


def test_missing_image_fails_fast_at_construction(tmp_path: Path) -> None:
    split, root = write_split(tmp_path, [unified("a.png", UnifiedClass.HELMET)])
    (root / "a.png").unlink()
    with pytest.raises(DatasetIOError, match="missing"):
        dataset(split, root)


def test_data_config_requires_stride_divisible_sizes(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import DataConfig

    with pytest.raises(InvalidTrainingConfigError, match="divisible by 32"):
        DataConfig(splits_dir="s", image_root="i", image_height=100, image_width=64)

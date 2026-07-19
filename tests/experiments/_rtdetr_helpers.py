"""Shared fixtures for the H4B RT-DETR integration tests.

Builds a tiny REAL corpus — synthetic PNGs on disk, unified objects through the
actual H2 corpus builder and H3 splitter — so the H4B tests consume genuine H3
manifests rather than hand-rolled files. Uniquely named (``_rtdetr_helpers``) for
pytest's prepend import mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from helmet_rtdetr import (
    BBox,
    CorpusBuilder,
    CorpusMember,
    CorpusVersion,
    ObjectProvenance,
    SplitBuilder,
    SplitRatios,
    UnifiedClass,
    UnifiedObject,
    export_splits,
)
from helmet_rtdetr.training import (
    AdamWConfig,
    CheckpointPolicy,
    CosineSchedulerConfig,
    ExperimentConfig,
    LoggingConfig,
    ResumeConfig,
)

HAVE_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

TORCH_SKIP_REASON = (
    "H4B tests exercise the real (tiny) torch/transformers path; install "
    "trafficpulse[rtdetr] to run them"
)


def unified(
    image_path: str,
    label: UnifiedClass = UnifiedClass.NO_HELMET,
    *,
    box: tuple[float, float, float, float] = (8.0, 8.0, 24.0, 24.0),
    ignore: bool = False,
) -> UnifiedObject:
    x, y, w, h = box
    return UnifiedObject(
        image_path=image_path,
        bbox=BBox(x=x, y=y, w=w, h=h),
        label=label,
        provenance=ObjectProvenance(
            dataset_id="helmet-myanmar", dataset_version="1", adapter="coco", source_label="x"
        ),
        ignore=ignore,
    )


def write_image(path: Path, *, seed: int, size: int = 64) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rng.random((size, size, 3)) * 255).astype("uint8")).save(path)
    return path


def make_split_fixture(
    tmp_path: Path, *, n_images: int = 8
) -> tuple[Path, Path]:
    """Real corpus -> real H3 splits: returns ``(splits_dir, image_root)``."""

    image_root = tmp_path / "images"
    objects = []
    for index in range(n_images):
        name = f"img{index:03d}.png"
        write_image(image_root / name, seed=index)
        label = UnifiedClass.HELMET if index % 2 else UnifiedClass.NO_HELMET
        objects.append(unified(name, label))
    corpus_version = CorpusVersion(
        corpus_id="test-corpus",
        version="1",
        members=(CorpusMember(dataset_id="helmet-myanmar", dataset_version="1"),),
    )
    corpus = CorpusBuilder(corpus_version).add(objects).build()
    result = SplitBuilder(
        corpus, ratios=SplitRatios(train=0.5, val=0.25, test=0.25), seed=3
    ).build()
    splits_dir = tmp_path / "splits"
    export_splits(result, splits_dir)
    return splits_dir, image_root


def make_rt_config(
    tmp_path: Path,
    *,
    name: str = "run-a",
    epochs: int = 2,
    seed: int = 7,
    resume: bool = False,
    keep_last: int = 2,
    save_best: bool = True,
    scheduler: object | None = None,
    device: str = "auto",
):  # -> RTDETRTrainerConfig (annotated loosely: module imports lazily under skip)
    from helmet_rtdetr.rtdetr import DataConfig, RTDETRModelConfig, RTDETRTrainerConfig

    splits_dir, image_root = (tmp_path / "splits"), (tmp_path / "images")
    return RTDETRTrainerConfig(
        experiment=ExperimentConfig(
            name=name,
            output_root=str(tmp_path / "runs"),
            seed=seed,
            epochs=epochs,
            batch_size=2,
            optimizer=AdamWConfig(lr=1e-4),
            scheduler=scheduler if scheduler is not None else CosineSchedulerConfig(warmup_steps=2),
            checkpoint=CheckpointPolicy(
                save_best=save_best,
                best_metric="val/loss" if save_best else None,
                best_mode="min",
                keep_last=keep_last,
            ),
            logging=LoggingConfig(backend="memory"),
            resume=ResumeConfig(enabled=resume),
        ),
        model=RTDETRModelConfig(checkpoint=None),
        data=DataConfig(
            splits_dir=str(splits_dir),
            image_root=str(image_root),
            image_height=64,
            image_width=64,
        ),
        amp=True,
        device=device,  # type: ignore[arg-type]
    )

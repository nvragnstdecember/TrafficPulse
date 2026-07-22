"""Split manifests + statistics (H3).

Deterministic, publication-grade metadata for a split: what corpus was split, at
what ratios, with what seed, and how the objects/images/classes/datasets landed.

Determinism
-----------
Every count map is sorted; the split JSONL and statistics are pure functions of
the corpus + seed + ratios. The manifest's ``generated_at`` is the **only**
run-specific field and is injectable (default ``None``), so a reproducibility test
gets byte-identical manifests and a production caller can stamp the real time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from .models import CorpusVersion, _Model
from .split import SPLIT_ORDER, SplitRatios
from .unified import UnifiedObject

GENERATOR_VERSION = "h3-splitter/1.0.0"

Splits = Mapping[str, Sequence[UnifiedObject]]


class DatasetStats(_Model):
    """Aggregate statistics over one set of objects (a split, or the whole corpus)."""

    objects: int
    images: int
    per_class: dict[str, int]
    per_dataset: dict[str, int]
    imbalance_ratio: float | None  # max class count / min class count (None if < 2 classes)
    minority_class: str | None


class SplitCounts(_Model):
    """The object/image counts of one split (the manifest's compact summary)."""

    objects: int
    images: int


class SplitStatistics(_Model):
    """Corpus-wide + per-split statistics, in canonical split order."""

    total: DatasetStats
    per_split: dict[str, DatasetStats]


class SplitManifest(_Model):
    """Reproducible provenance for one split run."""

    generator_version: str
    generated_at: datetime | None
    seed: int
    ratios: SplitRatios
    grouping: str
    corpus_hash: str
    corpus_version: CorpusVersion
    counts: dict[str, SplitCounts]
    total_objects: int
    total_images: int


def _stats_for(objects: Sequence[UnifiedObject]) -> DatasetStats:
    per_class: dict[str, int] = {}
    per_dataset: dict[str, int] = {}
    images: set[str] = set()
    for obj in objects:
        per_class[obj.label.value] = per_class.get(obj.label.value, 0) + 1
        ds = obj.provenance.dataset_id
        per_dataset[ds] = per_dataset.get(ds, 0) + 1
        images.add(obj.image_path)

    imbalance_ratio: float | None = None
    minority_class: str | None = None
    if per_class:
        counts = sorted(per_class.items())  # deterministic (name, count)
        minority_class = min(counts, key=lambda kv: (kv[1], kv[0]))[0]
        values = [c for _, c in counts]
        if len(values) >= 2 and min(values) > 0:
            imbalance_ratio = max(values) / min(values)
    return DatasetStats(
        objects=len(objects),
        images=len(images),
        per_class=dict(sorted(per_class.items())),
        per_dataset=dict(sorted(per_dataset.items())),
        imbalance_ratio=imbalance_ratio,
        minority_class=minority_class,
    )


def compute_statistics(splits: Splits) -> SplitStatistics:
    """Compute corpus-wide and per-split statistics (deterministic, sorted)."""

    all_objects: list[UnifiedObject] = []
    per_split: dict[str, DatasetStats] = {}
    for split in SPLIT_ORDER:
        objects = list(splits.get(split.value, ()))
        per_split[split.value] = _stats_for(objects)
        all_objects.extend(objects)
    return SplitStatistics(total=_stats_for(all_objects), per_split=per_split)


def build_manifest(
    *,
    corpus_version: CorpusVersion,
    corpus_hash: str,
    splits: Splits,
    ratios: SplitRatios,
    seed: int,
    grouping: str,
    generated_at: datetime | None = None,
) -> SplitManifest:
    """Assemble the reproducible manifest for one split run."""

    counts: dict[str, SplitCounts] = {}
    total_objects = 0
    total_images: set[str] = set()
    for split in SPLIT_ORDER:
        objects = list(splits.get(split.value, ()))
        images = {o.image_path for o in objects}
        counts[split.value] = SplitCounts(objects=len(objects), images=len(images))
        total_objects += len(objects)
        total_images |= images

    return SplitManifest(
        generator_version=GENERATOR_VERSION,
        generated_at=generated_at,
        seed=seed,
        ratios=ratios,
        grouping=grouping,
        corpus_hash=corpus_hash,
        corpus_version=corpus_version,
        counts=counts,
        total_objects=total_objects,
        total_images=len(total_images),
    )

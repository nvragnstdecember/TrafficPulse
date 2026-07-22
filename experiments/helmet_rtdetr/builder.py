"""Deterministic, leakage-safe split builder + validator + export (H3).

Turns a :class:`~helmet_rtdetr.corpus.UnifiedCorpus` into reproducible
train/val/test splits, a manifest, and statistics. The same corpus + seed +
ratios always produces byte-identical split files.

The algorithm
-------------
1. **Group** every object by the strategy's key (video-aware by default), so a
   group is the indivisible unit of assignment -- this is what prevents leakage.
2. **Shuffle deterministically:** order groups by ``sha256(f"{seed}:{key}")`` (a
   stable, process-independent hash -- never Python's salted ``hash()``), tie-broken
   by the key.
3. **Partition by cumulative object fraction:** walk the ordered groups; each group
   is assigned to the split its *starting* cumulative fraction falls into, so whole
   groups stay together and the object-level ratios are approached as closely as
   indivisible groups allow.
4. **Sort** each split's objects by the H2 content order and **validate** (leakage,
   duplicates, empties, provenance) before returning -- a leaky split can never be
   produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .corpus import UnifiedCorpus, _sort_key
from .errors import (
    DuplicateAnnotationError,
    EmptySplitError,
    InconsistentProvenanceError,
    LeakageError,
)
from .grouping import GroupingStrategy, VideoAwareGrouping
from .manifest import SplitManifest, SplitStatistics, build_manifest, compute_statistics
from .models import _Model
from .split import SPLIT_ORDER, SplitName, SplitRatios
from .unified import UnifiedObject


class SplitResult(_Model):
    """The immutable output of a split run: the objects, the manifest, the stats."""

    splits: dict[str, tuple[UnifiedObject, ...]]
    manifest: SplitManifest
    statistics: SplitStatistics

    def objects(self, split: SplitName) -> tuple[UnifiedObject, ...]:
        return self.splits.get(split.value, ())

    def split_jsonl(self, split: SplitName) -> str:
        """One object per line for ``split``, in deterministic order."""

        return "\n".join(obj.model_dump_json() for obj in self.objects(split))


class SplitValidator:
    """Independently re-checks a set of splits and raises the first typed failure.

    Run automatically by :meth:`SplitBuilder.build`, and exposed so an externally
    produced split can be audited. It recomputes group keys from scratch rather
    than trusting the builder -- defense in depth against a splitting bug.
    """

    def validate(
        self,
        splits: Mapping[str, Sequence[UnifiedObject]],
        *,
        ratios: SplitRatios,
        grouping: GroupingStrategy,
    ) -> None:
        self._check_no_duplicate_objects(splits)
        self._check_provenance_consistency(splits)
        self._check_no_group_leakage(splits, grouping)
        self._check_no_empty_requested_split(splits, ratios)

    @staticmethod
    def _check_no_duplicate_objects(splits: Mapping[str, Sequence[UnifiedObject]]) -> None:
        seen: dict[str, str] = {}  # object_id -> split it first appeared in
        for split in SPLIT_ORDER:
            for obj in splits.get(split.value, ()):
                oid = obj.object_id
                if oid in seen:
                    raise DuplicateAnnotationError(
                        f"object {oid} appears in splits {seen[oid]!r} and {split.value!r}"
                    )
                seen[oid] = split.value

    @staticmethod
    def _check_provenance_consistency(splits: Mapping[str, Sequence[UnifiedObject]]) -> None:
        dataset_by_image: dict[str, str] = {}
        for split in SPLIT_ORDER:
            for obj in splits.get(split.value, ()):
                ds = obj.provenance.dataset_id
                prior = dataset_by_image.get(obj.image_path)
                if prior is not None and prior != ds:
                    raise InconsistentProvenanceError(
                        f"image {obj.image_path!r} is attributed to both {prior!r} and {ds!r}"
                    )
                dataset_by_image[obj.image_path] = ds

    @staticmethod
    def _check_no_group_leakage(
        splits: Mapping[str, Sequence[UnifiedObject]], grouping: GroupingStrategy
    ) -> None:
        group_split: dict[str, str] = {}
        image_split: dict[str, str] = {}
        for split in SPLIT_ORDER:
            for obj in splits.get(split.value, ()):
                key = grouping.group_key(obj)
                if key in group_split and group_split[key] != split.value:
                    raise LeakageError(
                        f"group {key!r} spans splits {group_split[key]!r} and {split.value!r}"
                    )
                group_split[key] = split.value
                if obj.image_path in image_split and image_split[obj.image_path] != split.value:
                    raise LeakageError(
                        f"image {obj.image_path!r} spans splits "
                        f"{image_split[obj.image_path]!r} and {split.value!r}"
                    )
                image_split[obj.image_path] = split.value

    @staticmethod
    def _check_no_empty_requested_split(
        splits: Mapping[str, Sequence[UnifiedObject]], ratios: SplitRatios
    ) -> None:
        for split in SPLIT_ORDER:
            if ratios.requested(split) > 0.0 and not splits.get(split.value):
                raise EmptySplitError(
                    f"split {split.value!r} was requested (ratio "
                    f"{ratios.requested(split)}) but is empty; the corpus has too few "
                    "groups to satisfy the ratios without leakage"
                )


class SplitBuilder:
    """Builds a deterministic, leakage-safe :class:`SplitResult` from a corpus."""

    def __init__(
        self,
        corpus: UnifiedCorpus,
        *,
        ratios: SplitRatios,
        seed: int,
        grouping: GroupingStrategy | None = None,
    ) -> None:
        self._corpus = corpus
        self._ratios = ratios
        self._seed = seed
        self._grouping = grouping if grouping is not None else VideoAwareGrouping()

    def build(self, *, generated_at: datetime | None = None) -> SplitResult:
        objects = self._corpus.objects
        if not objects:
            raise EmptySplitError("cannot split an empty corpus")

        groups = self._grouping.group(objects)
        assignment = self._assign_groups(groups)

        splits: dict[str, list[UnifiedObject]] = {s.value: [] for s in SPLIT_ORDER}
        for key, split in assignment.items():
            splits[split.value].extend(groups[key])
        ordered: dict[str, tuple[UnifiedObject, ...]] = {
            name: tuple(sorted(items, key=_sort_key)) for name, items in splits.items()
        }

        SplitValidator().validate(ordered, ratios=self._ratios, grouping=self._grouping)

        statistics = compute_statistics(ordered)
        manifest = build_manifest(
            corpus_version=self._corpus.corpus_version,
            corpus_hash=self._corpus.content_hash(),
            splits=ordered,
            ratios=self._ratios,
            seed=self._seed,
            grouping=self._grouping.name,
            generated_at=generated_at,
        )
        return SplitResult(splits=ordered, manifest=manifest, statistics=statistics)

    def _assign_groups(self, groups: Mapping[str, list[UnifiedObject]]) -> dict[str, SplitName]:
        total = sum(len(objs) for objs in groups.values())
        ordered_keys = sorted(
            groups, key=lambda k: (hashlib.sha256(f"{self._seed}:{k}".encode()).hexdigest(), k)
        )
        train_end = self._ratios.train
        val_end = self._ratios.train + self._ratios.val

        assignment: dict[str, SplitName] = {}
        cumulative = 0
        for key in ordered_keys:
            fraction = cumulative / total  # fraction BEFORE adding this group
            if fraction < train_end:
                split = SplitName.TRAIN
            elif fraction < val_end:
                split = SplitName.VAL
            else:
                split = SplitName.TEST
            assignment[key] = split
            cumulative += len(groups[key])
        return assignment


def export_splits(result: SplitResult, splits_dir: Path) -> dict[str, Path]:
    """Write ``train/val/test.jsonl`` + ``manifest.json`` + ``statistics.json``.

    Byte-deterministic for a fixed corpus + seed + ratios (and a fixed
    ``manifest.generated_at``). Writes only under ``splits_dir``; downloads nothing.
    Returns the written paths keyed by artifact name.
    """

    splits_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        path = splits_dir / f"{split.value}.jsonl"
        text = result.split_jsonl(split)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        written[split.value] = path

    manifest_path = splits_dir / "manifest.json"
    manifest_path.write_text(
        result.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    written["manifest"] = manifest_path

    statistics_path = splits_dir / "statistics.json"
    statistics_path.write_text(
        result.statistics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    written["statistics"] = statistics_path
    return written

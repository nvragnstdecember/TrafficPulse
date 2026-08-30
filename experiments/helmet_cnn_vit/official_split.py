"""Official HELMET split assignment + independent leakage re-validation (P4-U5).

The HELMET authors ship ``data_split.csv`` -- a video-level
``training``/``validation``/``test`` assignment over all 910 clips. ``dataset-policy``
requires preserving an official split where one exists, and architecture-review §12
requires the split unit to be the source video, so the two agree here: this module
applies the authors' assignment **verbatim** and never re-partitions.

What is reused, and what is not
-------------------------------
The H3 :class:`~helmet_rtdetr.builder.SplitValidator` is reused as-is. It recomputes
group keys from scratch with :class:`~helmet_rtdetr.grouping.VideoAwareGrouping` and
raises on any group or frame that straddles splits -- defence in depth against a bug
in *this* module, exactly as it is for H3's own builder. H3's
:func:`~helmet_rtdetr.manifest.compute_statistics` is likewise reused.

H3's :class:`~helmet_rtdetr.manifest.SplitManifest` is deliberately **not** reused:
it requires a ``seed`` and ``SplitRatios``, and stamping those onto an externally
supplied assignment would describe this split as a seeded ratio partition, which it
is not. :class:`OfficialSplitManifest` below records what actually determined the
assignment -- the source file and its checksum -- plus the realised counts.

Site slices
-----------
The twelve observation sites are **shared across the three official splits**, so this
split measures video-level, not site-level, generalisation. The manifest records the
per-site breakdown of the test split so that limitation is visible in the artifact
rather than only in prose, and so per-site metrics can be reported.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from helmet_rtdetr.builder import SplitValidator
from helmet_rtdetr.grouping import VideoAwareGrouping
from helmet_rtdetr.manifest import SplitStatistics, compute_statistics
from helmet_rtdetr.models import _Model
from helmet_rtdetr.split import SPLIT_ORDER, SplitName, SplitRatios
from helmet_rtdetr.unified import ObjectProvenance, UnifiedClass, UnifiedObject

from .corpus import CropCorpus, CropRecord
from .errors import UnassignedVideoError, UnknownSplitNameError
from .labels import HelmetState

GENERATOR_VERSION = "p4u5-official-split/1.0.0"

#: The dataset id these crops are attributed to (matches registry/datasets/).
DATASET_ID = "helmet-myanmar"
DATASET_VERSION = "2020.0"
ADAPTER_NAME = "helmet-cnnvit-track-csv"

#: The authors' split names, mapped onto the project's canonical SplitName.
OFFICIAL_SPLIT_NAMES: dict[str, SplitName] = {
    "training": SplitName.TRAIN,
    "validation": SplitName.VAL,
    "test": SplitName.TEST,
}

_LABEL_TO_UNIFIED: dict[HelmetState, UnifiedClass] = {
    HelmetState.HELMET: UnifiedClass.HELMET,
    HelmetState.NO_HELMET: UnifiedClass.NO_HELMET,
}


class SplitCropCounts(_Model):
    """Realised counts for one split."""

    videos: int
    tracks: int
    crops: int
    crops_per_class: dict[str, int]
    no_helmet_share: float


class OfficialSplitManifest(_Model):
    """Reproducible provenance for the official split assignment.

    Frozen and committed before the first training run; its :meth:`content_hash`
    is what the results document cites.
    """

    generator_version: str
    dataset_id: str
    dataset_version: str
    source_file: str
    source_sha256: str
    corpus_hash: str
    sampling_policy: dict[str, object]
    counts: dict[str, SplitCropCounts]
    sites: tuple[str, ...]
    test_crops_per_site: dict[str, int]
    total_crops: int

    def content_hash(self) -> str:
        """SHA-256 over the canonical JSON of this manifest."""

        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


def frame_key(video_id: str, frame_index: int) -> str:
    """Stable logical identity of one source frame.

    Used for leakage validation and as the join key between a crop and the frame it
    is cut from. It is deliberately *not* an on-disk path -- where the frame lives
    inside the distributed archives is :mod:`helmet_cnn_vit.extract`'s concern.
    """

    return f"{video_id}/frame_{frame_index:06d}"


def read_official_split(path: Path) -> dict[str, SplitName]:
    """Parse ``data_split.csv`` into ``video_id -> SplitName``.

    Raises :class:`~helmet_cnn_vit.errors.UnknownSplitNameError` for any split name
    outside the authors' three, rather than defaulting it into a split.
    """

    assignment: dict[str, SplitName] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("Set") or "").strip()
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue
            if raw not in OFFICIAL_SPLIT_NAMES:
                raise UnknownSplitNameError(
                    f"{path}: video {video_id!r} has split {raw!r}, "
                    f"expected one of {sorted(OFFICIAL_SPLIT_NAMES)}"
                )
            assignment[video_id] = OFFICIAL_SPLIT_NAMES[raw]
    return assignment


def as_unified(record: CropRecord) -> UnifiedObject:
    """Project a crop onto the H2 schema, so H3's validator can audit it."""

    return UnifiedObject(
        image_path=frame_key(record.video_id, record.frame_index),
        bbox=record.bbox,
        label=_LABEL_TO_UNIFIED[record.driver_state],
        provenance=ObjectProvenance(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            adapter=ADAPTER_NAME,
            source_label=record.source_label,
        ),
        video_id=record.video_id,
        site_id=record.site_id,
        frame_index=record.frame_index,
        frame_id=f"{record.video_id}:{record.frame_index}",
    )


def assign(
    corpus: CropCorpus, assignment: Mapping[str, SplitName]
) -> dict[str, tuple[CropRecord, ...]]:
    """Partition the corpus by the official assignment, preserving corpus order.

    Raises :class:`~helmet_cnn_vit.errors.UnassignedVideoError` if the corpus holds a
    video the official split does not mention -- guessing its split would be exactly
    the leakage the policy forbids.
    """

    unknown = sorted({r.video_id for r in corpus.records} - set(assignment))
    if unknown:
        raise UnassignedVideoError(
            f"{len(unknown)} video(s) absent from the official split: {unknown[:5]}"
        )
    buckets: dict[str, list[CropRecord]] = {s.value: [] for s in SPLIT_ORDER}
    for record in corpus.records:
        buckets[assignment[record.video_id].value].append(record)
    return {name: tuple(items) for name, items in buckets.items()}


def validate_no_leakage(splits: Mapping[str, Sequence[CropRecord]]) -> None:
    """Re-check the assignment with H3's independent validator.

    Ratios are passed as the *realised* fractions purely so the validator's
    empty-split check knows which splits were requested; nothing here partitions by
    ratio. Raises :class:`~helmet_rtdetr.errors.LeakageError` (or a sibling) on any
    violation.
    """

    unified = {
        name: [as_unified(record) for record in records] for name, records in splits.items()
    }
    total = sum(len(v) for v in unified.values()) or 1
    realised = SplitRatios(
        train=len(unified.get(SplitName.TRAIN.value, ())) / total,
        val=len(unified.get(SplitName.VAL.value, ())) / total,
        test=len(unified.get(SplitName.TEST.value, ())) / total,
    )
    SplitValidator().validate(unified, ratios=realised, grouping=VideoAwareGrouping())


def _counts_for(records: Sequence[CropRecord]) -> SplitCropCounts:
    per_class: dict[str, int] = {}
    for record in records:
        key = record.driver_state.value
        per_class[key] = per_class.get(key, 0) + 1
    total = len(records)
    no_helmet = per_class.get(HelmetState.NO_HELMET.value, 0)
    return SplitCropCounts(
        videos=len({r.video_id for r in records}),
        tracks=len({(r.video_id, r.track_id) for r in records}),
        crops=total,
        crops_per_class=dict(sorted(per_class.items())),
        no_helmet_share=(no_helmet / total) if total else 0.0,
    )


def build_official_split(
    corpus: CropCorpus, split_csv: Path
) -> tuple[dict[str, tuple[CropRecord, ...]], OfficialSplitManifest, SplitStatistics]:
    """Assign, validate, and describe the official split in one call.

    Returns the per-split records, the frozen manifest, and H3's statistics.
    """

    assignment = read_official_split(split_csv)
    splits = assign(corpus, assignment)
    validate_no_leakage(splits)

    test_per_site: dict[str, int] = {}
    for record in splits[SplitName.TEST.value]:
        test_per_site[record.site_id] = test_per_site.get(record.site_id, 0) + 1

    manifest = OfficialSplitManifest(
        generator_version=GENERATOR_VERSION,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        source_file=split_csv.name,
        source_sha256=hashlib.sha256(split_csv.read_bytes()).hexdigest(),
        corpus_hash=corpus.content_hash(),
        sampling_policy=corpus.policy.model_dump(),
        counts={name: _counts_for(records) for name, records in sorted(splits.items())},
        sites=tuple(sorted({r.site_id for r in corpus.records})),
        test_crops_per_site=dict(sorted(test_per_site.items())),
        total_crops=len(corpus.records),
    )
    statistics = compute_statistics(
        {name: [as_unified(r) for r in records] for name, records in splits.items()}
    )
    return splits, manifest, statistics

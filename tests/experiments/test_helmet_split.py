"""Deterministic, leakage-safe splitting + validation (H3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helmet_rtdetr.builder import (
    SplitBuilder,
    SplitValidator,
    export_splits,
)
from helmet_rtdetr.corpus import CorpusBuilder, UnifiedCorpus
from helmet_rtdetr.errors import (
    DuplicateAnnotationError,
    EmptySplitError,
    InconsistentProvenanceError,
    InvalidRatioError,
    LeakageError,
)
from helmet_rtdetr.grouping import ImageGrouping, VideoAwareGrouping
from helmet_rtdetr.models import CorpusMember, CorpusVersion
from helmet_rtdetr.split import SplitName, SplitRatios
from helmet_rtdetr.unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject

CV = CorpusVersion(
    corpus_id="helmet-core",
    version="1",
    members=(CorpusMember(dataset_id="helmet-myanmar", dataset_version="1"),),
)
R_80_10_10 = SplitRatios(train=0.8, val=0.1, test=0.1)


def vobj(video: str, frame: int, label: UnifiedClass = UnifiedClass.HELMET) -> UnifiedObject:
    return UnifiedObject(
        image_path=f"{video}/f{frame}.jpg",
        bbox=BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        label=label,
        provenance=ObjectProvenance(
            dataset_id="helmet-myanmar", dataset_version="1", adapter="a", source_label="DHelmet"
        ),
        video_id=video,
        frame_index=frame,
        frame_id=f"{video}:{frame}",
    )


def iobj(image: str, dataset_id: str = "roboflow-moto-helmet") -> UnifiedObject:
    return UnifiedObject(
        image_path=image,
        bbox=BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        label=UnifiedClass.HELMET,
        provenance=ObjectProvenance(
            dataset_id=dataset_id, dataset_version="1", adapter="coco", source_label="With Helmet"
        ),
    )


def video_corpus(n_videos: int, frames: int = 5) -> UnifiedCorpus:
    objs = [vobj(f"vid{v:03d}", f) for v in range(n_videos) for f in range(frames)]
    return CorpusBuilder(CV).add(objs).build()


# --- ratio validation --------------------------------------------------------
def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(InvalidRatioError, match="sum"):
        SplitRatios(train=0.8, val=0.1, test=0.2)


def test_ratios_must_be_in_range() -> None:
    with pytest.raises(InvalidRatioError):
        SplitRatios(train=1.2, val=-0.1, test=-0.1)


def test_train_ratio_must_be_positive() -> None:
    with pytest.raises(InvalidRatioError, match="train"):
        SplitRatios(train=0.0, val=0.5, test=0.5)


def test_two_way_split_is_allowed() -> None:
    ratios = SplitRatios(train=0.8, val=0.2, test=0.0)
    result = SplitBuilder(video_corpus(20), ratios=ratios, seed=1).build()
    assert len(result.objects(SplitName.TEST)) == 0  # test=0 requested, legitimately empty


# --- leakage prevention (highest priority) -----------------------------------
def test_no_video_spans_splits() -> None:
    result = SplitBuilder(video_corpus(50), ratios=R_80_10_10, seed=7).build()
    video_to_split: dict[str, str] = {}
    for split in SplitName:
        for obj in result.objects(split):
            assert obj.video_id is not None
            prior = video_to_split.get(obj.video_id)
            assert prior is None or prior == split.value  # never two splits
            video_to_split[obj.video_id] = split.value
    assert len(video_to_split) == 50  # every video placed exactly once


def test_all_frames_of_a_video_stay_together() -> None:
    result = SplitBuilder(video_corpus(30, frames=6), ratios=R_80_10_10, seed=3).build()
    for split in SplitName:
        videos = {o.video_id for o in result.objects(split)}
        for video in videos:
            frames_here = {o.frame_index for o in result.objects(split) if o.video_id == video}
            assert frames_here == set(range(6))  # all six frames, or none


# --- determinism + reproducibility -------------------------------------------
def test_same_corpus_and_seed_is_byte_identical() -> None:
    corpus = video_corpus(40)
    a = SplitBuilder(corpus, ratios=R_80_10_10, seed=42).build()
    b = SplitBuilder(corpus, ratios=R_80_10_10, seed=42).build()
    for split in SplitName:
        assert a.split_jsonl(split) == b.split_jsonl(split)


def test_different_seed_produces_a_different_split() -> None:
    corpus = video_corpus(40)
    a = SplitBuilder(corpus, ratios=R_80_10_10, seed=1).build()
    b = SplitBuilder(corpus, ratios=R_80_10_10, seed=2).build()
    assert a.split_jsonl(SplitName.TRAIN) != b.split_jsonl(SplitName.TRAIN)


def test_ratios_are_approximately_honoured() -> None:
    result = SplitBuilder(video_corpus(100), ratios=R_80_10_10, seed=5).build()
    n = sum(len(result.objects(s)) for s in SplitName)
    train_frac = len(result.objects(SplitName.TRAIN)) / n
    assert 0.7 <= train_frac <= 0.9  # close to 0.8, bounded by whole-group assignment


# --- image-only corpora ------------------------------------------------------
def test_image_only_corpus_splits_by_image() -> None:
    objs = [iobj(f"img{i:03d}.jpg") for i in range(50)]
    corpus = CorpusBuilder(CV).add(objs).build()
    result = SplitBuilder(corpus, ratios=R_80_10_10, seed=9).build()

    # every image in exactly one split
    image_to_split: dict[str, str] = {}
    for split in SplitName:
        for obj in result.objects(split):
            assert image_to_split.get(obj.image_path) in (None, split.value)
            image_to_split[obj.image_path] = split.value
    assert len(image_to_split) == 50


def test_image_grouping_strategy_is_selectable() -> None:
    objs = [iobj(f"img{i}.jpg") for i in range(30)]
    corpus = CorpusBuilder(CV).add(objs).build()
    result = SplitBuilder(corpus, ratios=R_80_10_10, seed=1, grouping=ImageGrouping()).build()
    assert result.manifest.grouping == "image"


# --- tiny / degenerate corpora ----------------------------------------------
def test_empty_corpus_raises() -> None:
    corpus = UnifiedCorpus(corpus_version=CV, objects=())
    with pytest.raises(EmptySplitError, match="empty corpus"):
        SplitBuilder(corpus, ratios=R_80_10_10, seed=1).build()


def test_too_few_groups_for_requested_split_raises() -> None:
    """Two videos cannot fill an 80/10/10 split without leaving val/test empty."""

    with pytest.raises(EmptySplitError):
        SplitBuilder(video_corpus(2), ratios=R_80_10_10, seed=1).build()


def test_single_group_all_train_when_only_train_requested() -> None:
    result = SplitBuilder(
        video_corpus(1), ratios=SplitRatios(train=1.0, val=0.0, test=0.0), seed=1
    ).build()
    assert len(result.objects(SplitName.TRAIN)) == 5


# --- SplitValidator (independent audit) --------------------------------------
def test_validator_flags_group_leakage() -> None:
    o = vobj("vidA", 0)
    o2 = vobj("vidA", 1)  # same video -> same group
    with pytest.raises(LeakageError, match="spans"):
        SplitValidator().validate(
            {"train": (o,), "val": (o2,), "test": ()},
            ratios=R_80_10_10,
            grouping=VideoAwareGrouping(),
        )


def test_validator_flags_duplicate_objects() -> None:
    o = vobj("vidA", 0)
    with pytest.raises(DuplicateAnnotationError):
        SplitValidator().validate(
            {"train": (o,), "val": (o,), "test": ()},
            ratios=R_80_10_10,
            grouping=VideoAwareGrouping(),
        )


def test_validator_flags_inconsistent_provenance() -> None:
    # Same image path, DIFFERENT boxes (so distinct object_ids, not duplicates),
    # but attributed to two different datasets -- a genuine provenance conflict.
    a = UnifiedObject(
        image_path="shared.jpg",
        bbox=BBox(x=1.0, y=1.0, w=2.0, h=2.0),
        label=UnifiedClass.HELMET,
        provenance=ObjectProvenance(
            dataset_id="dataset-a", dataset_version="1", adapter="coco", source_label="With Helmet"
        ),
    )
    b = UnifiedObject(
        image_path="shared.jpg",
        bbox=BBox(x=9.0, y=9.0, w=2.0, h=2.0),
        label=UnifiedClass.HELMET,
        provenance=ObjectProvenance(
            dataset_id="dataset-b", dataset_version="1", adapter="coco", source_label="With Helmet"
        ),
    )
    with pytest.raises(InconsistentProvenanceError):
        SplitValidator().validate(
            {"train": (a,), "val": (b,), "test": ()},
            ratios=R_80_10_10,
            grouping=VideoAwareGrouping(),
        )


def test_validator_flags_empty_requested_split() -> None:
    o = vobj("vidA", 0)
    with pytest.raises(EmptySplitError):
        SplitValidator().validate(
            {"train": (o,), "val": (), "test": ()},
            ratios=R_80_10_10,  # val + test requested but empty
            grouping=VideoAwareGrouping(),
        )


def test_validator_flags_image_leakage_across_distinct_groups() -> None:
    # Same image path, same dataset, DIFFERENT video -> distinct group keys (so the
    # group check passes) and distinct boxes (so not duplicates), yet the image
    # itself lands in two splits: the image-level leakage guard must catch it.
    a = UnifiedObject(
        image_path="shared.jpg",
        bbox=BBox(x=1.0, y=1.0, w=2.0, h=2.0),
        label=UnifiedClass.HELMET,
        provenance=ObjectProvenance(
            dataset_id="helmet-myanmar", dataset_version="1", adapter="a", source_label="DHelmet"
        ),
        video_id="vidA",
        frame_index=0,
        frame_id="vidA:0",
    )
    b = a.model_copy(
        update={"bbox": BBox(x=9.0, y=9.0, w=2.0, h=2.0), "video_id": "vidB", "frame_id": "vidB:0"}
    )
    with pytest.raises(LeakageError, match="image"):
        SplitValidator().validate(
            {"train": (a,), "val": (b,), "test": ()},
            ratios=R_80_10_10,
            grouping=VideoAwareGrouping(),
        )


def test_split_ratios_as_dict() -> None:
    assert R_80_10_10.as_dict() == {"train": 0.8, "val": 0.1, "test": 0.1}


def test_validator_passes_a_clean_split() -> None:
    result = SplitBuilder(video_corpus(50), ratios=R_80_10_10, seed=1).build()
    SplitValidator().validate(
        {s.value: result.objects(s) for s in SplitName},
        ratios=R_80_10_10,
        grouping=VideoAwareGrouping(),
    )  # must not raise


# --- export ------------------------------------------------------------------
def test_export_writes_all_artifacts(tmp_path: Path) -> None:
    result = SplitBuilder(video_corpus(40), ratios=R_80_10_10, seed=1).build()
    written = export_splits(result, tmp_path / "splits")

    for name in ("train", "val", "test", "manifest", "statistics"):
        assert written[name].is_file()
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert (tmp_path / "splits" / "manifest.json").exists()
    assert (tmp_path / "splits" / "statistics.json").exists()


def test_export_is_byte_identical_for_identical_input(tmp_path: Path) -> None:
    corpus = video_corpus(40)
    export_splits(SplitBuilder(corpus, ratios=R_80_10_10, seed=1).build(), tmp_path / "a")
    export_splits(SplitBuilder(corpus, ratios=R_80_10_10, seed=1).build(), tmp_path / "b")

    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "statistics.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_exported_split_line_counts_match_objects(tmp_path: Path) -> None:
    result = SplitBuilder(video_corpus(40), ratios=R_80_10_10, seed=1).build()
    export_splits(result, tmp_path / "s")
    for split in SplitName:
        lines = [
            ln
            for ln in (tmp_path / "s" / f"{split.value}.jsonl").read_text("utf-8").splitlines()
            if ln
        ]
        assert len(lines) == len(result.objects(split))

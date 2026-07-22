"""Split manifest + statistics (H3)."""

from __future__ import annotations

from datetime import UTC, datetime

from helmet_rtdetr.builder import SplitBuilder
from helmet_rtdetr.corpus import CorpusBuilder, UnifiedCorpus
from helmet_rtdetr.manifest import GENERATOR_VERSION, compute_statistics
from helmet_rtdetr.models import CorpusMember, CorpusVersion
from helmet_rtdetr.split import SplitName, SplitRatios
from helmet_rtdetr.unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject

CV = CorpusVersion(
    corpus_id="helmet-core",
    version="1",
    members=(CorpusMember(dataset_id="helmet-myanmar", dataset_version="1"),),
)
R = SplitRatios(train=0.8, val=0.1, test=0.1)


def vobj(
    video: str, frame: int, label: UnifiedClass, dataset_id: str = "helmet-myanmar"
) -> UnifiedObject:
    return UnifiedObject(
        image_path=f"{video}/f{frame}.jpg",
        bbox=BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        label=label,
        provenance=ObjectProvenance(
            dataset_id=dataset_id, dataset_version="1", adapter="a", source_label="x"
        ),
        video_id=video,
        frame_index=frame,
        frame_id=f"{video}:{frame}",
    )


def corpus(n_videos: int = 40) -> UnifiedCorpus:
    objs = [
        vobj(f"vid{v:03d}", f, UnifiedClass.NO_HELMET if v % 4 == 0 else UnifiedClass.HELMET)
        for v in range(n_videos)
        for f in range(5)
    ]
    return CorpusBuilder(CV).add(objs).build()


# --- manifest ----------------------------------------------------------------
def test_manifest_records_run_parameters() -> None:
    manifest = SplitBuilder(corpus(), ratios=R, seed=13).build().manifest
    assert manifest.seed == 13
    assert manifest.ratios == R
    assert manifest.grouping == "video-aware"
    assert manifest.generator_version == GENERATOR_VERSION


def test_manifest_records_corpus_provenance() -> None:
    c = corpus()
    manifest = SplitBuilder(c, ratios=R, seed=1).build().manifest
    assert manifest.corpus_hash == c.content_hash()
    assert manifest.corpus_version == c.corpus_version
    assert manifest.corpus_version.members[0].dataset_id == "helmet-myanmar"


def test_manifest_counts_match_splits() -> None:
    result = SplitBuilder(corpus(), ratios=R, seed=1).build()
    for split in SplitName:
        assert result.manifest.counts[split.value].objects == len(result.objects(split))
    assert result.manifest.total_objects == 200


def test_manifest_generated_at_defaults_to_none_for_determinism() -> None:
    assert SplitBuilder(corpus(), ratios=R, seed=1).build().manifest.generated_at is None


def test_manifest_generated_at_is_injectable() -> None:
    ts = datetime(2026, 7, 17, tzinfo=UTC)
    manifest = SplitBuilder(corpus(), ratios=R, seed=1).build(generated_at=ts).manifest
    assert manifest.generated_at == ts


def test_manifest_is_byte_identical_with_same_injected_timestamp() -> None:
    ts = datetime(2026, 7, 17, tzinfo=UTC)
    c = corpus()
    a = SplitBuilder(c, ratios=R, seed=1).build(generated_at=ts).manifest
    b = SplitBuilder(c, ratios=R, seed=1).build(generated_at=ts).manifest
    assert a.model_dump_json() == b.model_dump_json()


# --- statistics --------------------------------------------------------------
def test_statistics_total_object_and_image_counts() -> None:
    stats = SplitBuilder(corpus(), ratios=R, seed=1).build().statistics
    assert stats.total.objects == 200
    assert stats.total.images == 200  # one object per synthetic frame image


def test_statistics_per_class_counts() -> None:
    # 40 videos: 10 are no_helmet (v%4==0), 30 helmet; 5 frames each.
    stats = SplitBuilder(corpus(), ratios=R, seed=1).build().statistics
    assert stats.total.per_class == {"helmet": 150, "no_helmet": 50}


def test_statistics_imbalance_ratio() -> None:
    stats = SplitBuilder(corpus(), ratios=R, seed=1).build().statistics
    assert stats.total.imbalance_ratio == 3.0  # 150 / 50
    assert stats.total.minority_class == "no_helmet"


def test_statistics_per_dataset_counts() -> None:
    objs = [vobj("vidA", 0, UnifiedClass.HELMET, dataset_id="helmet-myanmar")]
    objs += [vobj("vidB", 0, UnifiedClass.HELMET, dataset_id="helmet-myanmar")]
    stats = compute_statistics({"train": objs, "val": (), "test": ()})
    assert stats.total.per_dataset == {"helmet-myanmar": 2}


def test_statistics_cover_every_split_in_order() -> None:
    stats = SplitBuilder(corpus(), ratios=R, seed=1).build().statistics
    assert list(stats.per_split.keys()) == ["train", "val", "test"]
    summed = sum(stats.per_split[s].objects for s in ("train", "val", "test"))
    assert summed == stats.total.objects


def test_statistics_single_class_has_no_imbalance_ratio() -> None:
    objs = [vobj("vidA", i, UnifiedClass.HELMET) for i in range(3)]
    stats = compute_statistics({"train": objs, "val": (), "test": ()})
    assert stats.total.imbalance_ratio is None  # only one class present
    assert stats.total.minority_class == "helmet"


def test_empty_split_statistics_are_zero() -> None:
    stats = compute_statistics({"train": (), "val": (), "test": ()})
    assert stats.total.objects == 0
    assert stats.total.per_class == {}
    assert stats.total.imbalance_ratio is None
    assert stats.total.minority_class is None


def test_statistics_are_deterministic() -> None:
    c = corpus()
    a = SplitBuilder(c, ratios=R, seed=1).build().statistics
    b = SplitBuilder(c, ratios=R, seed=1).build().statistics
    assert a.model_dump_json() == b.model_dump_json()

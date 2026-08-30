"""Crop-corpus construction under the frozen sampling policy (P4-U5).

The policy decides which annotation rows become training examples, so it decides
the class balance. These tests pin its arithmetic on fixtures small enough to count
by hand, and pin the properties the experiment's fairness rests on: determinism,
and that nothing is dropped without being counted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cnnvit_helpers import (
    PAIR_NO_HELMET,
    SOLO_HELMET,
    tiny_corpus_fixture,
    track_rows,
    write_annotation,
)
from helmet_cnn_vit.corpus import (
    CropRecord,
    SamplingPolicy,
    build_corpus,
    select_frames,
    site_of,
)
from helmet_cnn_vit.errors import InconsistentTrackLabelError, MissingAnnotationError
from helmet_cnn_vit.labels import HelmetState

# --- the sampling arithmetic -----------------------------------------------------


def test_the_default_policy_keeps_every_fifth_frame_from_one() -> None:
    policy = SamplingPolicy()
    kept = [f for f in range(1, 101) if policy.selects_frame(f)]
    assert kept[:4] == [1, 6, 11, 16]
    assert kept[-1] == 96
    assert len(kept) == 20


def test_a_track_shorter_than_the_cap_keeps_all_its_frames() -> None:
    assert select_frames([1, 6, 11], SamplingPolicy()) == (1, 6, 11)


def test_a_long_track_is_capped_and_spread_across_its_trajectory() -> None:
    """Evenly spaced, not the first six -- crops must not cluster at the track start."""

    frames = list(range(1, 101, 5))  # the 20 stride-surviving frames
    chosen = select_frames(frames, SamplingPolicy(max_crops_per_track=6))
    assert len(chosen) == 6
    assert chosen == (1, 16, 31, 51, 66, 81)
    assert chosen[-1] - chosen[0] > 60  # spans most of the clip


def test_frame_selection_is_pure_and_repeatable() -> None:
    frames = list(range(1, 101, 5))
    policy = SamplingPolicy()
    assert select_frames(frames, policy) == select_frames(frames, policy)


# --- site derivation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("video_id", "site"),
    [
        ("Mandalay_1_28", "Mandalay_1"),
        ("Bago_highway_7", "Bago_highway"),
        ("Yangon_II_3", "Yangon_II"),
        ("Pathein_rural_12", "Pathein_rural"),
    ],
)
def test_site_is_the_clip_id_minus_its_index(video_id: str, site: str) -> None:
    assert site_of(video_id) == site


def test_a_clip_id_without_a_numeric_index_is_refused() -> None:
    with pytest.raises(ValueError, match="not of the form"):
        site_of("Mandalay")


# --- corpus construction ------------------------------------------------------------


def test_the_tiny_fixture_yields_the_hand_counted_corpus(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    corpus = build_corpus(annotation_dir)

    # 4 tracks x frames {1, 6, 11, 16} = 16 crops.
    assert corpus.statistics.videos == 3
    assert corpus.statistics.tracks == 4
    assert len(corpus) == 16
    assert corpus.statistics.crops_per_class == {"helmet": 8, "no_helmet": 8}
    assert corpus.statistics.tracks_per_class == {"helmet": 2, "no_helmet": 2}
    assert corpus.statistics.tracks_per_rider_count == {1: 2, 2: 2}


def test_every_row_is_either_a_crop_or_a_counted_exclusion(tmp_path: Path) -> None:
    """Nothing may vanish silently: the arithmetic over the fixture must close."""

    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    stats = build_corpus(annotation_dir).statistics
    # 4 tracks x 20 frames = 80 rows; 16 survive the stride, 64 do not.
    assert stats.rows_read == 80
    assert stats.rows_excluded_by_stride == 64
    assert stats.rows_read - stats.rows_excluded_by_stride == stats.crops


def test_the_corpus_is_deterministic(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    first = build_corpus(annotation_dir)
    second = build_corpus(annotation_dir)
    assert first.content_hash() == second.content_hash()
    assert [r.crop_id for r in first.records] == [r.crop_id for r in second.records]


def test_the_content_hash_tracks_the_policy(tmp_path: Path) -> None:
    """A different sampling policy is a different corpus, even at the same size."""

    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    default = build_corpus(annotation_dir)
    shifted = build_corpus(annotation_dir, policy=SamplingPolicy(frame_offset=2))
    assert default.content_hash() != shifted.content_hash()


def test_records_are_ordered_by_video_then_track_then_frame(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    records = build_corpus(annotation_dir).records
    keys = [(r.video_id, r.track_id, r.frame_index) for r in records]
    assert keys == sorted(keys)


def test_the_driver_label_is_carried_onto_every_crop(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    by_track: dict[str, list[CropRecord]] = {}
    for record in build_corpus(annotation_dir).records:
        by_track.setdefault(record.track_id, []).append(record)

    assert all(r.driver_state is HelmetState.HELMET for r in by_track["t1"])
    assert all(r.driver_state is HelmetState.NO_HELMET for r in by_track["t2"])
    assert all(r.rider_count == 2 for r in by_track["t2"])
    assert all(r.any_no_helmet for r in by_track["t2"])
    assert not any(r.any_no_helmet for r in by_track["t1"])


def test_crop_ids_are_unique_and_content_derived(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    records = build_corpus(annotation_dir).records
    assert len({r.crop_id for r in records}) == len(records)


# --- the size floor ------------------------------------------------------------------


def test_boxes_below_the_size_floor_are_excluded_and_counted(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir, "Alpha_1", track_rows("tiny", SOLO_HELMET, range(1, 21), w=8.0, h=8.0)
    )
    corpus = build_corpus(annotation_dir)
    assert len(corpus) == 0
    assert corpus.statistics.tracks == 1
    assert corpus.statistics.tracks_excluded_small_box == 1
    # The track still contributes to the label census, so the exclusion is visible.
    assert corpus.statistics.tracks_per_class == {"helmet": 1}


def test_the_size_floor_is_on_the_shorter_side(tmp_path: Path) -> None:
    """A tall sliver is as unusable as a small square."""

    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir, "Alpha_1", track_rows("sliver", SOLO_HELMET, [1], w=4.0, h=400.0)
    )
    assert len(build_corpus(annotation_dir)) == 0


# --- fault handling --------------------------------------------------------------------


def test_a_track_with_two_labels_is_a_hard_error(tmp_path: Path) -> None:
    """HELMET labels a configuration per track; a track that changes is a data fault."""

    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir,
        "Alpha_1",
        [
            *track_rows("t1", SOLO_HELMET, [1, 6]),
            *track_rows("t1", PAIR_NO_HELMET, [11]),
        ],
    )
    with pytest.raises(InconsistentTrackLabelError, match="t1"):
        build_corpus(annotation_dir)


def test_a_missing_annotation_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(MissingAnnotationError):
        build_corpus(tmp_path / "nope")


def test_a_csv_missing_a_required_column_is_reported(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotation"
    annotation_dir.mkdir()
    (annotation_dir / "Alpha_1.csv").write_text("track_id,frame_id,x,y,w\n", encoding="utf-8")
    with pytest.raises(MissingAnnotationError, match="missing columns"):
        build_corpus(annotation_dir)


def test_requesting_an_absent_video_is_reported(tmp_path: Path) -> None:
    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    with pytest.raises(MissingAnnotationError, match="Ghost_1"):
        build_corpus(annotation_dir, video_ids=["Alpha_site_1", "Ghost_1"])

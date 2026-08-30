"""Official split assignment and leakage safety (P4-U5).

Leakage is the failure this experiment most needs to rule out: adjacent frames of
one tracked motorcycle are near-identical, so a frame-level split would leak the
test set into training and inflate both models' scores.

These tests do two separate things, and the second matters as much as the first:
they check that the official assignment survives validation, **and** that the
validator actually rejects a planted violation. A safety net that never fires is
indistinguishable from no safety net.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cnnvit_helpers import SOLO_HELMET, tiny_corpus_fixture, track_rows, write_annotation
from helmet_cnn_vit.corpus import build_corpus
from helmet_cnn_vit.errors import UnassignedVideoError, UnknownSplitNameError
from helmet_cnn_vit.official_split import (
    OFFICIAL_SPLIT_NAMES,
    build_official_split,
    read_official_split,
    validate_no_leakage,
)
from helmet_rtdetr.errors import LeakageError
from helmet_rtdetr.split import SplitName

# --- reading the authors' file ------------------------------------------------------


def test_the_three_official_split_names_map_onto_the_project_names() -> None:
    assert OFFICIAL_SPLIT_NAMES == {
        "training": SplitName.TRAIN,
        "validation": SplitName.VAL,
        "test": SplitName.TEST,
    }


def test_the_official_split_file_is_read_verbatim(tmp_path: Path) -> None:
    _, split_csv = tiny_corpus_fixture(tmp_path)
    assignment = read_official_split(split_csv)
    assert assignment == {
        "Alpha_site_1": SplitName.TRAIN,
        "Alpha_site_2": SplitName.VAL,
        "Beta_site_1": SplitName.TEST,
    }


def test_an_unrecognised_split_name_is_refused_not_defaulted(tmp_path: Path) -> None:
    path = tmp_path / "data_split.csv"
    path.write_text("video_id,Set\nAlpha_1,holdout\n", encoding="utf-8")
    with pytest.raises(UnknownSplitNameError, match="holdout"):
        read_official_split(path)


# --- assignment ------------------------------------------------------------------------


def test_the_official_assignment_is_applied_and_validated(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    corpus = build_corpus(annotation_dir)
    splits, manifest, _ = build_official_split(corpus, split_csv)

    assert manifest.counts["train"].crops == 8  # two tracks x four frames
    assert manifest.counts["val"].crops == 4
    assert manifest.counts["test"].crops == 4
    assert manifest.total_crops == len(corpus)
    assert sum(len(v) for v in splits.values()) == len(corpus)


def test_no_video_appears_in_two_splits(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            assert seen.setdefault(record.video_id, name) == name


def test_no_track_spans_two_splits(tmp_path: Path) -> None:
    """The §12 requirement stated directly, independent of the group-key machinery."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    seen: dict[tuple[str, str], str] = {}
    for name, records in splits.items():
        for record in records:
            key = (record.video_id, record.track_id)
            assert seen.setdefault(key, name) == name


def test_a_video_absent_from_the_official_split_is_refused(tmp_path: Path) -> None:
    """Guessing its split is precisely the leakage the policy forbids."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    write_annotation(annotation_dir, "Gamma_site_9", track_rows("t9", SOLO_HELMET, [1, 6]))
    with pytest.raises(UnassignedVideoError, match="Gamma_site_9"):
        build_official_split(build_corpus(annotation_dir), split_csv)


# --- the safety net must actually fire ----------------------------------------------------


def test_the_validator_rejects_a_video_planted_in_two_splits(tmp_path: Path) -> None:
    """Defence in depth: H3's validator recomputes group keys from scratch."""

    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    records = build_corpus(annotation_dir).records
    alpha = [r for r in records if r.video_id == "Alpha_site_1"]
    assert len(alpha) >= 2

    leaky = {
        "train": tuple(alpha[:1]),
        "val": (),
        "test": tuple(alpha[1:]),  # same video, other split
    }
    with pytest.raises(LeakageError, match="Alpha_site_1"):
        validate_no_leakage(leaky)


def test_the_validator_rejects_one_frame_split_across_splits(tmp_path: Path) -> None:
    """Two crops from the same frame must not straddle splits either."""

    annotation_dir, _ = tiny_corpus_fixture(tmp_path)
    records = build_corpus(annotation_dir).records
    same_frame = [
        r for r in records if r.video_id == "Alpha_site_1" and r.frame_index == 1
    ]
    assert len(same_frame) == 2  # tracks t1 and t2 both appear in frame 1

    with pytest.raises(LeakageError):
        validate_no_leakage({"train": (same_frame[0],), "val": (), "test": (same_frame[1],)})


def test_a_clean_assignment_passes_the_validator(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    validate_no_leakage(splits)  # must not raise


# --- manifest provenance ----------------------------------------------------------------


def test_the_manifest_pins_the_split_file_by_checksum(tmp_path: Path) -> None:
    """The assignment's authority is the authors' file, so it is recorded by hash."""

    import hashlib

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    _, manifest, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    assert manifest.source_sha256 == hashlib.sha256(split_csv.read_bytes()).hexdigest()
    assert manifest.source_file == "data_split.csv"


def test_the_manifest_records_the_sampling_policy(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    corpus = build_corpus(annotation_dir)
    _, manifest, _ = build_official_split(corpus, split_csv)
    assert manifest.sampling_policy["frame_stride"] == 5
    assert manifest.sampling_policy["max_crops_per_track"] == 6
    assert manifest.corpus_hash == corpus.content_hash()


def test_the_manifest_hash_is_deterministic(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    corpus = build_corpus(annotation_dir)
    first = build_official_split(corpus, split_csv)[1]
    second = build_official_split(corpus, split_csv)[1]
    assert first.content_hash() == second.content_hash()


def test_the_manifest_records_the_per_site_test_breakdown(tmp_path: Path) -> None:
    """Sites are shared across the official split, so per-site slices are the
    substitute for a whole-site holdout and must be reported."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    _, manifest, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    assert manifest.sites == ("Alpha_site", "Beta_site")
    assert manifest.test_crops_per_site == {"Beta_site": 4}

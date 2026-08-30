"""Crop harvesting from the HELMET archives (P4-U5).

Pillow-gated: the extractor is the one part of the experiment that touches images,
so these tests skip when Pillow is absent (as it is in a bare install) exactly as
the overlay tests do. The archive-layout tests below need no image library and run
everywhere.

They build real zip archives in a temp directory rather than mocking ``zipfile``,
because the bug this module actually shipped was a filename-padding assumption --
precisely the kind a mock would have reproduced faithfully and hidden.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from _cnnvit_helpers import SOLO_HELMET, tiny_corpus_fixture, track_rows, write_annotation
from helmet_cnn_vit.corpus import build_corpus
from helmet_cnn_vit.errors import CorpusBuildError
from helmet_cnn_vit.extract import (
    ExtractionConfig,
    build_frame_index,
    crop_box,
    extract_crops,
    load_crop_index,
    member_name,
)
from helmet_cnn_vit.official_split import build_official_split

PIL = pytest.importorskip("PIL", reason="Pillow is the optional overlay/extract backend")


FRAME_W, FRAME_H = 640, 480


def make_archive(image_dir: Path, part: str, videos: dict[str, int]) -> Path:
    """A real zip in the archive's real layout: ``part_N/<video>/<NN>.jpg``."""

    from PIL import Image

    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{part}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for video_id, frames in videos.items():
            archive.writestr(f"{part}/{video_id}/", "")
            for frame in range(1, frames + 1):
                buffer = image_dir / "_scratch.jpg"
                Image.new("RGB", (FRAME_W, FRAME_H), (frame % 256, 40, 90)).save(buffer)
                archive.write(buffer, f"{part}/{video_id}/{frame:02d}.jpg")
                buffer.unlink()
    return path


# --- the padding bug this module shipped once -------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [(1, "01.jpg"), (7, "07.jpg"), (9, "09.jpg"), (10, "10.jpg"), (96, "96.jpg"), (100, "100.jpg")],
)
def test_frame_members_are_zero_padded_to_two_digits(frame: int, expected: str) -> None:
    """Frames 1-9 are ``01.jpg``..``09.jpg``; an unpadded name silently misses them."""

    assert member_name("part_1/Alpha_1", frame) == f"part_1/Alpha_1/{expected}"


# --- the frame index -----------------------------------------------------------------


def test_the_frame_index_maps_clips_to_their_archive(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_1": 3})
    make_archive(image_dir, "part_2", {"Beta_1": 3})
    index = build_frame_index(image_dir)
    assert index["Alpha_1"] == ("part_1.zip", "part_1/Alpha_1")
    assert index["Beta_1"] == ("part_2.zip", "part_2/Beta_1")


def test_a_clip_in_two_archives_is_a_hard_error(tmp_path: Path) -> None:
    """Otherwise the provenance of a crop would be ambiguous."""

    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_1": 2})
    make_archive(image_dir, "part_2", {"Alpha_1": 2})
    with pytest.raises(CorpusBuildError, match="appears in both"):
        build_frame_index(image_dir)


def test_a_truncated_archive_is_reported_as_such(tmp_path: Path) -> None:
    """The likeliest operational failure: an interrupted 4.7 GB download."""

    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "part_1.zip").write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(CorpusBuildError, match="not a readable zip"):
        build_frame_index(image_dir)


def test_an_empty_image_directory_is_reported(tmp_path: Path) -> None:
    (tmp_path / "image").mkdir()
    with pytest.raises(CorpusBuildError, match="no part_"):
        build_frame_index(tmp_path / "image")


# --- crop geometry ----------------------------------------------------------------------


def test_a_box_running_off_the_frame_is_clamped(tmp_path: Path) -> None:
    """PIL would pad an out-of-frame crop with black, which looks like real pixels."""

    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir,
        "Alpha_1",
        track_rows("t", SOLO_HELMET, [1], x=600.0, y=440.0, w=200.0, h=200.0),
    )
    record = build_corpus(annotation_dir).records[0]
    assert crop_box(record, (FRAME_W, FRAME_H), ExtractionConfig()) == (600, 440, 640, 480)


def test_a_box_entirely_outside_the_frame_is_refused(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir, "Alpha_1", track_rows("t", SOLO_HELMET, [1], x=9000.0, y=9000.0)
    )
    record = build_corpus(annotation_dir).records[0]
    with pytest.raises(CorpusBuildError, match="empty after clamping"):
        crop_box(record, (FRAME_W, FRAME_H), ExtractionConfig())


def test_context_expansion_grows_the_box_symmetrically(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotation"
    write_annotation(
        annotation_dir,
        "Alpha_1",
        track_rows("t", SOLO_HELMET, [1], x=200.0, y=200.0, w=100.0, h=100.0),
    )
    record = build_corpus(annotation_dir).records[0]
    assert crop_box(record, (FRAME_W, FRAME_H), ExtractionConfig(context_fraction=0.2)) == (
        190,
        190,
        310,
        310,
    )


def test_crops_are_square_padded_not_stretched(tmp_path: Path) -> None:
    """Aspect ratio must survive: a stretched crop is an artefact the model can learn."""

    from helmet_cnn_vit.extract import square_pad_resize
    from PIL import Image

    tall = Image.new("RGB", (40, 120), (255, 0, 0))
    out = square_pad_resize(tall, ExtractionConfig(image_size=64, pad_rgb=(0, 0, 0)))
    assert out.size == (64, 64)
    # The pad is black at the left edge, the content red down the middle.
    assert out.getpixel((2, 32)) == (0, 0, 0)
    assert out.getpixel((32, 32))[0] > 200


# --- end to end over a real archive ----------------------------------------------------------


def test_every_crop_is_written_and_indexed(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_site_1": 20, "Alpha_site_2": 20, "Beta_site_1": 20})

    corpus = build_corpus(annotation_dir)
    splits, _, _ = build_official_split(corpus, split_csv)
    out = tmp_path / "crops"
    report = extract_crops(splits, image_dir=image_dir, output_dir=out)

    assert report.crops_written == len(corpus) == 16
    assert report.missing_frames == ()
    assert report.per_split == {"test": 4, "train": 8, "val": 4}
    assert report.per_class == {"helmet": 8, "no_helmet": 8}
    # Each 1080p-equivalent frame is decoded once, not once per motorcycle.
    assert report.frames_read == 12  # 3 videos x 4 sampled frames


def test_crops_land_in_split_and_class_directories(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_site_1": 20, "Alpha_site_2": 20, "Beta_site_1": 20})
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    out = tmp_path / "crops"
    extract_crops(splits, image_dir=image_dir, output_dir=out)

    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["test", "train", "val"]
    assert (out / "train" / "helmet").is_dir()
    assert (out / "train" / "no_helmet").is_dir()
    assert len(list((out / "test").rglob("*.jpg"))) == 4


def test_the_crop_index_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_site_1": 20, "Alpha_site_2": 20, "Beta_site_1": 20})
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    out = tmp_path / "crops"
    extract_crops(splits, image_dir=image_dir, output_dir=out)

    rows = load_crop_index(out)
    assert len(rows) == 16
    assert [r["crop_id"] for r in rows] == sorted(r["crop_id"] for r in rows)
    row = rows[0]
    assert set(row) >= {"crop_id", "split", "label", "site_id", "rider_count", "path", "box_h"}
    assert (out / row["path"]).is_file()


def test_a_clip_absent_from_the_archives_is_reported_not_skipped(tmp_path: Path) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    image_dir = tmp_path / "image"
    make_archive(image_dir, "part_1", {"Alpha_site_1": 20})  # the other two are absent
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    report = extract_crops(splits, image_dir=image_dir, output_dir=tmp_path / "crops")

    assert report.crops_written == 8
    assert "Alpha_site_2/*" in report.missing_frames
    assert "Beta_site_1/*" in report.missing_frames


def test_reading_an_index_that_was_never_written_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusBuildError, match="run extraction first"):
        load_crop_index(tmp_path)

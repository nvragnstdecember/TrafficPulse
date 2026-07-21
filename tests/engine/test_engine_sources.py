"""Frame sources: file replay, live adapter discipline, identity stamping (H6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _engine_helpers import frame_records
from _slice_fixtures import FRAME_COUNT, write_wrong_way_clip

from trafficpulse.engine import (
    FileFrameSource,
    FrameSource,
    FrameSourceError,
    IterableFrameSource,
    frame_record_from_array,
)
from trafficpulse.ingestion.video import SourceNotFoundError


# --- file source ----------------------------------------------------------------
def test_file_source_validates_eagerly(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        FileFrameSource(tmp_path / "missing.mp4")


def test_file_source_yields_the_ingestion_stream(tmp_path: Path) -> None:
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    source = FileFrameSource(clip, camera_id="cam-a")
    records = list(source.frames())
    assert len(records) == FRAME_COUNT
    assert [r.frame_index for r in records] == list(range(FRAME_COUNT))
    assert all(r.camera_id == "cam-a" for r in records)
    assert all(r.source_id == source.source_id for r in records)


def test_file_source_replays_identically(tmp_path: Path) -> None:
    source = FileFrameSource(write_wrong_way_clip(tmp_path / "clip.mp4"))
    first = [(r.frame_id, r.timestamp_seconds) for r in source.frames()]
    second = [(r.frame_id, r.timestamp_seconds) for r in source.frames()]
    assert first == second  # a fresh reader per frames() call: deterministic replay


def test_file_source_satisfies_the_protocol(tmp_path: Path) -> None:
    source = FileFrameSource(write_wrong_way_clip(tmp_path / "clip.mp4"))
    assert isinstance(source, FrameSource)


# --- live adapter source -----------------------------------------------------------
def test_iterable_source_yields_in_order() -> None:
    records = frame_records(5)
    source = IterableFrameSource(records, source_id="vsrc-live")
    assert [r.frame_index for r in source.frames()] == [0, 1, 2, 3, 4]


def test_iterable_source_is_single_shot() -> None:
    source = IterableFrameSource(frame_records(2), source_id="vsrc-live")
    list(source.frames())
    with pytest.raises(FrameSourceError, match="single-shot"):
        source.frames()


def test_iterable_source_rejects_non_ascending_frame_index() -> None:
    records = frame_records(3)
    shuffled = [records[0], records[2], records[1]]
    source = IterableFrameSource(shuffled, source_id="vsrc-live")
    with pytest.raises(FrameSourceError, match="strictly ascending"):
        list(source.frames())


def test_iterable_source_rejects_decreasing_timestamps() -> None:
    from dataclasses import replace

    records = frame_records(2)
    bad = [records[1], replace(records[0], frame_index=9)]  # index ascends, time reverses
    source = IterableFrameSource(bad, source_id="vsrc-live")
    with pytest.raises(FrameSourceError, match="non-decreasing"):
        list(source.frames())


def test_iterable_source_requires_source_id() -> None:
    with pytest.raises(FrameSourceError, match="source_id"):
        IterableFrameSource((), source_id="")


# --- live identity stamping ----------------------------------------------------------
def test_frame_record_from_array_is_deterministic() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    first = frame_record_from_array(
        image, source_id="vsrc-cam0", frame_index=7, timestamp_seconds=0.7
    )
    second = frame_record_from_array(
        image, source_id="vsrc-cam0", frame_index=7, timestamp_seconds=0.7
    )
    assert first.frame_id == second.frame_id
    assert first.frame_id.startswith("vfrm-")
    assert (first.width, first.height) == (32, 24)


def test_frame_record_from_array_distinct_ids_per_index() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    ids = {
        frame_record_from_array(
            image, source_id="vsrc-cam0", frame_index=i, timestamp_seconds=float(i)
        ).frame_id
        for i in range(4)
    }
    assert len(ids) == 4


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((8, 8), dtype=np.uint8), "shape"),
        (np.zeros((8, 8, 4), dtype=np.uint8), "shape"),
        (np.zeros((8, 8, 3), dtype=np.float32), "RGB uint8"),
    ],
)
def test_frame_record_from_array_rejects_bad_images(
    image: np.ndarray, message: str
) -> None:
    with pytest.raises(FrameSourceError, match=message):
        frame_record_from_array(
            image, source_id="vsrc-cam0", frame_index=0, timestamp_seconds=0.0
        )


def test_frame_record_from_array_rejects_negative_index_and_time() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(FrameSourceError, match="frame_index"):
        frame_record_from_array(
            image, source_id="s", frame_index=-1, timestamp_seconds=0.0
        )
    with pytest.raises(FrameSourceError, match="timestamp_seconds"):
        frame_record_from_array(
            image, source_id="s", frame_index=0, timestamp_seconds=-0.5
        )

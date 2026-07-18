"""HELMET layout adapters + sniffer (H2).

The HELMET on-disk format is UNVERIFIED (dataset unavailable at Step 0), so these
tests define the exact layout each adapter *assumes* via synthetic fixtures and
prove the adapter parses that assumed layout correctly and deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helmet_rtdetr.convert import (
    HelmetFlatCsvAdapter,
    HelmetTrackCsvAdapter,
    sniff_helmet_layout,
)
from helmet_rtdetr.convert.helmet import HELMET_ADAPTERS
from helmet_rtdetr.errors import (
    MalformedAnnotationError,
    UnknownHelmetLayoutError,
    UnsupportedLabelError,
)
from helmet_rtdetr.unified import UnifiedClass

_TRACK_HEADER = "frame,track_id,x,y,w,h,label\n"
_FLAT_HEADER = "video,frame,x,y,w,h,label\n"


def make_track_layout(root: Path, rows_by_video: dict[str, list[str]]) -> Path:
    ann = root / "annotation"
    ann.mkdir(parents=True, exist_ok=True)
    for video, rows in rows_by_video.items():
        (ann / f"{video}.csv").write_text(_TRACK_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return root


def make_flat_layout(root: Path, rows: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "annotations.csv").write_text(_FLAT_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return root


# --- track-csv layout --------------------------------------------------------
def test_track_csv_converts_and_maps_labels(tmp_path: Path) -> None:
    root = make_track_layout(
        tmp_path,
        {
            "vidA": ["0,1,10,20,30,40,DNoHelmet", "0,2,50,60,10,10,DHelmet"],
            "vidB": ["3,1,5,5,5,5,motorbike"],
        },
    )
    objs = list(
        HelmetTrackCsvAdapter().convert(root, dataset_id="helmet-myanmar", dataset_version="2020.0")
    )

    assert len(objs) == 3
    labels = {(o.video_id, o.label) for o in objs}
    assert ("vidA", UnifiedClass.NO_HELMET) in labels
    assert ("vidA", UnifiedClass.HELMET) in labels
    assert ("vidB", UnifiedClass.MOTORCYCLE) in labels


def test_track_csv_populates_frame_identity(tmp_path: Path) -> None:
    root = make_track_layout(tmp_path, {"vidA": ["7,1,10,20,30,40,DNoHelmet"]})
    o = next(iter(HelmetTrackCsvAdapter().convert(root, dataset_id="d", dataset_version="1")))

    assert o.video_id == "vidA"
    assert o.frame_index == 7
    assert o.frame_id == "vidA:7"
    assert o.provenance.adapter == "helmet-track-csv"
    assert o.provenance.source_label == "DNoHelmet"


def test_track_csv_detect(tmp_path: Path) -> None:
    root = make_track_layout(tmp_path, {"vidA": ["0,1,1,1,1,1,DHelmet"]})
    assert HelmetTrackCsvAdapter().detect(root) is True
    assert HelmetTrackCsvAdapter().detect(tmp_path / "nope") is False


def test_track_csv_unsupported_label_raises(tmp_path: Path) -> None:
    root = make_track_layout(tmp_path, {"vidA": ["0,1,1,1,1,1,Spaceship"]})
    with pytest.raises(UnsupportedLabelError, match="Spaceship"):
        list(HelmetTrackCsvAdapter().convert(root, dataset_id="d", dataset_version="1"))


def test_track_csv_non_numeric_row_raises(tmp_path: Path) -> None:
    root = make_track_layout(tmp_path, {"vidA": ["zero,1,1,1,1,1,DHelmet"]})
    with pytest.raises(MalformedAnnotationError):
        list(HelmetTrackCsvAdapter().convert(root, dataset_id="d", dataset_version="1"))


def test_track_csv_bad_header_missing_column_raises(tmp_path: Path) -> None:
    ann = tmp_path / "annotation"
    ann.mkdir(parents=True)
    (ann / "vidA.csv").write_text("frame,x,y,w,h,label\n0,1,1,1,1,DHelmet\n", encoding="utf-8")
    with pytest.raises(MalformedAnnotationError, match="missing columns"):
        list(HelmetTrackCsvAdapter().convert(tmp_path, dataset_id="d", dataset_version="1"))


# --- flat-csv layout ---------------------------------------------------------
def test_flat_csv_converts(tmp_path: Path) -> None:
    root = make_flat_layout(
        tmp_path,
        ["vidA,0,10,20,30,40,DNoHelmet", "vidB,2,1,1,1,1,DHelmet"],
    )
    objs = list(HelmetFlatCsvAdapter().convert(root, dataset_id="d", dataset_version="1"))

    assert {o.video_id for o in objs} == {"vidA", "vidB"}
    assert objs[0].provenance.adapter == "helmet-flat-csv"


def test_flat_csv_order_is_deterministic(tmp_path: Path) -> None:
    root = make_flat_layout(
        tmp_path,
        ["vidB,5,1,1,1,1,DHelmet", "vidA,0,2,2,2,2,DNoHelmet", "vidA,1,3,3,3,3,DHelmet"],
    )
    def run() -> list[str]:
        adapter = HelmetFlatCsvAdapter()
        return [o.object_id for o in adapter.convert(root, dataset_id="d", dataset_version="1")]

    assert run() == run()


def test_flat_csv_bad_header_missing_video_raises(tmp_path: Path) -> None:
    (tmp_path / "annotations.csv").write_text(
        "frame,x,y,w,h,label\n0,1,1,1,1,DHelmet\n", encoding="utf-8"
    )
    with pytest.raises(MalformedAnnotationError, match="video"):
        list(HelmetFlatCsvAdapter().convert(tmp_path, dataset_id="d", dataset_version="1"))


# --- the sniffer (multi-layout support without hardcoding) -------------------
def test_sniffer_selects_track_layout(tmp_path: Path) -> None:
    root = make_track_layout(tmp_path, {"vidA": ["0,1,1,1,1,1,DHelmet"]})
    assert sniff_helmet_layout(root).name == "helmet-track-csv"


def test_sniffer_selects_flat_layout(tmp_path: Path) -> None:
    root = make_flat_layout(tmp_path, ["vidA,0,1,1,1,1,DHelmet"])
    assert sniff_helmet_layout(root).name == "helmet-flat-csv"


def test_sniffer_raises_on_unknown_layout(tmp_path: Path) -> None:
    (tmp_path / "mystery.txt").write_text("???", encoding="utf-8")
    with pytest.raises(UnknownHelmetLayoutError, match="helmet-track-csv"):
        sniff_helmet_layout(tmp_path)


def test_two_layouts_are_registered() -> None:
    """Genuinely demonstrates multi-layout support, not a single hardcoded format."""

    assert {a.name for a in HELMET_ADAPTERS} == {"helmet-track-csv", "helmet-flat-csv"}

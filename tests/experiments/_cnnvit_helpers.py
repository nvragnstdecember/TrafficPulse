"""Synthetic HELMET-shaped fixtures for the CNN-vs-ViT experiment tests (P4-U5).

Real files in a temp directory, not mocks -- the same philosophy as
``_rtdetr_helpers.make_split_fixture``: the corpus builder reads genuine CSVs with
the genuine header, so a change to the parsing contract fails a test rather than
sliding past a stub.

The fixtures are deliberately tiny and hand-countable, so every expected number in
the tests can be verified by reading the fixture rather than by running the code.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from helmet_cnn_vit.corpus import ANNOTATION_COLUMNS

#: A helmeted solo rider and an unhelmeted pillion pair -- the two commonest
#: real labels, and enough to exercise both classes and the rider-count covariate.
SOLO_HELMET = "DHelmet"
PAIR_NO_HELMET = "DNoHelmetP1NoHelmet"


def track_rows(
    track_id: str,
    label: str,
    frames: Iterable[int],
    *,
    x: float = 100.0,
    y: float = 200.0,
    w: float = 60.0,
    h: float = 99.0,
) -> list[dict[str, str]]:
    """Rows for one track: a constant label and box across ``frames``."""

    return [
        {
            "track_id": track_id,
            "frame_id": str(frame),
            "x": f"{x:g}",
            "y": f"{y:g}",
            "w": f"{w:g}",
            "h": f"{h:g}",
            "label": label,
        }
        for frame in frames
    ]


def write_annotation(
    annotation_dir: Path, video_id: str, rows: Sequence[Mapping[str, str]]
) -> Path:
    """Write one ``<video_id>.csv`` in the real HELMET layout."""

    annotation_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_dir / f"{video_id}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_split_csv(path: Path, assignment: Mapping[str, str]) -> Path:
    """Write a ``data_split.csv`` in the authors' ``video_id,Set`` layout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "Set"])
        for video_id, split in assignment.items():
            writer.writerow([video_id, split])
    return path


def tiny_corpus_fixture(root: Path) -> tuple[Path, Path]:
    """A three-video, two-site corpus with a known, hand-countable shape.

    Layout (frames 1..20; the default policy keeps frames 1, 6, 11, 16):

    ==================  =====  ==========================  ======  =====
    video               site   tracks                      split   crops
    ==================  =====  ==========================  ======  =====
    ``Alpha_site_1``    Alpha  t1 SOLO_HELMET (1..20)      train   4
                               t2 PAIR_NO_HELMET (1..20)   train   4
    ``Alpha_site_2``    Alpha  t3 SOLO_HELMET (1..20)      val     4
    ``Beta_site_1``     Beta   t4 PAIR_NO_HELMET (1..20)   test    4
    ==================  =====  ==========================  ======  =====

    Returns ``(annotation_dir, split_csv)``.
    """

    annotation_dir = root / "annotation"
    write_annotation(
        annotation_dir,
        "Alpha_site_1",
        [
            *track_rows("t1", SOLO_HELMET, range(1, 21)),
            *track_rows("t2", PAIR_NO_HELMET, range(1, 21), x=400.0),
        ],
    )
    write_annotation(annotation_dir, "Alpha_site_2", track_rows("t3", SOLO_HELMET, range(1, 21)))
    write_annotation(annotation_dir, "Beta_site_1", track_rows("t4", PAIR_NO_HELMET, range(1, 21)))
    split_csv = write_split_csv(
        root / "data_split.csv",
        {
            "Alpha_site_1": "training",
            "Alpha_site_2": "validation",
            "Beta_site_1": "test",
        },
    )
    return annotation_dir, split_csv

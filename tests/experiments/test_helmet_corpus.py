"""Corpus builder, validation, and deterministic export (H2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helmet_rtdetr.corpus import (
    CorpusBuilder,
    DuplicatePolicy,
    UnifiedCorpus,
    export_corpus,
    require_image_references,
    validate_image_references,
)
from helmet_rtdetr.errors import (
    DuplicateAnnotationError,
    FrameNumberingError,
    MissingImageError,
)
from helmet_rtdetr.models import CorpusMember, CorpusVersion
from helmet_rtdetr.unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject

CV = CorpusVersion(
    corpus_id="helmet-core",
    version="1",
    members=(CorpusMember(dataset_id="helmet-myanmar", dataset_version="2020.0"),),
)


def prov(dataset_id: str = "helmet-myanmar", adapter: str = "helmet-track-csv") -> ObjectProvenance:
    return ObjectProvenance(
        dataset_id=dataset_id, dataset_version="1", adapter=adapter, source_label="DNoHelmet"
    )


def obj(
    image_path: str,
    box: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
    label: UnifiedClass = UnifiedClass.NO_HELMET,
    *,
    dataset_id: str = "helmet-myanmar",
    video_id: str | None = None,
    frame_index: int | None = None,
) -> UnifiedObject:
    x, y, w, h = box
    return UnifiedObject(
        image_path=image_path,
        bbox=BBox(x=x, y=y, w=w, h=h),
        label=label,
        provenance=prov(dataset_id),
        video_id=video_id,
        frame_index=frame_index,
        frame_id=f"{video_id}:{frame_index}" if video_id and frame_index is not None else None,
    )


# --- merge + provenance ------------------------------------------------------
def test_builder_merges_objects_from_multiple_adds() -> None:
    corpus = (
        CorpusBuilder(CV)
        .add([obj("a.jpg")])
        .add([obj("b.jpg", label=UnifiedClass.HELMET)])
        .build()
    )
    assert len(corpus) == 2
    assert corpus.corpus_version.corpus_id == "helmet-core"


def test_label_counts() -> None:
    corpus = CorpusBuilder(CV).add(
        [
            obj("a.jpg", label=UnifiedClass.NO_HELMET),
            obj("b.jpg", label=UnifiedClass.NO_HELMET),
            obj("c.jpg", label=UnifiedClass.HELMET),
        ]
    ).build()
    assert corpus.label_counts() == {"helmet": 1, "no_helmet": 2}


# --- deterministic ordering --------------------------------------------------
def test_ordering_is_content_determined_not_insertion_order() -> None:
    objs = [
        obj("c.jpg"),
        obj("a.jpg"),
        obj("b.jpg"),
    ]
    forward = CorpusBuilder(CV).add(objs).build()
    backward = CorpusBuilder(CV).add(list(reversed(objs))).build()
    assert forward.content_hash() == backward.content_hash()
    assert [o.image_path for o in forward.objects] == ["a.jpg", "b.jpg", "c.jpg"]


def test_content_hash_changes_with_content() -> None:
    a = CorpusBuilder(CV).add([obj("a.jpg")]).build()
    b = CorpusBuilder(CV).add([obj("a.jpg"), obj("b.jpg")]).build()
    assert a.content_hash() != b.content_hash()


# --- duplicate detection -----------------------------------------------------
def test_duplicate_annotations_error_by_default() -> None:
    dup = obj("a.jpg", (1.0, 2.0, 3.0, 4.0))
    same = obj("a.jpg", (1.0, 2.0, 3.0, 4.0), dataset_id="roboflow-moto-helmet")
    with pytest.raises(DuplicateAnnotationError):
        CorpusBuilder(CV).add([dup, same]).build()


def test_duplicate_annotations_can_be_dropped() -> None:
    dup = obj("a.jpg", (1.0, 2.0, 3.0, 4.0))
    same = obj("a.jpg", (1.0, 2.0, 3.0, 4.0), dataset_id="roboflow-moto-helmet")
    corpus = CorpusBuilder(CV).add([dup, same]).build(on_duplicate=DuplicatePolicy.DROP)
    assert len(corpus) == 1


def test_near_but_distinct_boxes_are_not_duplicates() -> None:
    corpus = CorpusBuilder(CV).add(
        [obj("a.jpg", (1.0, 2.0, 3.0, 4.0)), obj("a.jpg", (9.0, 2.0, 3.0, 4.0))]
    ).build()
    assert len(corpus) == 2


# --- frame numbering ---------------------------------------------------------
def test_consistent_frame_numbering_is_accepted() -> None:
    corpus = CorpusBuilder(CV).add(
        [
            obj("v/f0.jpg", video_id="vidA", frame_index=0),
            obj("v/f1.jpg", (5.0, 5.0, 5.0, 5.0), video_id="vidA", frame_index=1),
        ]
    ).build()
    assert len(corpus) == 2


def test_mixing_framed_and_unframed_in_one_video_is_rejected() -> None:
    with pytest.raises(FrameNumberingError):
        CorpusBuilder(CV).add(
            [
                obj("v/f0.jpg", video_id="vidA", frame_index=0),
                obj("v/x.jpg", (5.0, 5.0, 5.0, 5.0), video_id="vidA", frame_index=None),
            ]
        ).build()


# --- image reference validation ----------------------------------------------
def test_validate_image_references_reports_missing(tmp_path: Path) -> None:
    (tmp_path / "present.jpg").write_bytes(b"x")
    corpus = CorpusBuilder(CV).add([obj("present.jpg"), obj("absent.jpg")]).build()

    missing = validate_image_references(corpus, tmp_path)
    assert missing == ("absent.jpg",)


def test_validate_image_references_checks_each_path_once(tmp_path: Path) -> None:
    """Two objects sharing an image_path are validated as one reference."""

    (tmp_path / "shared.jpg").write_bytes(b"x")
    corpus = CorpusBuilder(CV).add(
        [obj("shared.jpg", (1.0, 1.0, 1.0, 1.0)), obj("shared.jpg", (9.0, 9.0, 9.0, 9.0))]
    ).build()
    assert validate_image_references(corpus, tmp_path) == ()


def test_require_image_references_raises_on_missing(tmp_path: Path) -> None:
    corpus = CorpusBuilder(CV).add([obj("absent.jpg")]).build()
    with pytest.raises(MissingImageError):
        require_image_references(corpus, tmp_path)


def test_require_image_references_passes_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    corpus = CorpusBuilder(CV).add([obj("a.jpg")]).build()
    require_image_references(corpus, tmp_path)  # must not raise


# --- deterministic export ----------------------------------------------------
def test_export_is_byte_identical_for_identical_input(tmp_path: Path) -> None:
    objs = [obj("b.jpg"), obj("a.jpg", (2.0, 2.0, 2.0, 2.0))]
    p1 = export_corpus(CorpusBuilder(CV).add(objs).build(), tmp_path / "a.jsonl")
    p2 = export_corpus(
        CorpusBuilder(CV).add(list(reversed(objs))).build(), tmp_path / "b.jsonl"
    )
    assert p1.read_bytes() == p2.read_bytes()


def test_export_round_trips_line_count(tmp_path: Path) -> None:
    corpus = CorpusBuilder(CV).add([obj("a.jpg"), obj("b.jpg")]).build()
    path = export_corpus(corpus, tmp_path / "out" / "corpus.jsonl")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2


def test_empty_corpus_exports_cleanly(tmp_path: Path) -> None:
    corpus = UnifiedCorpus(corpus_version=CV, objects=())
    path = export_corpus(corpus, tmp_path / "empty.jsonl")
    assert path.read_text(encoding="utf-8") == ""


def test_to_jsonl_is_deterministic() -> None:
    objs = [obj("c.jpg"), obj("a.jpg"), obj("b.jpg")]
    corpus = CorpusBuilder(CV).add(objs).build()
    assert corpus.to_jsonl() == corpus.to_jsonl()

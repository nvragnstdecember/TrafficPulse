"""COCO converter (H2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helmet_rtdetr.convert import CocoAdapter
from helmet_rtdetr.errors import MalformedAnnotationError, UnsupportedLabelError
from helmet_rtdetr.unified import UnifiedClass

LABEL_MAP = {
    "With Helmet": UnifiedClass.HELMET,
    "Without Helmet": UnifiedClass.NO_HELMET,
    "License Plate": None,  # recognised but intentionally skipped
}


def convert(root: Path) -> list:
    """Run the COCO adapter on ``root`` with the shared label map (test brevity)."""

    return list(CocoAdapter(LABEL_MAP).convert(root, dataset_id="rf", dataset_version="1"))


def write_coco(root: Path, annotations: list[dict], categories: list[dict] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    doc = {
        "images": [{"id": 1, "file_name": "img1.jpg"}, {"id": 2, "file_name": "img2.jpg"}],
        "categories": categories
        or [
            {"id": 10, "name": "With Helmet"},
            {"id": 20, "name": "Without Helmet"},
            {"id": 30, "name": "License Plate"},
        ],
        "annotations": annotations,
    }
    path = root / "_annotations.coco.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_converts_annotations_to_unified_objects(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path,
        [
            {"id": 1, "image_id": 1, "category_id": 20, "bbox": [5, 6, 7, 8]},
            {"id": 2, "image_id": 2, "category_id": 10, "bbox": [1, 2, 3, 4]},
        ],
    )
    objs = convert(root)

    assert len(objs) == 2
    by_image = {o.image_path: o for o in objs}
    assert by_image["img1.jpg"].label is UnifiedClass.NO_HELMET
    assert by_image["img2.jpg"].label is UnifiedClass.HELMET
    assert by_image["img1.jpg"].provenance.adapter == "coco"
    assert by_image["img1.jpg"].provenance.source_label == "Without Helmet"


def test_none_mapped_labels_are_skipped(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path,
        [
            {"id": 1, "image_id": 1, "category_id": 30, "bbox": [1, 2, 3, 4]},  # plate -> skip
            {"id": 2, "image_id": 1, "category_id": 10, "bbox": [5, 6, 7, 8]},
        ],
    )
    objs = convert(root)
    assert [o.label for o in objs] == [UnifiedClass.HELMET]


def test_unsupported_label_raises(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path,
        [{"id": 1, "image_id": 1, "category_id": 99, "bbox": [1, 2, 3, 4]}],
        categories=[{"id": 99, "name": "Unicorn"}],
    )
    with pytest.raises(UnsupportedLabelError, match="Unicorn"):
        convert(root)


def test_conversion_order_is_deterministic(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path,
        [
            {"id": 3, "image_id": 2, "category_id": 10, "bbox": [1, 1, 1, 1]},
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [2, 2, 2, 2]},
            {"id": 2, "image_id": 1, "category_id": 20, "bbox": [3, 3, 3, 3]},
        ],
    )
    a = [o.object_id for o in convert(root)]
    b = [o.object_id for o in convert(root)]
    assert a == b  # sorted by (image_id, id), stable


def test_detect_finds_the_annotation_file(tmp_path: Path) -> None:
    root = write_coco(tmp_path, [])
    assert CocoAdapter(LABEL_MAP).detect(root) is True
    assert CocoAdapter(LABEL_MAP).detect(tmp_path / "empty") is False


def test_missing_annotation_file_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(MalformedAnnotationError):
        convert(tmp_path / "empty")


def test_malformed_json_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "_annotations.coco.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(MalformedAnnotationError):
        convert(tmp_path)


def test_bad_bbox_arity_raises(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path, [{"id": 1, "image_id": 1, "category_id": 10, "bbox": [1, 2, 3]}]
    )
    with pytest.raises(MalformedAnnotationError, match="bbox"):
        convert(root)


def test_annotation_referencing_unknown_category_raises(tmp_path: Path) -> None:
    root = write_coco(
        tmp_path, [{"id": 1, "image_id": 1, "category_id": 777, "bbox": [1, 2, 3, 4]}]
    )
    with pytest.raises(MalformedAnnotationError):
        convert(root)  # 777 is not in categories -> per-annotation KeyError


def test_missing_required_top_level_key_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": []}), encoding="utf-8"  # no categories
    )
    with pytest.raises(MalformedAnnotationError):
        convert(tmp_path)

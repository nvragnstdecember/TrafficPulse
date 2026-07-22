"""The unified annotation schema (H2)."""

from __future__ import annotations

import pytest
from helmet_rtdetr.unified import (
    BBox,
    ObjectProvenance,
    UnifiedClass,
    UnifiedObject,
)
from pydantic import ValidationError

PROV = ObjectProvenance(
    dataset_id="ds", dataset_version="1", adapter="coco", source_label="Without Helmet"
)


def obj(**kw: object) -> UnifiedObject:
    base: dict[str, object] = {
        "image_path": "img/a.jpg",
        "bbox": BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        "label": UnifiedClass.NO_HELMET,
        "provenance": PROV,
    }
    base.update(kw)
    return UnifiedObject(**base)  # type: ignore[arg-type]


# --- BBox --------------------------------------------------------------------
def test_bbox_derived_properties() -> None:
    box = BBox(x=10.0, y=20.0, w=30.0, h=40.0)
    assert (box.x2, box.y2, box.area) == (40.0, 60.0, 1200.0)


def test_bbox_rejects_non_positive_size() -> None:
    with pytest.raises(ValidationError):
        BBox(x=0.0, y=0.0, w=0.0, h=1.0)
    with pytest.raises(ValidationError):
        BBox(x=0.0, y=0.0, w=1.0, h=-1.0)


def test_bbox_rejects_negative_origin() -> None:
    with pytest.raises(ValidationError):
        BBox(x=-1.0, y=0.0, w=1.0, h=1.0)


def test_bbox_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        BBox(x=float("nan"), y=0.0, w=1.0, h=1.0)
    with pytest.raises(ValidationError):
        BBox(x=0.0, y=0.0, w=float("inf"), h=1.0)


def test_bbox_quantised_key_is_float_noise_stable() -> None:
    a = BBox(x=1.0, y=2.0, w=3.0, h=4.0)
    b = BBox(x=1.0004, y=2.0, w=3.0, h=4.0)  # within 3-dp rounding
    assert a.quantised_key() == b.quantised_key()


# --- object_id determinism ---------------------------------------------------
def test_object_id_depends_only_on_content_not_provenance() -> None:
    a = obj()
    other_prov = ObjectProvenance(
        dataset_id="different",
        dataset_version="9",
        adapter="helmet-track-csv",
        source_label="DNoHelmet",
    )
    b = obj(provenance=other_prov)
    assert a.object_id == b.object_id  # same image+box+label -> same id


def test_object_id_changes_with_label() -> None:
    assert obj().object_id != obj(label=UnifiedClass.HELMET).object_id


def test_object_id_changes_with_box() -> None:
    assert obj().object_id != obj(bbox=BBox(x=9.0, y=2.0, w=3.0, h=4.0)).object_id


def test_object_id_is_stable_across_calls() -> None:
    o = obj()
    assert o.object_id == o.object_id


# --- frame consistency -------------------------------------------------------
def test_still_image_object_has_no_frame_identity() -> None:
    o = obj()
    assert o.video_id is None and o.frame_index is None and o.frame_id is None


def test_framed_object_is_accepted() -> None:
    o = obj(video_id="vid1", frame_index=7, frame_id="vid1:7")
    assert o.frame_index == 7


def test_negative_frame_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        obj(video_id="vid1", frame_index=-1)


def test_frame_id_without_video_or_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        obj(frame_id="orphan")
    with pytest.raises(ValidationError):
        obj(video_id="vid1", frame_id="vid1:?")  # index missing


# --- schema hygiene ----------------------------------------------------------
def test_objects_are_frozen_and_strict() -> None:
    o = obj()
    with pytest.raises(ValidationError):
        o.label = UnifiedClass.HELMET  # type: ignore[misc]
    with pytest.raises(ValidationError):
        obj(unknown="x")


def test_class_space_is_exactly_three_labels() -> None:
    assert {c.value for c in UnifiedClass} == {"helmet", "no_helmet", "motorcycle"}
    assert "turban" not in {c.value for c in UnifiedClass}  # deliberate omission

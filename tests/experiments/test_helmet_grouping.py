"""Grouping strategies (H3)."""

from __future__ import annotations

from helmet_rtdetr.grouping import ImageGrouping, VideoAwareGrouping
from helmet_rtdetr.unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject


def obj(
    *,
    dataset_id: str = "helmet-myanmar",
    image_path: str = "img.jpg",
    video_id: str | None = None,
    frame_index: int | None = None,
) -> UnifiedObject:
    return UnifiedObject(
        image_path=image_path,
        bbox=BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        label=UnifiedClass.NO_HELMET,
        provenance=ObjectProvenance(
            dataset_id=dataset_id, dataset_version="1", adapter="a", source_label="DNoHelmet"
        ),
        video_id=video_id,
        frame_index=frame_index,
        frame_id=f"{video_id}:{frame_index}" if video_id and frame_index is not None else None,
    )


# --- video-aware -------------------------------------------------------------
def test_video_objects_group_by_video() -> None:
    g = VideoAwareGrouping()
    a = g.group_key(obj(image_path="v/f0.jpg", video_id="vidA", frame_index=0))
    b = g.group_key(obj(image_path="v/f9.jpg", video_id="vidA", frame_index=9))
    assert a == b  # different frames of one video share a key


def test_different_videos_get_different_keys() -> None:
    g = VideoAwareGrouping()
    assert g.group_key(obj(video_id="vidA", frame_index=0)) != g.group_key(
        obj(video_id="vidB", frame_index=0)
    )


def test_image_objects_group_by_image_when_no_video() -> None:
    g = VideoAwareGrouping()
    assert g.group_key(obj(image_path="p.jpg")) != g.group_key(obj(image_path="q.jpg"))
    assert g.group_key(obj(image_path="p.jpg")) == g.group_key(obj(image_path="p.jpg"))


def test_dataset_id_is_part_of_the_key() -> None:
    """Two datasets that reuse a video id must not collide."""

    g = VideoAwareGrouping()
    assert g.group_key(obj(dataset_id="a", video_id="vid1", frame_index=0)) != g.group_key(
        obj(dataset_id="b", video_id="vid1", frame_index=0)
    )


# --- image grouping ----------------------------------------------------------
def test_image_grouping_ignores_video() -> None:
    g = ImageGrouping()
    a = g.group_key(obj(image_path="same.jpg", video_id="vidA", frame_index=0))
    b = g.group_key(obj(image_path="same.jpg", video_id="vidB", frame_index=0))
    assert a == b  # same image path -> same group regardless of video


# --- group() bucketing -------------------------------------------------------
def test_group_buckets_objects_by_key() -> None:
    g = VideoAwareGrouping()
    objs = [
        obj(image_path="v/f0.jpg", video_id="vidA", frame_index=0),
        obj(image_path="v/f1.jpg", video_id="vidA", frame_index=1),
        obj(image_path="w/f0.jpg", video_id="vidB", frame_index=0),
    ]
    groups = g.group(objs)
    assert len(groups) == 2
    assert sorted(len(v) for v in groups.values()) == [1, 2]


def test_strategy_names_are_stable() -> None:
    assert VideoAwareGrouping().name == "video-aware"
    assert ImageGrouping().name == "image"

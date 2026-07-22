"""Grouping strategies for leakage-safe splitting (H3).

A grouping strategy assigns every unified object a **group key**; the splitter
then keeps all objects sharing a key in the same split. This is the mechanism
that prevents leakage -- if the key is the video, no video's frames can straddle
splits.

The strategy is an abstraction so future grouping units (e.g. by camera site, or
by tracked-instance) drop in without changing the splitter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .unified import UnifiedObject

_SEP = "\x1f"


class GroupingStrategy(ABC):
    """Assigns each object a group key; all objects with one key share a split."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy id, recorded in the split manifest."""

    @abstractmethod
    def group_key(self, obj: UnifiedObject) -> str:
        """The leakage-safety key for one object."""

    def group(self, objects: Iterable[UnifiedObject]) -> dict[str, list[UnifiedObject]]:
        """Bucket objects by :meth:`group_key`, preserving first-seen order."""

        groups: dict[str, list[UnifiedObject]] = {}
        for obj in objects:
            groups.setdefault(self.group_key(obj), []).append(obj)
        return groups


class VideoAwareGrouping(GroupingStrategy):
    """The default, leakage-safe strategy: by video when known, else by image.

    Handles a mixed corpus correctly: HELMET frames (which carry ``video_id``)
    group by video, while Roboflow still images (no ``video_id``) group by image.
    The dataset id is part of the key, so two datasets that happen to reuse a video
    id never collide.
    """

    @property
    def name(self) -> str:
        return "video-aware"

    def group_key(self, obj: UnifiedObject) -> str:
        tail = f"v:{obj.video_id}" if obj.video_id is not None else f"i:{obj.image_path}"
        return f"{obj.provenance.dataset_id}{_SEP}{tail}"


class ImageGrouping(GroupingStrategy):
    """Groups strictly by image.

    Leakage-safe **only** for datasets without video structure; applying it to
    video data would let frames of one clip land in different splits. Provided for
    image-only corpora and as an explicit, documented alternative.
    """

    @property
    def name(self) -> str:
        return "image"

    def group_key(self, obj: UnifiedObject) -> str:
        return f"{obj.provenance.dataset_id}{_SEP}i:{obj.image_path}"

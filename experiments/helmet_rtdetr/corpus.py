"""Unified corpus builder + deterministic export (H2).

Merges converted objects from any number of datasets into one validated,
deterministically ordered :class:`UnifiedCorpus`, carrying the H1
:class:`CorpusVersion` for provenance. All logic here is dataset-agnostic -- it
operates only on :class:`UnifiedObject`\\ s.

Determinism
-----------
Objects are ordered by ``(image_path, x, y, w, h, label, object_id)`` -- a pure
function of content -- so identical inputs always export byte-identical output,
regardless of the order datasets were added or objects were produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Self

from .errors import (
    DuplicateAnnotationError,
    FrameNumberingError,
    MissingImageError,
)
from .models import CorpusVersion, _Model
from .unified import UnifiedObject


class DuplicatePolicy(StrEnum):
    """What :meth:`CorpusBuilder.build` does when two objects share an ``object_id``."""

    ERROR = "error"  # raise DuplicateAnnotationError (default; loud)
    DROP = "drop"  # keep the first occurrence (deterministic after sort)


def _sort_key(obj: UnifiedObject) -> tuple[str, float, float, float, float, str, str]:
    box = obj.bbox
    return (obj.image_path, box.x, box.y, box.w, box.h, obj.label.value, obj.object_id)


class UnifiedCorpus(_Model):
    """A validated, deterministically ordered set of unified objects + provenance."""

    corpus_version: CorpusVersion
    objects: tuple[UnifiedObject, ...]

    def __len__(self) -> int:
        return len(self.objects)

    def to_jsonl(self) -> str:
        """One object per line, in deterministic order. The canonical export form."""

        return "\n".join(obj.model_dump_json() for obj in self.objects)

    def content_hash(self) -> str:
        """SHA-256 over the corpus-version identity + the ordered objects."""

        preimage = self.corpus_version.content_hash() + "\n" + self.to_jsonl()
        return hashlib.sha256(preimage.encode("utf-8")).hexdigest()

    def label_counts(self) -> dict[str, int]:
        """Count of objects per unified label (for balance reporting)."""

        counts: dict[str, int] = {}
        for obj in self.objects:
            counts[obj.label.value] = counts.get(obj.label.value, 0) + 1
        return dict(sorted(counts.items()))


class CorpusBuilder:
    """Accumulates unified objects and builds a validated :class:`UnifiedCorpus`."""

    def __init__(self, corpus_version: CorpusVersion) -> None:
        self._corpus_version = corpus_version
        self._objects: list[UnifiedObject] = []

    def add(self, objects: Iterable[UnifiedObject]) -> Self:
        """Add converted objects (from any adapter). Chainable."""

        self._objects.extend(objects)
        return self

    def build(self, *, on_duplicate: DuplicatePolicy = DuplicatePolicy.ERROR) -> UnifiedCorpus:
        """Validate, de-duplicate, order deterministically, and return the corpus."""

        ordered = sorted(self._objects, key=_sort_key)
        deduped = self._resolve_duplicates(ordered, on_duplicate)
        self._check_frame_numbering(deduped)
        return UnifiedCorpus(
            corpus_version=self._corpus_version, objects=tuple(deduped)
        )

    @staticmethod
    def _resolve_duplicates(
        ordered: list[UnifiedObject], policy: DuplicatePolicy
    ) -> list[UnifiedObject]:
        seen: dict[str, UnifiedObject] = {}
        result: list[UnifiedObject] = []
        for obj in ordered:
            oid = obj.object_id
            if oid in seen:
                if policy is DuplicatePolicy.ERROR:
                    raise DuplicateAnnotationError(
                        f"duplicate annotation {oid} on image {obj.image_path!r} "
                        f"(label {obj.label.value}); sources "
                        f"{seen[oid].provenance.dataset_id!r} and "
                        f"{obj.provenance.dataset_id!r}"
                    )
                continue  # DROP: keep the first (already in result)
            seen[oid] = obj
            result.append(obj)
        return result

    @staticmethod
    def _check_frame_numbering(objects: list[UnifiedObject]) -> None:
        """Within one video, objects must be uniformly framed or uniformly unframed."""

        framed: dict[str, bool] = {}
        for obj in objects:
            if obj.video_id is None:
                continue
            has_frame = obj.frame_index is not None
            if obj.video_id in framed and framed[obj.video_id] != has_frame:
                raise FrameNumberingError(
                    f"video {obj.video_id!r} mixes framed and unframed objects; "
                    "frame numbering must be consistent within a video"
                )
            framed[obj.video_id] = has_frame


def validate_image_references(corpus: UnifiedCorpus, image_root: Path) -> tuple[str, ...]:
    """Return the distinct ``image_path``\\ s that do not exist under ``image_root``.

    Reports missing images as data (an empty tuple means all present). Use
    :func:`require_image_references` for the raising variant.
    """

    missing: list[str] = []
    seen: set[str] = set()
    for obj in corpus.objects:
        if obj.image_path in seen:
            continue
        seen.add(obj.image_path)
        if not (image_root / obj.image_path).is_file():
            missing.append(obj.image_path)
    return tuple(sorted(missing))


def require_image_references(corpus: UnifiedCorpus, image_root: Path) -> None:
    """Raise :class:`MissingImageError` if any referenced image is absent."""

    missing = validate_image_references(corpus, image_root)
    if missing:
        raise MissingImageError(
            f"{len(missing)} referenced image(s) missing under {image_root}: "
            f"{list(missing[:5])}{'…' if len(missing) > 5 else ''}"
        )


def export_corpus(corpus: UnifiedCorpus, path: Path) -> Path:
    """Write the corpus as deterministic JSONL to ``path``; return ``path``.

    Identical corpora always produce byte-identical files. Writes only where
    directed (tests use a temp dir); downloads nothing.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    text = corpus.to_jsonl()
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return path

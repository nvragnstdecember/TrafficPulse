"""HELMET-family converters (H2) -- UNVERIFIED layout, handled adaptively.

Step-0 status (mandatory, honest)
---------------------------------
The HELMET dataset was **not available** for schema verification when this was
written: ``data/raw/helmet-myanmar`` is absent (nothing has been downloaded), and
the authors' repository (``LinHanhe/Helmet_use_detection``) contains only a README
and a demo GIF -- no loader code and no format specification. The exact on-disk
annotation layout therefore **cannot be confirmed against real files**.

Per the H2 mandate, the HELMET format is handled by a **pluggable set of layout
adapters** behind a sniffer, rather than by hardcoding one assumed format. Two
plausible layouts are implemented -- a per-video CSV and a single flat CSV -- each
using the documented HELMET/AI-City-lineage class vocabulary. Adding a third
layout is a new adapter, not a change to shared code.

**Before trusting any output here**, download the dataset and confirm which
adapter's :meth:`detect` matches the real files (or add one that does). The
synthetic fixtures in the tests define exactly the layout each adapter assumes.

Class vocabulary
----------------
The HELMET/AI-City lineage labels each rider slot with its helmet state:
``D`` = driver, ``P1``/``P2`` = pillion passengers. All *Helmet -> helmet, all
*NoHelmet -> no_helmet, ``motorbike`` -> motorcycle. There is no turban label in
the source (consistent with the unified schema's deliberate omission).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ..errors import MalformedAnnotationError, UnknownHelmetLayoutError
from ..unified import BBox, ObjectProvenance, UnifiedClass, UnifiedObject
from .base import AnnotationAdapter, LabelMap, map_label

# Documented HELMET / AI-City-lineage class map (UNVERIFIED against real files).
HELMET_LABEL_MAP: LabelMap = {
    "DHelmet": UnifiedClass.HELMET,
    "DNoHelmet": UnifiedClass.NO_HELMET,
    "P1Helmet": UnifiedClass.HELMET,
    "P1NoHelmet": UnifiedClass.NO_HELMET,
    "P2Helmet": UnifiedClass.HELMET,
    "P2NoHelmet": UnifiedClass.NO_HELMET,
    "motorbike": UnifiedClass.MOTORCYCLE,
    "motorcycle": UnifiedClass.MOTORCYCLE,
}

_BBOX_COLUMNS = ("x", "y", "w", "h")


def _row_object(
    row: dict[str, str],
    *,
    columns: tuple[str, ...],
    dataset_id: str,
    dataset_version: str,
    adapter: str,
    video_id: str,
    source: Path,
) -> UnifiedObject | None:
    """Build one unified object from a parsed CSV row (or ``None`` if skipped)."""

    missing = [c for c in columns if c not in row]
    if missing:
        raise MalformedAnnotationError(f"{source}: row missing columns {missing}: {row!r}")
    try:
        frame_index = int(row["frame"])
        x, y, w, h = (float(row[c]) for c in _BBOX_COLUMNS)
    except (ValueError, TypeError) as exc:
        raise MalformedAnnotationError(f"{source}: non-numeric field in row {row!r}") from exc

    mapped = map_label(HELMET_LABEL_MAP, row["label"], adapter=adapter)
    if mapped is None:
        return None
    return UnifiedObject(
        image_path=f"{video_id}/frame_{frame_index:06d}.jpg",
        bbox=BBox(x=x, y=y, w=w, h=h),
        label=mapped,
        provenance=ObjectProvenance(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            adapter=adapter,
            source_label=row["label"],
        ),
        video_id=video_id,
        frame_index=frame_index,
        frame_id=f"{video_id}:{frame_index}",
    )


class HelmetLayoutAdapter(AnnotationAdapter):
    """Base for HELMET layout adapters (shares the HELMET class vocabulary)."""


class HelmetTrackCsvAdapter(HelmetLayoutAdapter):
    """Layout A (ASSUMED): one CSV per video under ``annotation/``.

    Each ``annotation/<video_id>.csv`` has a header row and the columns
    ``frame,track_id,x,y,w,h,label``. The video id is the CSV file stem.
    """

    _COLUMNS = ("frame", "track_id", "x", "y", "w", "h", "label")

    @property
    def name(self) -> str:
        return "helmet-track-csv"

    def _csv_files(self, root: Path) -> list[Path]:
        annotation_dir = root / "annotation"
        return sorted(annotation_dir.glob("*.csv")) if annotation_dir.is_dir() else []

    def detect(self, root: Path) -> bool:
        return bool(self._csv_files(root))

    def convert(
        self, root: Path, *, dataset_id: str, dataset_version: str
    ) -> Iterator[UnifiedObject]:
        for path in self._csv_files(root):
            video_id = path.stem
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    obj = _row_object(
                        row,
                        columns=self._COLUMNS,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        adapter=self.name,
                        video_id=video_id,
                        source=path,
                    )
                    if obj is not None:
                        yield obj


class HelmetFlatCsvAdapter(HelmetLayoutAdapter):
    """Layout B (ASSUMED): a single flat ``annotations.csv`` at the dataset root.

    Header row with columns ``video,frame,x,y,w,h,label``; the video id is a
    column value rather than a filename.
    """

    _COLUMNS = ("video", "frame", "x", "y", "w", "h", "label")

    @property
    def name(self) -> str:
        return "helmet-flat-csv"

    def _file(self, root: Path) -> Path:
        return root / "annotations.csv"

    def detect(self, root: Path) -> bool:
        return self._file(root).is_file()

    def convert(
        self, root: Path, *, dataset_id: str, dataset_version: str
    ) -> Iterator[UnifiedObject]:
        path = self._file(root)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        # Deterministic order independent of file line order.
        for row in sorted(rows, key=lambda r: (r.get("video", ""), r.get("frame", ""))):
            if "video" not in row:
                raise MalformedAnnotationError(f"{path}: row missing 'video' column: {row!r}")
            obj = _row_object(
                row,
                columns=self._COLUMNS,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                adapter=self.name,
                video_id=row["video"],
                source=path,
            )
            if obj is not None:
                yield obj


# The registered HELMET layouts, tried in order by the sniffer.
HELMET_ADAPTERS: tuple[HelmetLayoutAdapter, ...] = (
    HelmetTrackCsvAdapter(),
    HelmetFlatCsvAdapter(),
)


def sniff_helmet_layout(
    root: Path, *, adapters: tuple[HelmetLayoutAdapter, ...] = HELMET_ADAPTERS
) -> HelmetLayoutAdapter:
    """Return the first registered adapter whose layout matches ``root``.

    Raises :class:`UnknownHelmetLayoutError` if none match -- the pipeline refuses
    to parse an unrecognised layout rather than mis-parse it.
    """

    for adapter in adapters:
        if adapter.detect(root):
            return adapter
    raise UnknownHelmetLayoutError(
        f"no registered HELMET layout adapter recognises {root} "
        f"(tried: {[a.name for a in adapters]}); confirm the dataset layout and add an adapter"
    )

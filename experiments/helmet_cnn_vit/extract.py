"""Crop harvesting from the distributed HELMET archives (P4-U5).

Reads frames **directly out of the seven ``part_N.zip`` archives** rather than
extracting them first. The archives hold 91,000 1080p frames totalling ~28.9 GB;
extracting them would double that for no benefit, since each frame is needed once
and only a ~200 px region of it survives. Streaming from the zip keeps peak disk at
the download size plus the ~0.6 GB crop set.

Archive layout (verified against ``part_1.zip`` on 2026-08-29)
--------------------------------------------------------------
``part_N/<video_id>/<frame_id>.jpg`` -- frame ids are **zero-padded to two digits**
(``01.jpg`` .. ``99.jpg``, then ``100.jpg``), and every clip directory holds exactly
100 frames. The padding is easy to miss: a sample of member names shows only
two-digit stems, so an unpadded ``f"{n}.jpg"`` silently fails for frames 1-9 only.
:func:`extract_crops` therefore *counts* every member it cannot open instead of
skipping it, which is how that mistake was caught rather than shipped.

Preprocessing
-------------
Each crop is cut tight to the annotated motorcycle box, padded to a square with a
constant colour (so aspect ratio is preserved and the model never learns from a
stretch artefact), then resized to ``image_size``. This is applied **identically to
both model families** -- §12 requires comparable input-resolution policy, and doing
the geometry once here rather than in each family's transform pipeline is what
guarantees it.

Crops are written as JPEG. The re-encode is lossy, but it is the same loss for both
families and is applied before any train/eval split is consulted, so it cannot
favour either model. Quality is recorded in the report for reproducibility.

Pillow is imported lazily: nothing in this module's import graph pulls an image
library into CI.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helmet_rtdetr.models import _Model
from pydantic import Field

from .corpus import CropRecord
from .errors import CorpusBuildError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

#: The seven archives the image folder ships as.
PART_NAMES: tuple[str, ...] = tuple(f"part_{i}" for i in range(1, 8))


class ExtractionConfig(_Model):
    """Frozen preprocessing geometry, recorded in the extraction report."""

    image_size: int = Field(default=224, ge=32, le=1024)
    #: Constant square-pad colour. Black; the alternative (edge replication) would
    #: invent texture that the annotation does not support.
    pad_rgb: tuple[int, int, int] = (0, 0, 0)
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    #: Fractional box expansion before cropping. Zero keeps the crop exactly as
    #: annotated, so no unvalidated context heuristic enters the comparison.
    context_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionReport(_Model):
    """What was written, and what could not be."""

    config: ExtractionConfig
    crops_written: int
    frames_read: int
    videos: int
    missing_frames: tuple[str, ...]
    per_split: dict[str, int]
    per_class: dict[str, int]


def build_frame_index(
    image_dir: Path, *, archives: Sequence[Path] | None = None
) -> dict[str, tuple[str, str]]:
    """Map ``video_id -> (archive filename, member prefix)`` across all parts.

    Scans each archive's central directory once (cheap -- no decompression) and
    fails loudly if one clip appears in two archives, which would make the source
    of a crop ambiguous. ``archives`` restricts the scan to an explicit list, which
    is how the tests index a single part.

    A truncated archive -- an interrupted or still-running download -- is reported
    as such rather than surfacing a bare ``BadZipFile``: it is the likeliest
    operational failure here, and silently skipping it would produce a frame index
    that is quietly missing whole sites.
    """

    index: dict[str, tuple[str, str]] = {}
    found = sorted(archives) if archives is not None else sorted(image_dir.glob("part_*.zip"))
    if not found:
        raise CorpusBuildError(f"no part_*.zip archives under {image_dir}")

    for archive in found:
        try:
            handle = zipfile.ZipFile(archive)
        except zipfile.BadZipFile as exc:
            raise CorpusBuildError(
                f"{archive.name} is not a readable zip ({exc}); if a download is still "
                f"running or was interrupted, complete it and re-verify its byte size"
            ) from exc
        with handle:
            for name in handle.namelist():
                if not name.endswith("/"):
                    continue
                segments = name.strip("/").split("/")
                if len(segments) != 2:
                    continue
                prefix, video_id = segments
                previous = index.get(video_id)
                if previous is not None and previous[0] != archive.name:
                    raise CorpusBuildError(
                        f"clip {video_id!r} appears in both {previous[0]} and {archive.name}"
                    )
                index[video_id] = (archive.name, f"{prefix}/{video_id}")
    return index


def member_name(prefix: str, frame_index: int) -> str:
    """The archive member holding one frame.

    Frame ids are zero-padded to two digits, so frame 7 is ``07.jpg``; frame 100 is
    the one three-digit name and ``:02d`` already renders it correctly.
    """

    return f"{prefix}/{frame_index:02d}.jpg"


def square_pad_resize(image: Image.Image, config: ExtractionConfig) -> Image.Image:
    """Pad to a centred square, then resize to ``config.image_size``."""

    from PIL import Image as PilImage

    width, height = image.size
    side = max(width, height)
    if width != height:
        canvas = PilImage.new("RGB", (side, side), config.pad_rgb)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        image = canvas
    size = config.image_size
    if image.size != (size, size):
        image = image.resize((size, size), PilImage.BILINEAR)
    return image


def crop_box(
    record: CropRecord, frame_size: tuple[int, int], config: ExtractionConfig
) -> tuple[int, int, int, int]:
    """The integer pixel box to cut, expanded by ``context_fraction`` and clamped.

    Clamping to the frame matters: HELMET boxes can run off the edge for a
    motorcycle that is partly out of shot, and PIL would silently pad such a crop
    with black, which would look like a real (dark) observation to the model.
    """

    width, height = frame_size
    box = record.bbox
    pad_x = box.w * config.context_fraction / 2.0
    pad_y = box.h * config.context_fraction / 2.0
    left = max(0, int(round(box.x - pad_x)))
    top = max(0, int(round(box.y - pad_y)))
    right = min(width, int(round(box.x2 + pad_x)))
    bottom = min(height, int(round(box.y2 + pad_y)))
    if right <= left or bottom <= top:
        raise CorpusBuildError(
            f"crop {record.crop_id} for {record.video_id} frame {record.frame_index} "
            f"is empty after clamping to the {width}x{height} frame"
        )
    return left, top, right, bottom


def _grouped_by_video(
    records: Iterable[tuple[str, CropRecord]],
) -> dict[str, list[tuple[str, CropRecord]]]:
    grouped: dict[str, list[tuple[str, CropRecord]]] = {}
    for split, record in records:
        grouped.setdefault(record.video_id, []).append((split, record))
    return grouped


def extract_crops(
    splits: Mapping[str, Sequence[CropRecord]],
    *,
    image_dir: Path,
    output_dir: Path,
    config: ExtractionConfig | None = None,
    frame_index: Mapping[str, tuple[str, str]] | None = None,
) -> ExtractionReport:
    """Write every crop to ``output_dir/<split>/<class>/<crop_id>.jpg`` plus an index.

    Work is grouped by clip and then by frame, so each 1080p frame is decoded once
    no matter how many motorcycles it contains.
    """

    from PIL import Image as PilImage

    config = config or ExtractionConfig()
    index = frame_index if frame_index is not None else build_frame_index(image_dir)

    flat = [(split, record) for split, records in splits.items() for record in records]
    by_video = _grouped_by_video(flat)

    missing: list[str] = []
    frames_read = 0
    written = 0
    per_split: dict[str, int] = {}
    per_class: dict[str, int] = {}
    index_rows: list[dict[str, Any]] = []

    for video_id in sorted(by_video):
        located = index.get(video_id)
        if located is None:
            missing.append(f"{video_id}/*")
            continue
        archive_name, prefix = located
        with zipfile.ZipFile(image_dir / archive_name) as archive:
            by_frame: dict[int, list[tuple[str, CropRecord]]] = {}
            for split, record in by_video[video_id]:
                by_frame.setdefault(record.frame_index, []).append((split, record))

            for frame_number in sorted(by_frame):
                name = member_name(prefix, frame_number)
                try:
                    payload = archive.read(name)
                except KeyError:
                    missing.append(name)
                    continue

                # Every image is closed explicitly. A decoded 1080p RGB frame is
                # ~6 MB, and there are ~11.5k of them: leaving the source images to
                # refcounting exhausted a 16 GB machine partway through the corpus.
                # Reading the member into bytes first also keeps PIL off the
                # non-seekable zip stream, which it would otherwise buffer itself.
                with PilImage.open(BytesIO(payload)) as opened:
                    frame = opened.convert("RGB")
                del payload
                frames_read += 1

                try:
                    for split, record in by_frame[frame_number]:
                        left, top, right, bottom = crop_box(record, frame.size, config)
                        region = frame.crop((left, top, right, bottom))
                        try:
                            crop = square_pad_resize(region, config)
                            label = record.driver_state.value
                            destination = output_dir / split / label
                            destination.mkdir(parents=True, exist_ok=True)
                            crop.save(
                                destination / f"{record.crop_id}.jpg",
                                format="JPEG",
                                quality=config.jpeg_quality,
                            )
                            if crop is not region:
                                crop.close()
                        finally:
                            region.close()

                        written += 1
                        per_split[split] = per_split.get(split, 0) + 1
                        per_class[label] = per_class.get(label, 0) + 1
                        index_rows.append(
                            {
                                "crop_id": record.crop_id,
                                "split": split,
                                "label": label,
                                "video_id": record.video_id,
                                "site_id": record.site_id,
                                "track_id": record.track_id,
                                "frame_index": record.frame_index,
                                "rider_count": record.rider_count,
                                "any_no_helmet": record.any_no_helmet,
                                "source_label": record.source_label,
                                "box_w": record.bbox.w,
                                "box_h": record.bbox.h,
                                "path": f"{split}/{label}/{record.crop_id}.jpg",
                            }
                        )
                finally:
                    frame.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows.sort(key=lambda row: str(row["crop_id"]))
    (output_dir / "crops.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8"
    )

    report = ExtractionReport(
        config=config,
        crops_written=written,
        frames_read=frames_read,
        videos=len(by_video),
        missing_frames=tuple(sorted(missing)),
        per_split=dict(sorted(per_split.items())),
        per_class=dict(sorted(per_class.items())),
    )
    (output_dir / "extraction_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_crop_index(output_dir: Path) -> tuple[dict[str, Any], ...]:
    """Read back the crop index written by :func:`extract_crops`."""

    path = output_dir / "crops.jsonl"
    if not path.is_file():
        raise CorpusBuildError(f"no crop index at {path}; run extraction first")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)

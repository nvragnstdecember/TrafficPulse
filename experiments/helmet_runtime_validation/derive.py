"""Derive runtime-equivalent head crops through the production TrafficPulse path.

One pass over the frozen P4-U5 val+test splits, single-rider crops only. Every stage is a
production component imported from ``src/trafficpulse`` -- there is no second detection
pipeline here, and nothing is reimplemented.

    frame -> RTDetrDetector -> DetectionAdapter -> IouTracker -> associate_riders
          -> GT motorcycle match (IoU >= 0.50) -> extract_head_region -> min-height gate

Every eligible annotated rider ends in exactly one bucket: a written crop, or a named
failure reason. Counts are the denominator the report is obliged to publish (PROTOCOL §9),
so nothing is dropped silently.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.datasets import load_rows  # noqa: E402
from helmet_cnn_vit.extract import build_frame_index  # noqa: E402

from trafficpulse.association.riders import RiderAssociationConfig, associate_riders  # noqa: E402
from trafficpulse.contracts import BoundingBox  # noqa: E402
from trafficpulse.contracts.enums import ObjectClass  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.frame import Frame  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    extract_head_region,
)
from trafficpulse.tracking import IouTracker  # noqa: E402

#: serve.py's production mapping, verbatim (``motorbike`` is RT-DETR's native label).
LABEL_MAP: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}

#: Pre-committed in PROTOCOL §3, before any recovery rate was observed.
GT_MATCH_IOU = 0.50
DETECTOR_CHECKPOINT = "PekingU/rtdetr_r50vd"
SCORE_THRESHOLD = 0.5
BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Eligible:
    """One single-rider annotated crop from the frozen corpus, with its GT box."""

    crop_id: str
    split: str
    video_id: str
    frame_index: int
    track_id: str
    label: str
    site_id: str
    gt_box: BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_annotation_boxes(
    annotation_dir: Path, video_id: str
) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    """``(track_id, frame_id) -> (x, y, w, h)`` from one HELMET annotation CSV."""

    path = annotation_dir / f"{video_id}.csv"
    boxes: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["track_id"], int(row["frame_id"]))
            boxes[key] = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
    return boxes


def eligible_population(
    crop_dir: Path, annotation_dir: Path, splits: tuple[str, ...]
) -> list[Eligible]:
    """Single-rider crops from the frozen splits, joined to their annotated boxes."""

    out: list[Eligible] = []
    for split in splits:
        rows = [r for r in load_rows(crop_dir, split) if r.rider_count == 1]
        by_video: dict[str, list] = defaultdict(list)
        for row in rows:
            by_video[row.video_id].append(row)
        for video_id, video_rows in by_video.items():
            boxes = load_annotation_boxes(annotation_dir, video_id)
            for row in video_rows:
                key = (row.track_id, row.frame_index)
                if key not in boxes:  # pragma: no cover - corpus/annotation disagreement
                    continue
                x, y, w, h = boxes[key]
                out.append(
                    Eligible(
                        crop_id=row.crop_id,
                        split=split,
                        video_id=video_id,
                        frame_index=row.frame_index,
                        track_id=row.track_id,
                        label=row.label,
                        site_id=row.site_id,
                        gt_box=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
                    )
                )
    return out


def _read_frame(zf: zipfile.ZipFile, member: str) -> NDArray[np.uint8]:
    return np.asarray(Image.open(io.BytesIO(zf.read(member))).convert("RGB"), dtype=np.uint8)


def derive(
    *,
    raw_dir: Path,
    crop_dir: Path,
    out_dir: Path,
    splits: tuple[str, ...] = ("val", "test"),
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, object]:
    """Run the pipeline over the eligible population; write crops + a manifest."""

    from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector

    population = eligible_population(crop_dir, raw_dir / "annotation", splits)
    if limit is not None:
        population = population[:limit]
    by_frame: dict[tuple[str, int], list[Eligible]] = defaultdict(list)
    for item in population:
        by_frame[(item.video_id, item.frame_index)].append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_out = out_dir / "crops"
    crops_out.mkdir(exist_ok=True)

    detector = RTDetrDetector(
        RTDetrConfig(
            checkpoint=DETECTOR_CHECKPOINT,
            device=device,
            local_files_only=True,
            threshold=SCORE_THRESHOLD,
        )
    )
    adapter = DetectionAdapter(
        DetectorConfig(label_map=LABEL_MAP, score_threshold=SCORE_THRESHOLD)
    )
    association_config = RiderAssociationConfig()

    index = build_frame_index(raw_dir / "image")
    reasons: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    frames_done = 0
    started = time.perf_counter()

    by_video: dict[str, list[tuple[int, list[Eligible]]]] = defaultdict(list)
    for (video_id, frame_index), items in by_frame.items():
        by_video[video_id].append((frame_index, items))

    for video_id in sorted(by_video):
        if video_id not in index:
            for _, items in by_video[video_id]:
                reasons["video_missing_from_archives"] += len(items)
            continue
        archive, prefix = index[video_id]
        with zipfile.ZipFile(raw_dir / "image" / archive) as zf:
            for frame_index, items in sorted(by_video[video_id]):
                member = f"{prefix}/{frame_index:02d}.jpg"
                try:
                    image = _read_frame(zf, member)
                except KeyError:
                    reasons["frame_missing_in_archive"] += len(items)
                    continue

                frame = Frame(
                    camera_id=video_id,
                    frame_index=frame_index,
                    timestamp=BASE_TS,
                    image=image,
                )
                detections = adapter.adapt_from(detector, frame)
                # Fresh tracker per frame: sampled frames are 0.5s apart and are not a
                # tracking sequence. Identity only; box geometry is unchanged (PROTOCOL 2.1).
                states = IouTracker().update(detections)
                associations = associate_riders(states, config=association_config)

                by_id = {s.track_id: s for s in states}
                motorcycles = [
                    s
                    for s in states
                    if s.object_class is ObjectClass.MOTORCYCLE and not s.tainted
                ]
                riders_of: dict[str, list[str]] = defaultdict(list)
                for assoc in associations:
                    riders_of[assoc.object_track_id].append(assoc.subject_track_id)

                used: set[str] = set()
                for item in sorted(items, key=lambda i: i.crop_id):
                    best_id, best_iou = None, 0.0
                    for moto in motorcycles:
                        if moto.track_id in used:
                            continue
                        value = iou(item.gt_box, moto.bbox)
                        if value > best_iou:
                            best_id, best_iou = moto.track_id, value
                    if best_id is None or best_iou < GT_MATCH_IOU:
                        reasons["no_motorcycle_detected_at_iou_0.50"] += 1
                        continue
                    used.add(best_id)

                    rider_ids = riders_of.get(best_id, [])
                    if not rider_ids:
                        reasons["motorcycle_matched_but_no_rider_associated"] += 1
                        continue
                    if len(rider_ids) > 1:
                        reasons["multiple_riders_associated_to_single_rider_gt"] += 1
                        continue

                    rider = by_id[rider_ids[0]]
                    region = extract_head_region(
                        rider.bbox, image, head_fraction=DEFAULT_HEAD_FRACTION
                    )
                    if region.image is None:
                        reasons["head_region_off_frame"] += 1
                        continue
                    if region.height_px < DEFAULT_MIN_CROP_HEIGHT_PX:
                        reasons["head_crop_below_min_height_gate"] += 1
                        continue

                    name = f"{item.crop_id}.png"
                    Image.fromarray(region.image, mode="RGB").save(
                        crops_out / name, format="PNG", compress_level=1
                    )
                    height, width = region.image.shape[0], region.image.shape[1]
                    records.append(
                        {
                            "crop_id": item.crop_id,
                            "split": item.split,
                            "video_id": item.video_id,
                            "frame_index": item.frame_index,
                            "track_id": item.track_id,
                            "site_id": item.site_id,
                            "label": item.label,
                            "file": name,
                            "crop_h": height,
                            "crop_w": width,
                            "head_height_px": region.height_px,
                            "gt_match_iou": best_iou,
                            "rider_score": rider.confidence,
                        }
                    )
                    reasons["recovered"] += 1
                frames_done += 1
                if frames_done % 100 == 0:
                    rate = frames_done / (time.perf_counter() - started)
                    print(
                        f"  {frames_done} frames | {reasons['recovered']} crops | "
                        f"{rate:.2f} fps",
                        flush=True,
                    )

    (out_dir / "crops.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
    )
    summary = {
        "eligible_single_rider": len(population),
        "distinct_frames": len(by_frame),
        "outcomes": dict(sorted(reasons.items())),
        "detector": DETECTOR_CHECKPOINT,
        "device": device,
        "score_threshold": SCORE_THRESHOLD,
        "gt_match_iou": GT_MATCH_IOU,
        "head_fraction": DEFAULT_HEAD_FRACTION,
        "min_crop_height_px": DEFAULT_MIN_CROP_HEIGHT_PX,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "derivation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "raw" / "helmet-myanmar"))
    parser.add_argument(
        "--crops", default=str(REPO_ROOT / "data" / "processed" / "helmet-cnnvit")
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", default="val,test")
    args = parser.parse_args()

    summary = derive(
        raw_dir=Path(args.raw),
        crop_dir=Path(args.crops),
        out_dir=Path(args.out),
        splits=tuple(args.splits.split(",")),
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

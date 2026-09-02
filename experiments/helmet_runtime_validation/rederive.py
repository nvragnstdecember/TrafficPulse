"""P4-U9: cut the head crops for P4-U8's corrected recovery population.

Reads the rider boxes P4-U8 recorded for every recovered annotation, re-opens the original
HELMET frames, and cuts each crop with the **production** ``extract_head_region``. The
output is a ``crops/`` directory plus a ``crops.jsonl`` in exactly the schema P4-U6-V's
``derive.py`` wrote, so ``score.py`` and ``analyse.py`` run over it unmodified.

The detector is **not** re-run. P4-U8 proved the offline detection record reproduces
P4-U6-V's recovered crop-id set exactly, so re-running RT-DETR would consume an hour to
produce the boxes already on disk -- and would introduce a second inference pass whose
agreement with the first would then itself need proving. Only pixels are read here.

The gates are re-applied against the real image rather than assumed: a crop that
``extract_head_region`` cannot cut, or that falls under the minimum-height floor, is
counted with its reason and never silently dropped. P4-U8 predicts zero such crops (its
gate replication is asserted against the production function in the tests); a non-zero
count here would be a discrepancy worth knowing about, so it is recorded either way.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.extract import build_frame_index  # noqa: E402

from helmet_runtime_validation.rematch import REASON_RECOVERED  # noqa: E402
from trafficpulse.contracts.primitives import BoundingBox  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    extract_head_region,
)


def recovered_rows(outcomes_path: Path) -> list[dict[str, Any]]:
    """The recovered outcomes of one split, in a deterministic order."""

    rows = [
        json.loads(line)
        for line in outcomes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(
        (row for row in rows if row["reason"] == REASON_RECOVERED),
        key=lambda row: row["crop_id"],
    )


def _read_frame(archive: zipfile.ZipFile, member: str) -> NDArray[np.uint8]:
    return np.asarray(
        Image.open(io.BytesIO(archive.read(member))).convert("RGB"), dtype=np.uint8
    )


def rederive(
    *, raw_dir: Path, source_dir: Path, out_dir: Path, splits: tuple[str, ...]
) -> dict[str, Any]:
    """Cut every recovered crop and write the P4-U6-V-shaped manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in splits:
        for row in recovered_rows(source_dir / f"outcomes_rider_inclusive_{split}.jsonl"):
            wanted[row["video_id"]].append(row)

    index = build_frame_index(raw_dir / "image")
    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    started = time.perf_counter()
    frames_done = 0

    for video_id in sorted(wanted):
        if video_id not in index:  # pragma: no cover - P4-U8 only lists processed frames
            reasons["video_missing_from_archives"] += len(wanted[video_id])
            continue
        archive_name, prefix = index[video_id]
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in wanted[video_id]:
            by_frame[row["frame_index"]].append(row)
        with zipfile.ZipFile(raw_dir / "image" / archive_name) as archive:
            for frame_index in sorted(by_frame):
                try:
                    image = _read_frame(archive, f"{prefix}/{frame_index:02d}.jpg")
                except KeyError:  # pragma: no cover - the frame was read once already
                    reasons["frame_missing_in_archive"] += len(by_frame[frame_index])
                    continue
                for row in by_frame[frame_index]:
                    box = row["rider_box"]
                    region = extract_head_region(
                        BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                        image,
                        head_fraction=DEFAULT_HEAD_FRACTION,
                    )
                    if region.image is None:  # pragma: no cover - P4-U8 gates these out
                        reasons["head_region_off_frame"] += 1
                        continue
                    if region.height_px < DEFAULT_MIN_CROP_HEIGHT_PX:  # pragma: no cover
                        reasons["head_crop_below_min_height_gate"] += 1
                        continue
                    name = f"{row['crop_id']}.png"
                    Image.fromarray(region.image, mode="RGB").save(
                        crops_dir / name, format="PNG", compress_level=1
                    )
                    records.append(
                        {
                            "crop_id": row["crop_id"],
                            "split": row["split"],
                            "video_id": row["video_id"],
                            "frame_index": row["frame_index"],
                            "track_id": row["track_id"],
                            "site_id": row["site_id"],
                            "label": row["label"],
                            "file": name,
                            "crop_h": int(region.image.shape[0]),
                            "crop_w": int(region.image.shape[1]),
                            "head_height_px": region.height_px,
                            "gt_match_iou": row["best_iou"],
                            "motorcycle_only_iou": row["motorcycle_only_iou"],
                            "gt_area": row["gt_area"],
                            "suspicious_match": row["suspicious_match"],
                        }
                    )
                    reasons["recovered"] += 1
                frames_done += 1

    records.sort(key=lambda record: str(record["crop_id"]))
    (out_dir / "crops.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "source": str(source_dir),
        "convention": "rider_inclusive (PROTOCOL_P4U8 section 3)",
        "splits": list(splits),
        "expected_from_p4u8": sum(len(rows) for rows in wanted.values()),
        "frames_read": frames_done,
        "outcomes": dict(sorted(reasons.items())),
        "per_split": dict(sorted(Counter(r["split"] for r in records).items())),
        "head_fraction": DEFAULT_HEAD_FRACTION,
        "min_crop_height_px": DEFAULT_MIN_CROP_HEIGHT_PX,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "derivation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "raw" / "helmet-myanmar"))
    parser.add_argument(
        "--source", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation_p4u8")
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation_p4u9")
    )
    parser.add_argument("--splits", default="val,test")
    args = parser.parse_args()

    summary = rederive(
        raw_dir=Path(args.raw),
        source_dir=Path(args.source),
        out_dir=Path(args.out),
        splits=tuple(args.splits.split(",")),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

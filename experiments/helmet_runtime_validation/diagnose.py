"""P4-U7 diagnostic pass: record every detection once, analyse every threshold offline.

RT-DETR is run over the P4-U6-V frames a single time with the score floor dropped to 0.01,
and every ``motorbike``/``person`` detection is written out with its score and box, next to
every annotated motorcycle in that frame. Nothing is decided here.

This shape is deliberate. Re-running the detector per candidate threshold would be both
slow and methodologically weak -- it invites stopping when a number looks good. With the
full low-threshold record on disk, each threshold in PROTOCOL_P4U7 §5 is evaluated by
filtering the same fixed evidence, so the grid is genuinely swept rather than searched.

Writes ``detection_dump.jsonl`` only. P4-U6-V's outputs are never touched.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.extract import build_frame_index  # noqa: E402

from helmet_runtime_validation.derive import (  # noqa: E402
    BASE_TS,
    DETECTOR_CHECKPOINT,
    LABEL_MAP,
    eligible_population,
)
from trafficpulse.contracts.enums import ObjectClass  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.frame import Frame  # noqa: E402

#: The floor for the diagnostic pass. Well below every grid point in §5, so the record can
#: answer any of them. Not an operating point.
DIAGNOSTIC_FLOOR = 0.01


def all_annotated_motorcycles(
    annotation_dir: Path, video_id: str
) -> dict[int, list[dict[str, object]]]:
    """``frame_id -> [{track_id, box}]`` for **every** annotated motorcycle in a clip.

    The full frame annotation, not the eligible subset: detection precision must be
    measured against every genuine motorcycle, including the multi-rider ones this project
    excludes from classification (PROTOCOL_P4U7 §5).
    """

    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    with (annotation_dir / f"{video_id}.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x, y, w, h = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            by_frame[int(row["frame_id"])].append(
                {
                    "track_id": row["track_id"],
                    "box": [x, y, x + w, y + h],
                    "label": row["label"],
                }
            )
    return by_frame


def _read_frame(zf: zipfile.ZipFile, member: str) -> NDArray[np.uint8]:
    return np.asarray(Image.open(io.BytesIO(zf.read(member))).convert("RGB"), dtype=np.uint8)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "raw" / "helmet-myanmar"))
    parser.add_argument(
        "--crops", default=str(REPO_ROOT / "data" / "processed" / "helmet-cnnvit")
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "helmet_detection_recovery"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    population = eligible_population(
        Path(args.crops), raw_dir / "annotation", ("val", "test")
    )
    if args.limit is not None:
        population = population[: args.limit]
    eligible_by_frame: dict[tuple[str, int], list] = defaultdict(list)
    for item in population:
        eligible_by_frame[(item.video_id, item.frame_index)].append(item)

    detector = RTDetrDetector(
        RTDetrConfig(
            checkpoint=DETECTOR_CHECKPOINT,
            device=args.device,
            local_files_only=True,
            threshold=DIAGNOSTIC_FLOOR,
        )
    )
    adapter = DetectionAdapter(
        DetectorConfig(label_map=LABEL_MAP, score_threshold=DIAGNOSTIC_FLOOR)
    )

    index = build_frame_index(raw_dir / "image")
    by_video: dict[str, list[int]] = defaultdict(list)
    for video_id, frame_index in eligible_by_frame:
        by_video[video_id].append(frame_index)

    destination = out_dir / "detection_dump.jsonl"
    written = 0
    started = time.perf_counter()
    with destination.open("w", encoding="utf-8") as sink:
        for video_id in sorted(by_video):
            if video_id not in index:
                continue
            archive, prefix = index[video_id]
            annotations = all_annotated_motorcycles(raw_dir / "annotation", video_id)
            with zipfile.ZipFile(raw_dir / "image" / archive) as zf:
                for frame_index in sorted(by_video[video_id]):
                    try:
                        image = _read_frame(zf, f"{prefix}/{frame_index:02d}.jpg")
                    except KeyError:
                        continue
                    frame = Frame(
                        camera_id=video_id,
                        frame_index=frame_index,
                        timestamp=BASE_TS,
                        image=image,
                    )
                    detections = adapter.adapt_from(detector, frame)
                    motorcycles = [
                        {
                            "box": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                            "score": d.confidence,
                        }
                        for d in detections
                        if d.object_class is ObjectClass.MOTORCYCLE
                    ]
                    persons = [
                        {
                            "box": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                            "score": d.confidence,
                        }
                        for d in detections
                        if d.object_class is ObjectClass.PERSON
                    ]
                    eligible = [
                        {
                            "crop_id": e.crop_id,
                            "split": e.split,
                            "track_id": e.track_id,
                            "label": e.label,
                            "site_id": e.site_id,
                            "box": [e.gt_box.x1, e.gt_box.y1, e.gt_box.x2, e.gt_box.y2],
                        }
                        for e in eligible_by_frame[(video_id, frame_index)]
                    ]
                    sink.write(
                        json.dumps(
                            {
                                "video_id": video_id,
                                "frame_index": frame_index,
                                "frame_w": int(image.shape[1]),
                                "frame_h": int(image.shape[0]),
                                "gt_all": annotations.get(frame_index, []),
                                "eligible": eligible,
                                "motorcycles": motorcycles,
                                "persons": persons,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    written += 1
                    if written % 200 == 0:
                        rate = written / (time.perf_counter() - started)
                        print(f"  {written} frames | {rate:.2f} fps", flush=True)

    print(
        json.dumps(
            {
                "frames": written,
                "diagnostic_floor": DIAGNOSTIC_FLOOR,
                "detector": DETECTOR_CHECKPOINT,
                "seconds": round(time.perf_counter() - started, 1),
                "dump": str(destination),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""P4-U10: qualitative real-video audit of the runtime helmet path, with visual evidence.

Strictly **look, do not tune** (PROTOCOL.md section 7). Nothing observed here changes a
label, a threshold, or a model, and nothing it produces enters any quantitative result. Its
only job is to make the runtime path *inspectable* on real footage so a demo-readiness
claim can be grounded in what the system actually does rather than in a metric.

It differs from ``qualitative.py`` in two ways that matter for an audit:

* **The tracker runs continuously over consecutive frames**, in short windows, which is what
  a deployment does. ``qualitative.py`` re-created the tracker per sampled frame because the
  HELMET corpus samples 0.5s apart and is not a tracking sequence; a real clip is one.
* **Every frame is rendered with its boxes, associations and predictions drawn on it**, so a
  reader can see which rider produced which call instead of taking a JSON row on trust.

It deliberately does **not** run the no-helmet violation rule. Doing so would require
acknowledging the turban capability guard for a binary backend, and switching that off to
make a demo run is exactly the bypass the guard exists to prevent. Classification is the
question here; the violation decision is not.

DeiT is not scored here -- it needs ``timm``, which lives in the research environment. The
manifest this writes is in ``score.py``'s schema, so DeiT is scored over the same crops by
running ``score.py --backend deit --split test`` against this output directory.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_runtime_validation.derive import (  # noqa: E402
    BASE_TS,
    DETECTOR_CHECKPOINT,
    LABEL_MAP,
    SCORE_THRESHOLD,
)
from trafficpulse.association.riders import RiderAssociationConfig, associate_riders  # noqa: E402
from trafficpulse.contracts import TrackState  # noqa: E402
from trafficpulse.contracts.enums import ObjectClass  # noqa: E402
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.frame import Frame  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    extract_head_region,
    head_region_box,
    rider_slot,
)
from trafficpulse.tracking import IouTracker  # noqa: E402

MOTORCYCLE_COLOUR = (64, 160, 255)
RIDER_COLOUR = (80, 220, 120)
HEAD_COLOUR = (255, 210, 60)
GATED_COLOUR = (255, 90, 90)

#: Nominal spacing between consecutive frames. Ordering only; see the Frame construction.
NOMINAL_FRAME_MS = 33


def windows_of(spec: str, length: int) -> list[tuple[int, int]]:
    """``"0,300,600"`` with ``length=40`` -> three 40-frame consecutive windows."""

    return [(int(start), int(start) + length) for start in spec.split(",") if start.strip()]


def decode_windows(
    video: Path, windows: list[tuple[int, int]]
) -> Iterator[tuple[int, int, NDArray[np.uint8]]]:
    """Yield ``(window_index, frame_index, rgb)`` for the requested frame ranges.

    One decode pass, in order: seeking a webm/ogv by frame index is unreliable, and the
    audit is small enough that reading forward is simpler and exactly reproducible.
    """

    import av

    last = max(end for _, end in windows)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= last:
                return
            for window, (start, end) in enumerate(windows):
                if start <= index < end:
                    yield window, index, np.asarray(
                        frame.to_image().convert("RGB"), dtype=np.uint8
                    )
                    break


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def draw_frame(
    image: NDArray[np.uint8],
    states: list[TrackState],
    riders_of: dict[str, list[str]],
    annotations: dict[str, dict[str, Any]],
) -> Image.Image:
    """Render one frame's detections, associations and predictions onto the pixels."""

    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, canvas.height // 60))
    small = _font(max(12, canvas.height // 75))
    by_id = {state.track_id: state for state in states}

    for state in states:
        if state.object_class is not ObjectClass.MOTORCYCLE:
            continue
        box = state.bbox
        count = len(riders_of.get(state.track_id, []))
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=MOTORCYCLE_COLOUR, width=3)
        draw.text(
            (box.x1 + 2, max(0.0, box.y1 - 18)),
            f"{state.track_id} riders={count} slot={rider_slot(count).value}",
            fill=MOTORCYCLE_COLOUR,
            font=small,
        )

    for motorcycle_id, rider_ids in riders_of.items():
        motorcycle = by_id[motorcycle_id]
        for rider_id in rider_ids:
            rider = by_id[rider_id]
            box = rider.bbox
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=RIDER_COLOUR, width=2)
            draw.line(
                [
                    (box.x1 + box.x2) / 2,
                    (box.y1 + box.y2) / 2,
                    (motorcycle.bbox.x1 + motorcycle.bbox.x2) / 2,
                    (motorcycle.bbox.y1 + motorcycle.bbox.y2) / 2,
                ],
                fill=RIDER_COLOUR,
                width=1,
            )
            head = head_region_box(box, head_fraction=DEFAULT_HEAD_FRACTION)
            annotation = annotations.get(rider_id)
            colour = HEAD_COLOUR if annotation is not None else GATED_COLOUR
            if head is not None:
                draw.rectangle([head.x1, head.y1, head.x2, head.y2], outline=colour, width=3)
            label = (
                f"{rider_id} R:{annotation['resnet_label']} {annotation['resnet_score']:.2f}"
                f" | Z:{annotation['zeroshot_label']} {annotation['zeroshot_score']:.2f}"
                f" | h={annotation['head_height_px']:.0f}px"
                if annotation is not None
                else f"{rider_id} GATED (head < {DEFAULT_MIN_CROP_HEIGHT_PX:.0f}px)"
            )
            draw.text((box.x1, max(0.0, box.y1 - 34)), label, fill=colour, font=font)
    return canvas


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", default="0,240,480")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-every", type=int, default=5)
    args = parser.parse_args()

    from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector

    out_dir = Path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    detector = RTDetrDetector(
        RTDetrConfig(
            checkpoint=DETECTOR_CHECKPOINT,
            device=args.device,
            local_files_only=True,
            threshold=SCORE_THRESHOLD,
        )
    )
    adapter = DetectionAdapter(
        DetectorConfig(label_map=LABEL_MAP, score_threshold=SCORE_THRESHOLD)
    )
    from helmet_runtime_validation.score import RESNET_CHECKPOINT, ZEROSHOT_CHECKPOINT
    from trafficpulse.classifier import (
        ResNetHelmetClassifier,
        ResNetHelmetConfig,
        ZeroShotHelmetClassifier,
        ZeroShotHelmetConfig,
    )
    from trafficpulse.classifier.resnet import NATIVE_LABELS

    zero_config = ZeroShotHelmetConfig(
        checkpoint=ZEROSHOT_CHECKPOINT, device=args.device, local_files_only=True
    )
    zero = ZeroShotHelmetClassifier(zero_config)
    zero_labels = list(zero_config.prompts)
    zero_prompts = [zero_config.prompts[name] for name in zero_labels]
    resnet = ResNetHelmetClassifier(
        ResNetHelmetConfig(checkpoint=RESNET_CHECKPOINT, device=args.device)
    )

    tracker = IouTracker()
    current_window = -1
    records: list[dict[str, Any]] = []
    frame_stats: list[dict[str, Any]] = []
    started = time.perf_counter()

    for window, frame_index, image in decode_windows(
        Path(args.video), windows_of(args.windows, args.length)
    ):
        if window != current_window:
            tracker.reset()
            current_window = window
        # The tracker requires a strictly increasing timestamp, so frames carry a
        # nominal media time derived from their index. Only the ordering is used --
        # nothing here depends on the clip's true frame rate.
        frame = Frame(
            camera_id="audit",
            frame_index=frame_index,
            timestamp=BASE_TS + timedelta(milliseconds=NOMINAL_FRAME_MS * frame_index),
            image=image,
        )
        detections = adapter.adapt_from(detector, frame)
        states = list(tracker.update(detections))
        associations = associate_riders(states, config=RiderAssociationConfig())
        by_id = {state.track_id: state for state in states}
        riders_of: dict[str, list[str]] = {}
        for association in associations:
            riders_of.setdefault(association.object_track_id, []).append(
                association.subject_track_id
            )

        crops: list[NDArray[np.uint8]] = []
        pending: list[dict[str, Any]] = []
        gated = 0
        for motorcycle_id, rider_ids in riders_of.items():
            for rider_id in rider_ids:
                rider = by_id[rider_id]
                region = extract_head_region(
                    rider.bbox, image, head_fraction=DEFAULT_HEAD_FRACTION
                )
                if region.image is None or region.height_px < DEFAULT_MIN_CROP_HEIGHT_PX:
                    gated += 1
                    continue
                name = f"w{window}_f{frame_index:05d}_{rider_id}.png"
                Image.fromarray(region.image, mode="RGB").save(out_dir / "crops" / name)
                crops.append(region.image)
                pending.append(
                    {
                        "crop_id": name.removesuffix(".png"),
                        "split": "test",
                        "video_id": Path(args.video).name,
                        "frame_index": frame_index,
                        "window": window,
                        "track_id": rider_id,
                        "motorcycle_track": motorcycle_id,
                        "riders_on_this_motorcycle": len(rider_ids),
                        "rider_slot": rider_slot(len(rider_ids)).value,
                        "head_height_px": round(region.height_px, 1),
                        "crop_h": int(region.image.shape[0]),
                        "crop_w": int(region.image.shape[1]),
                        "rider_confidence": rider.confidence,
                        "label": "unlabelled",
                        "file": name,
                    }
                )

        annotations: dict[str, dict[str, Any]] = {}
        if crops:
            zero_rows = zero._engine.infer(crops, zero_prompts)  # noqa: SLF001
            resnet_rows = resnet._engine.infer(crops)  # noqa: SLF001
            for entry, zero_row, resnet_row in zip(pending, zero_rows, resnet_rows, strict=True):
                zero_scores = {
                    name: round(float(v), 4)
                    for name, v in zip(zero_labels, zero_row, strict=True)
                }
                resnet_scores = {
                    name: round(float(v), 4)
                    for name, v in zip(NATIVE_LABELS, resnet_row, strict=True)
                }
                entry["zeroshot"] = zero_scores
                entry["resnet"] = resnet_scores
                zero_label = max(zero_scores, key=lambda k: zero_scores[k])
                resnet_label = max(resnet_scores, key=lambda k: resnet_scores[k])
                entry["zeroshot_label"] = zero_label
                entry["resnet_label"] = resnet_label
                annotations[str(entry["track_id"])] = {
                    "zeroshot_label": zero_label,
                    "zeroshot_score": zero_scores[zero_label],
                    "resnet_label": resnet_label,
                    "resnet_score": resnet_scores[resnet_label],
                    "head_height_px": entry["head_height_px"],
                }
            records.extend(pending)

        frame_stats.append(
            {
                "frame_index": frame_index,
                "window": window,
                "motorcycles": sum(
                    1 for s in states if s.object_class is ObjectClass.MOTORCYCLE
                ),
                "persons": sum(1 for s in states if s.object_class is ObjectClass.PERSON),
                "associated_riders": sum(len(v) for v in riders_of.values()),
                "ridden_motorcycles": len(riders_of),
                "crops": len(crops),
                "gated_crops": gated,
            }
        )

        if frame_index % args.render_every == 0:
            draw_frame(image, states, riders_of, annotations).save(
                out_dir / "frames" / f"w{window}_f{frame_index:05d}.jpg", quality=88
            )
        print(
            f"  f{frame_index:05d} moto={frame_stats[-1]['motorcycles']} "
            f"person={frame_stats[-1]['persons']} crops={len(crops)} gated={gated}",
            flush=True,
        )

    (out_dir / "crops.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "video": args.video,
        "windows": args.windows,
        "window_length": args.length,
        "frames": len(frame_stats),
        "crops": len(records),
        "gated_crops": sum(int(s["gated_crops"]) for s in frame_stats),
        "frames_with_no_motorcycle": sum(1 for s in frame_stats if s["motorcycles"] == 0),
        "frames_with_motorcycle_but_no_association": sum(
            1 for s in frame_stats if s["motorcycles"] and not s["ridden_motorcycles"]
        ),
        "multi_rider_crops": sum(
            1 for r in records if int(r["riders_on_this_motorcycle"]) > 1
        ),
        "crops_under_30px": sum(1 for r in records if float(r["head_height_px"]) < 30.0),
        "seconds": round(time.perf_counter() - started, 1),
        "per_frame": frame_stats,
        "records": records,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    headline = {k: v for k, v in summary.items() if k not in ("per_frame", "records")}
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

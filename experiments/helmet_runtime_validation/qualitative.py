"""Qualitative smoke test on real video, through the same runtime crop path.

Strictly **look, do not tune** (PROTOCOL §7): this produces crops and per-backend
predictions from real footage so a human can see what the models are actually being shown.
No label, threshold, or model is changed on the basis of anything seen here, and nothing it
produces enters the quantitative result.

It deliberately does **not** run the no-helmet rule. The rule would require the turban
capability guard to be acknowledged for a binary backend, and switching that on to make a
demo run is exactly the bypass the guard exists to prevent. Classification is the question
here; the violation decision is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
from trafficpulse.detector.adapter import DetectionAdapter  # noqa: E402
from trafficpulse.detector.config import DetectorConfig  # noqa: E402
from trafficpulse.detector.frame import Frame  # noqa: E402
from trafficpulse.observations.helmet import (  # noqa: E402
    DEFAULT_HEAD_FRACTION,
    DEFAULT_MIN_CROP_HEIGHT_PX,
    extract_head_region,
)
from trafficpulse.tracking import IouTracker  # noqa: E402


def frames_from(video: Path, *, stride: int, limit: int):
    """Decode frames through the production PyAV ingestion path."""

    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index % stride:
                continue
            yield index, np.asarray(frame.to_image().convert("RGB"), dtype=np.uint8)
            limit -= 1
            if limit <= 0:
                return


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from trafficpulse.detector.rtdetr import RTDetrConfig, RTDetrDetector

    out_dir = Path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

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

    records, images = [], []
    for index, image in frames_from(Path(args.video), stride=args.stride, limit=args.limit):
        frame = Frame(camera_id="qual", frame_index=index, timestamp=BASE_TS, image=image)
        states = IouTracker().update(adapter.adapt_from(detector, frame))
        associations = associate_riders(states, config=RiderAssociationConfig())
        by_id = {s.track_id: s for s in states}
        riders_per_bike: dict[str, list[str]] = {}
        for assoc in associations:
            riders_per_bike.setdefault(assoc.object_track_id, []).append(assoc.subject_track_id)

        for bike_id, rider_ids in riders_per_bike.items():
            for rider_id in rider_ids:
                rider = by_id[rider_id]
                region = extract_head_region(
                    rider.bbox, image, head_fraction=DEFAULT_HEAD_FRACTION
                )
                if region.image is None or region.height_px < DEFAULT_MIN_CROP_HEIGHT_PX:
                    continue
                name = f"f{index:05d}_{rider_id}.png"
                Image.fromarray(region.image, mode="RGB").save(out_dir / "crops" / name)
                images.append(region.image)
                records.append(
                    {
                        "file": name,
                        "frame_index": index,
                        "motorcycle_track": bike_id,
                        "rider_track": rider_id,
                        "riders_on_this_motorcycle": len(rider_ids),
                        "head_height_px": round(region.height_px, 1),
                        "crop_h": int(region.image.shape[0]),
                        "crop_w": int(region.image.shape[1]),
                        "rider_confidence": rider.confidence,
                    }
                )
        print(f"  frame {index}: {len(records)} crops so far", flush=True)

    if not records:
        print("no rider crops recovered from this clip")
        (out_dir / "qualitative.json").write_text(
            json.dumps({"video": args.video, "crops": []}, indent=2), encoding="utf-8"
        )
        return 0

    # Score with the two backends available in this environment. DeiT needs timm and is
    # scored separately in the research environment, exactly as in the controlled run.
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
    labels = list(zero_config.prompts)
    zero_rows = zero._engine.infer(images, [zero_config.prompts[n] for n in labels])  # noqa: SLF001

    resnet = ResNetHelmetClassifier(
        ResNetHelmetConfig(checkpoint=RESNET_CHECKPOINT, device=args.device)
    )
    resnet_rows = resnet._engine.infer(images)  # noqa: SLF001

    for record, zero_row, resnet_row in zip(records, zero_rows, resnet_rows, strict=True):
        record["zeroshot"] = {n: round(float(v), 4) for n, v in zip(labels, zero_row, strict=True)}
        record["resnet"] = {
            n: round(float(v), 4) for n, v in zip(NATIVE_LABELS, resnet_row, strict=True)
        }

    (out_dir / "qualitative.json").write_text(
        json.dumps({"video": args.video, "crops": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'qualitative.json'} ({len(records)} crops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

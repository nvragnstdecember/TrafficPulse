"""Score the derived runtime crops with each helmet backend.

Every backend sees the **same** PNG files, decoded identically, so any difference in the
numbers is a property of the model and not of the input. Predictions are written per
backend to ``scores_<backend>_<split>.json`` and never overwritten in place, so a scoring
run can be inspected after the fact.

Backends
--------
* ``zeroshot`` -- the production :class:`ZeroShotHelmetClassifier` (CLIP), through the real
  seam. Turban-capable; see PROTOCOL §8 for how ``turban`` is handled.
* ``resnet`` -- the production :class:`ResNetHelmetClassifier` (torchvision), i.e. exactly
  the code path a deployment would run.
* ``deit`` -- research-only. There is no production DeiT backend, and the P4-U5 DeiT
  checkpoint is a ``timm``-style ViT whose keys torchvision cannot load. It is scored here
  through ``timm`` in the research environment so the ViT direction is evaluated on the
  same crops rather than being dropped for want of a production wrapper.

The raw posterior is always recorded. Abstention is applied later, at analysis time, so a
threshold can be chosen on val without re-running any model.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)

#: P4-U5 seed-0 checkpoints. Seed 0 is fixed a priori (PROTOCOL §6.1): this experiment
#: evaluates one checkpoint per family and makes no seed-consistency claim.
RESNET_CHECKPOINT = (
    REPO_ROOT / "runs/helmet_cnn_vit/final/resnet50_lr0.001_s0/checkpoints/best.pt"
)
DEIT_CHECKPOINT = (
    REPO_ROOT / "runs/helmet_cnn_vit/final/deit_small_lr0.0001_s0/checkpoints/best.pt"
)
ZEROSHOT_CHECKPOINT = "openai/clip-vit-base-patch32"


def load_manifest(out_dir: Path, split: str) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in (out_dir / "crops.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["split"] == split]


def load_images(out_dir: Path, rows: Sequence[dict[str, object]]) -> list[NDArray[np.uint8]]:
    crops = out_dir / "crops"
    return [
        np.asarray(Image.open(crops / str(r["file"])).convert("RGB"), dtype=np.uint8)
        for r in rows
    ]


def _crops(rows: Sequence[dict[str, object]], images: Sequence[NDArray[np.uint8]]):
    from trafficpulse.classifier import Crop

    return [
        Crop(
            camera_id=str(row["video_id"]),
            frame_index=int(row["frame_index"]),
            timestamp=BASE_TS,
            track_id=str(row["track_id"]),
            image=image,
        )
        for row, image in zip(rows, images, strict=True)
    ]


def score_zeroshot(rows, images, *, device: str = "cpu", batch: int = 32) -> list[dict]:
    """Production zero-shot backend, recording the **full** prompt posterior.

    The classifier's public ``classify`` returns only the arg-max label and its score,
    which is all the runtime needs but is not enough to compare backends fairly: without
    the whole distribution there is no way to ask what zero-shot would have said if
    restricted to the binary vocabulary the P4-U5 models are limited to. The engine seam is
    therefore used directly, exactly as the ResNet scorer does, and the full row is stored.
    Nothing about the backend is modified.
    """

    from trafficpulse.classifier import ZeroShotHelmetClassifier, ZeroShotHelmetConfig

    config = ZeroShotHelmetConfig(
        checkpoint=ZEROSHOT_CHECKPOINT, device=device, local_files_only=True
    )
    classifier = ZeroShotHelmetClassifier(config)
    labels = list(config.prompts)
    prompts = [config.prompts[name] for name in labels]
    engine = classifier._engine  # noqa: SLF001 - full posterior, not the arg-max
    print(f"  zeroshot vocabulary: {labels}", flush=True)

    out: list[dict] = []
    for start in range(0, len(images), batch):
        chunk = list(images[start : start + batch])
        scored = engine.infer(chunk, prompts)
        for row, scores in zip(rows[start : start + batch], scored, strict=True):
            out.append(
                {
                    "crop_id": row["crop_id"],
                    "probabilities": {
                        name: float(value) for name, value in zip(labels, scores, strict=True)
                    },
                }
            )
        print(f"  zeroshot {min(start + batch, len(images))}/{len(images)}", flush=True)
    return out


def score_resnet(rows, images, *, device: str = "cpu", batch: int = 64) -> list[dict]:
    """Production ResNet backend (torchvision) -- the real deployment code path.

    Scored with ``abstain_below=None`` so the raw winning posterior is recorded; the
    abstention threshold is applied at analysis time from a val-chosen value.
    """

    from trafficpulse.classifier import ResNetHelmetClassifier, ResNetHelmetConfig
    from trafficpulse.classifier.resnet import NATIVE_LABELS

    classifier = ResNetHelmetClassifier(
        ResNetHelmetConfig(checkpoint=RESNET_CHECKPOINT, device=device)
    )
    engine = classifier._engine  # noqa: SLF001 - we want the full posterior, not the argmax
    out: list[dict] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        scored = engine.infer(chunk)
        for row, probabilities in zip(rows[start : start + batch], scored, strict=True):
            out.append(
                {
                    "crop_id": row["crop_id"],
                    "probabilities": {
                        name: float(p) for name, p in zip(NATIVE_LABELS, probabilities, strict=True)
                    },
                }
            )
        print(f"  resnet {min(start + batch, len(images))}/{len(images)}", flush=True)
    return out


def score_deit(rows, images, *, device: str = "cpu", batch: int = 64) -> list[dict]:
    """Research-only DeiT scoring through ``timm`` (no production DeiT backend exists).

    Preprocessing is the *same* square-pad + BILINEAR 224 geometry the production ResNet
    backend applies, with DeiT's own ImageNet normalisation, so the two models differ only
    in architecture and weights -- not in how the crop reached them.
    """

    import timm
    import torch

    from trafficpulse.classifier.resnet import square_pad_resize

    model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False, num_classes=2)
    payload = torch.load(DEIT_CHECKPOINT, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    config = model.pretrained_cfg
    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)
    print(f"  deit normalisation mean={config['mean']} std={config['std']}", flush=True)

    out: list[dict] = []
    for start in range(0, len(images), batch):
        chunk = [square_pad_resize(image) for image in images[start : start + batch]]
        array = (np.stack(chunk).astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2))).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor).float(), dim=-1).cpu().tolist()
        for row, probability in zip(rows[start : start + batch], probabilities, strict=True):
            out.append(
                {
                    "crop_id": row["crop_id"],
                    "probabilities": {
                        "helmet": float(probability[0]),
                        "no_helmet": float(probability[1]),
                    },
                }
            )
        print(f"  deit {min(start + batch, len(images))}/{len(images)}", flush=True)
    return out


SCORERS = {"zeroshot": score_zeroshot, "resnet": score_resnet, "deit": score_deit}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "helmet_runtime_validation"))
    parser.add_argument("--backend", required=True, choices=sorted(SCORERS))
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows = load_manifest(out_dir, args.split)
    print(f"{args.backend} / {args.split}: {len(rows)} crops", flush=True)
    images = load_images(out_dir, rows)
    predictions = SCORERS[args.backend](rows, images, device=args.device)

    destination = out_dir / f"scores_{args.backend}_{args.split}.json"
    destination.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "split": args.split,
                "count": len(predictions),
                "predictions": predictions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line driver for the CNN-vs-ViT helmet experiment (P4-U5).

Run from the repository root with the experiment interpreter, e.g.::

    .venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit prepare
    .venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit train --model resnet50 --seed 0

``experiments/`` must be on ``sys.path``; the ``prepare``/``train``/``evaluate``
subcommands add it themselves when invoked as a script from the repo root.

A real ``__main__`` module (rather than a heredoc) also matters operationally: the
DataLoader's Windows workers use the ``spawn`` start method, which re-imports the
parent's ``__main__``, and that cannot be done for a script piped through stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "experiments") not in sys.path:  # pragma: no cover - script entry
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helmet_cnn_vit.corpus import SamplingPolicy, build_corpus  # noqa: E402
from helmet_cnn_vit.extract import (  # noqa: E402
    ExtractionConfig,
    build_frame_index,
    extract_crops,
)
from helmet_cnn_vit.official_split import build_official_split  # noqa: E402

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "helmet-myanmar"
DEFAULT_CROPS = REPO_ROOT / "data" / "processed" / "helmet-cnnvit"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits" / "helmet-cnnvit"


def cmd_prepare(args: argparse.Namespace) -> int:
    """Build the corpus, apply the official split, and harvest crops."""

    raw = Path(args.raw)
    policy = SamplingPolicy(
        frame_stride=args.frame_stride,
        max_crops_per_track=args.max_crops_per_track,
    )
    corpus = build_corpus(raw / "annotation", policy=policy)
    print(f"corpus: {len(corpus)} crops from {corpus.statistics.tracks} tracks")
    print(f"        class balance {corpus.statistics.crops_per_class}")

    splits, manifest, statistics = build_official_split(corpus, raw / "data_split.csv")
    splits_dir = Path(args.splits)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (splits_dir / "statistics.json").write_text(
        statistics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (splits_dir / "manifest.sha256").write_text(manifest.content_hash() + "\n", encoding="utf-8")
    for name, counts in manifest.counts.items():
        print(f"  {name:5s} {counts.crops:6d} crops  {counts.crops_per_class}")
    print(f"manifest sha256: {manifest.content_hash()}")

    if args.no_extract:
        return 0

    crop_dir = Path(args.crops)
    index = build_frame_index(raw / "image")
    missing_clips = sorted({r.video_id for r in corpus.records} - set(index))
    if missing_clips:
        print(
            f"REFUSING to extract: {len(missing_clips)} clip(s) are not in any archive, "
            f"e.g. {missing_clips[:3]}. Complete the part_*.zip downloads first.",
            file=sys.stderr,
        )
        return 2

    report = extract_crops(
        splits,
        image_dir=raw / "image",
        output_dir=crop_dir,
        config=ExtractionConfig(image_size=args.image_size),
        frame_index=index,
    )
    print(
        f"extracted {report.crops_written} crops from {report.frames_read} frames "
        f"({len(report.missing_frames)} missing)"
    )
    if report.missing_frames or report.crops_written != len(corpus):
        print(
            f"REFUSING to continue: expected {len(corpus)} crops, wrote "
            f"{report.crops_written}; missing members {report.missing_frames[:5]}",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train one model at one seed."""

    from helmet_cnn_vit.train import TrainConfig, train_one

    config = TrainConfig(
        model=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    result = train_one(config, crop_dir=Path(args.crops), runs_root=Path(args.runs))
    print(
        f"{config.run_name()}: best val macro-F1 {result.best_val_macro_f1:.4f} "
        f"at epoch {result.best_epoch} ({result.train_seconds:.0f}s, {result.device})"
    )
    print(json.dumps({"run_dir": result.run_dir, "parameters": result.parameters}))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the pre-registered tuning grid and the final seed runs for both families."""

    from helmet_cnn_vit.sweep import run_sweep

    record, finals = run_sweep(
        crop_dir=Path(args.crops),
        runs_root=Path(args.runs),
        num_workers=args.num_workers,
    )
    for model, results in sorted(finals.items()):
        scores = [r.best_val_macro_f1 for r in results]
        print(f"{model}: lr={record.selections[model].chosen_lr:g} val macro-F1 per seed {scores}")
    return 0


DEVIATIONS: tuple[str, ...] = (
    "Binary helmet/no_helmet rather than the 3-class task: HELMET carries neither an "
    "`uncertain` nor a `turban` label, and inventing either would fabricate supervision.",
    "HELMET only: AI City Track 5 is recorded PROPRIETARY/REJECTED and custom Indian "
    "footage has no ethics or permission clearance on file.",
    "Per-site test slices rather than a whole-site holdout: the authors' official split is "
    "video-level and shares all 12 sites across splits, and dataset-policy requires "
    "preserving it.",
    "3 val-selected configs per family rather than up to 8; the budget is reduced but equal.",
    "Motorcycle crops rather than head crops, which is the annotation's native granularity.",
    "No mixup/CutMix for either family (DeiT's published recipe includes them); omitted "
    "symmetrically because soft targets would change the loss path for one family only.",
    "ConvNeXt-T / Swin-T ablation pair skipped (architecture-review §12 marks it optional).",
)

LIMITATIONS: tuple[str, ...] = (
    "Myanmar footage recorded in 2016, daytime only. No claim of transfer to Indian roads.",
    "Day/night slices are not reported: the corpus has no illumination annotation.",
    "Sites are shared across the official split, so this measures video-level rather than "
    "site-level generalisation; per-site slices are reported instead.",
    "Trained on motorcycle crops while the runtime classifies derived head crops: the "
    "winner is measured, not deployed.",
    "The §12 absolute crop-height buckets are degenerate on this corpus (HELMET boxes are "
    "whole motorcycles at 1080p); train-derived tertiles are reported alongside them.",
    "No comparison against the HELMET authors' published numbers: the task framing differs "
    "from their weighted F-measure.",
)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Assemble results.json from the finished runs."""

    import json as _json

    from helmet_cnn_vit.evaluate import build_model_result, compare, load_selection
    from helmet_cnn_vit.report import ExperimentReport, Provenance, git_sha, save_report

    runs_root = Path(args.runs)
    crop_dir = Path(args.crops)
    selection = load_selection(runs_root)

    benchmarks: dict[str, dict[str, object]] = {}
    if args.bench:
        from helmet_cnn_vit.bench import benchmark_model
        from helmet_cnn_vit.models import MODEL_SPECS

        for name, spec in sorted(MODEL_SPECS.items()):
            result = benchmark_model(spec, iters=args.bench_iters)
            benchmarks[name] = result.model_dump()
            print(f"[bench] {name}: {result.batches[0].median_latency_ms:.2f} ms @ batch 1")

    models = {
        name: build_model_result(
            name,
            runs_root=runs_root,
            crop_dir=crop_dir,
            selection=selection[name],
            benchmark=benchmarks.get(name),
        )
        for name in sorted(selection)
    }
    first, second = sorted(models)
    verdict = compare(models[first], models[second], runs_root=runs_root)

    manifest = _json.loads(
        (Path(args.splits) / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = Provenance(
        dataset_id=manifest["dataset_id"],
        dataset_version=manifest["dataset_version"],
        dataset_licence="CC-BY-4.0",
        dataset_attribution=(
            "Siebert, F.W. and Lin, H. - HELMET dataset (OSF, https://osf.io/4pwj8/), CC-BY 4.0"
        ),
        split_manifest_sha256=(Path(args.splits) / "manifest.sha256")
        .read_text(encoding="utf-8")
        .strip(),
        corpus_sha256=manifest["corpus_hash"],
        sampling_policy=manifest["sampling_policy"],
        git_sha=git_sha(REPO_ROOT),
        torch_version=_torch_version(),
        timm_version=_timm_version(),
        device=_device_name(),
        commands=(
            "python -m helmet_cnn_vit prepare",
            "python -m helmet_cnn_vit sweep --num-workers 6",
            "python -m helmet_cnn_vit evaluate --bench",
        ),
    )
    report = ExperimentReport(
        title="CNN vs ViT for motorcycle driver helmet-state classification",
        generated_at=None,
        preregistration="experiments/helmet_cnn_vit/PREREGISTRATION.md",
        provenance=provenance,
        models=models,
        verdict=verdict,
        deviations=DEVIATIONS,
        limitations=LIMITATIONS,
    )
    written = save_report(report, Path(args.out))
    print(verdict.rationale)
    for key, path in sorted(written.items()):
        print(f"  {key}: {path}")
    return 0


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _timm_version() -> str | None:
    try:
        import timm
    except ImportError:  # pragma: no cover - optional extra
        return None
    return str(timm.__version__)


def _device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helmet_cnn_vit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build corpus + official split + crops")
    prepare.add_argument("--raw", default=str(DEFAULT_RAW))
    prepare.add_argument("--crops", default=str(DEFAULT_CROPS))
    prepare.add_argument("--splits", default=str(DEFAULT_SPLITS))
    prepare.add_argument("--frame-stride", type=int, default=5)
    prepare.add_argument("--max-crops-per-track", type=int, default=6)
    prepare.add_argument("--image-size", type=int, default=224)
    prepare.add_argument("--no-extract", action="store_true", help="split only, no images")
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="train one model at one seed")
    train.add_argument("--model", required=True, choices=["resnet50", "deit_small"])
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--num-workers", type=int, default=3)
    train.add_argument("--crops", default=str(DEFAULT_CROPS))
    train.add_argument("--runs", default=str(REPO_ROOT / "runs" / "helmet_cnn_vit"))
    train.set_defaults(func=cmd_train)

    sweep = sub.add_parser("sweep", help="pre-registered tuning grid + final seed runs")
    sweep.add_argument("--crops", default=str(DEFAULT_CROPS))
    sweep.add_argument("--runs", default=str(REPO_ROOT / "runs" / "helmet_cnn_vit"))
    sweep.add_argument("--num-workers", type=int, default=3)
    sweep.set_defaults(func=cmd_sweep)

    evaluate = sub.add_parser("evaluate", help="assemble results.json from finished runs")
    evaluate.add_argument("--crops", default=str(DEFAULT_CROPS))
    evaluate.add_argument("--splits", default=str(DEFAULT_SPLITS))
    evaluate.add_argument("--runs", default=str(REPO_ROOT / "runs" / "helmet_cnn_vit"))
    evaluate.add_argument("--out", default=str(REPO_ROOT / "runs" / "helmet_cnn_vit" / "report"))
    evaluate.add_argument("--bench", action="store_true", help="also measure latency/VRAM")
    evaluate.add_argument("--bench-iters", type=int, default=1000)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

"""Evaluation and report assembly from finished runs (P4-U5).

Reads the per-crop predictions each training run wrote, computes the full §12
metric set on the shared test split, fits temperature scaling on validation and
applies it to test, computes the robustness slices that need no re-inference, runs
the pre-committed statistics, and hands the result to
:mod:`helmet_cnn_vit.report`.

Why this stage re-reads predictions instead of re-running models
----------------------------------------------------------------
Every training run writes ``artifacts/predictions_{val,test}.json`` -- the crop ids,
the truth, and the model's ``no_helmet`` probability for each crop. Everything below
is a pure function of those files. That means the whole analysis is reproducible
without a GPU, is cheap to re-run when a metric definition changes, and cannot
accidentally evaluate a different checkpoint than the one that was selected.

Slices over crop metadata (site, crop height) likewise need no re-inference: the
crop index carries the metadata, and the predictions carry the crop ids, so a slice
is an index selection. Corruption slices *do* need re-inference and are handled
separately by the corruption pass, which is opt-in because it costs GPU time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .calibrate import apply_temperature, fit_temperature
from .datasets import load_rows
from .errors import CnnVitError
from .metrics import compute_metrics, predictions_from_scores
from .report import ModelResult, SeedResult, SliceResult
from .robustness import SPEC_HEIGHT_EDGES, TRAIN_HEIGHT_TERTILES, height_bucket
from .stats import bootstrap_delta_macro_f1_pooled, decide, mcnemar
from .train import SplitPredictions, load_predictions


def _seed_dirs(runs_root: Path, model: str) -> list[Path]:
    """Every final run directory for one model, ordered by seed."""

    final = runs_root / "final"
    if not final.is_dir():
        raise CnnVitError(f"no final runs under {final}; run the sweep first")
    dirs = sorted(d for d in final.iterdir() if d.is_dir() and d.name.startswith(f"{model}_"))
    if not dirs:
        raise CnnVitError(f"no final runs for {model!r} under {final}")
    return dirs


def _run_metrics(run_dir: Path) -> dict[str, object]:
    payload = json.loads((run_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload)


def evaluate_seed(run_dir: Path) -> SeedResult:
    """Compute the §12 metric set for one finished run, calibrated and uncalibrated."""

    metrics_payload = _run_metrics(run_dir)
    config = dict(metrics_payload["config"])  # type: ignore[arg-type]

    val = load_predictions(run_dir, "val")
    test = load_predictions(run_dir, "test")

    uncalibrated = compute_metrics(
        list(test.truth),
        list(test.predicted),
        no_helmet_scores=list(test.no_helmet_scores),
        confidences=list(test.confidences),
    )

    # Temperature is fitted on VALIDATION and applied unchanged to test.
    fit = fit_temperature(list(val.truth), list(val.no_helmet_scores))
    scaled = apply_temperature(list(test.no_helmet_scores), fit.temperature)
    scaled_labels, scaled_conf = predictions_from_scores(scaled)
    calibrated = compute_metrics(
        list(test.truth),
        list(scaled_labels),
        no_helmet_scores=list(scaled),
        confidences=list(scaled_conf),
    )

    return SeedResult(
        seed=int(config["seed"]),
        run_dir=str(run_dir),
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        best_epoch=int(metrics_payload["best_epoch"]),  # type: ignore[arg-type]
        val_macro_f1=metrics_payload.get("best_val_macro_f1"),  # type: ignore[arg-type]
        test=uncalibrated,
        test_calibrated=calibrated,
        temperature=fit,
        train_seconds=float(metrics_payload["train_seconds"]),  # type: ignore[arg-type]
    )


def compute_slices(
    predictions: SplitPredictions, crop_dir: Path, *, split: str = "test"
) -> list[SliceResult]:
    """Metrics per observation site and per crop-height bucket.

    Both slices are pure index selections over the crop index, so they cost nothing
    and cannot disagree with the main metrics about which crops they cover.
    """

    rows = {row.crop_id: row for row in load_rows(crop_dir, split)}
    missing = [cid for cid in predictions.crop_ids if cid not in rows]
    if missing:
        raise CnnVitError(f"{len(missing)} predicted crops are absent from the index")

    groups: dict[tuple[str, str], list[int]] = {}
    for index, crop_id in enumerate(predictions.crop_ids):
        row = rows[crop_id]
        groups.setdefault(("site", row.site_id), []).append(index)
        groups.setdefault(
            ("height_tertile", height_bucket(row.box_h, TRAIN_HEIGHT_TERTILES)), []
        ).append(index)
        groups.setdefault(
            ("height_spec_s12", height_bucket(row.box_h, SPEC_HEIGHT_EDGES)), []
        ).append(index)

    results: list[SliceResult] = []
    for (slice_name, bucket), indices in sorted(groups.items()):
        truth = [predictions.truth[i] for i in indices]
        predicted = [predictions.predicted[i] for i in indices]
        metrics = compute_metrics(truth, predicted)
        results.append(
            SliceResult(
                slice_name=slice_name,
                bucket=bucket,
                samples=len(indices),
                macro_f1=metrics.macro_f1,
                balanced_accuracy=metrics.balanced_accuracy,
                accuracy=metrics.accuracy,
            )
        )
    return results


def build_model_result(
    model: str,
    *,
    runs_root: Path,
    crop_dir: Path,
    selection: Mapping[str, object],
    benchmark: dict[str, object] | None = None,
) -> ModelResult:
    """Assemble one family's complete result across its seeds."""

    dirs = _seed_dirs(runs_root, model)
    seeds = [evaluate_seed(d) for d in dirs]
    first = _run_metrics(dirs[0])

    # Slices are computed on the first seed; per-seed slice tables would multiply
    # the report's size without changing what it supports.
    slices = compute_slices(load_predictions(dirs[0], "test"), crop_dir)

    return ModelResult(
        model=model,
        timm_id=str(first["timm_id"]),
        family=str(first["family"]),
        parameters=int(first["parameters"]),  # type: ignore[arg-type]
        checkpoint_bytes=int(first["checkpoint_bytes"]),  # type: ignore[arg-type]
        selected_lr=float(selection["chosen_lr"]),  # type: ignore[arg-type]
        tuning_grid=dict(selection["grid"]),  # type: ignore[arg-type]
        seeds=tuple(seeds),
        slices=tuple(slices),
        benchmark=benchmark,
    )


def compare(
    first: ModelResult,
    second: ModelResult,
    *,
    runs_root: Path,
    resamples: int = 10_000,
    seed: int = 0,
):
    """Apply the pre-committed §12 rule to two assembled model results.

    Per-seed deltas pair seed *i* of one family against seed *i* of the other. The
    bootstrap pools the seeds by using each family's first seed's predictions on the
    shared test split, which is the same set of crops for both.
    """

    per_seed: list[float] = []
    for a, b in zip(first.seeds, second.seeds, strict=False):
        if a.test.macro_f1 is None or b.test.macro_f1 is None:  # pragma: no cover
            raise CnnVitError("a seed produced no macro-F1; cannot apply the decision rule")
        per_seed.append(a.test.macro_f1 - b.test.macro_f1)

    a_runs = [load_predictions(Path(s.run_dir), "test") for s in first.seeds]
    b_runs = [load_predictions(Path(s.run_dir), "test") for s in second.seeds]
    reference = a_runs[0].crop_ids
    for predictions in (*a_runs, *b_runs):
        if predictions.crop_ids != reference:
            raise CnnVitError(
                "runs were evaluated on different crops; the comparison would not be paired"
            )

    # §12 asks for a POOLED interval: within each resample both families are scored
    # on the same crops and averaged over their seeds, so the interval carries
    # seed-to-seed variation as well as crop sampling. Bootstrapping one seed would
    # describe that seed's run and understate the uncertainty the claim must survive.
    interval = bootstrap_delta_macro_f1_pooled(
        list(a_runs[0].truth),
        [list(r.predicted) for r in a_runs],
        [list(r.predicted) for r in b_runs],
        resamples=resamples,
        seed=seed,
    )

    paired = [
        mcnemar(
            [t == p for t, p in zip(a.truth, a.predicted, strict=True)],
            [t == p for t, p in zip(b.truth, b.predicted, strict=True)],
        )
        for a, b in zip(a_runs, b_runs, strict=False)
    ]

    return decide(
        first_name=first.model,
        second_name=second.model,
        per_seed_delta=per_seed,
        interval=interval,
        mcnemar_results=paired,
    )


def load_selection(runs_root: Path) -> dict[str, dict[str, object]]:
    """Read the sweep's selection record (written before any final run began)."""

    path = runs_root / "selection.json"
    if not path.is_file():
        raise CnnVitError(f"no selection record at {path}; run the sweep first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: dict(v) for k, v in payload["selections"].items()}


def seed_order(results: Sequence[SeedResult]) -> tuple[int, ...]:
    """The seeds a family was evaluated at, in run order (for the report)."""

    return tuple(r.seed for r in results)

"""Result assembly for the CNN-vs-ViT comparison (P4-U5).

Collects the finished runs into one ``results.json`` that carries every number the
write-up is allowed to cite, plus the provenance needed to reproduce it: dataset id
and version, split-manifest hash, model and pretrained-weight ids, git SHA, configs,
seeds, and the exact commands.

Conventions inherited from H5's reporter
-----------------------------------------
* ``None`` means undefined, never a fabricated ``0.0``.
* ``generated_at`` is ``None`` unless a clock is injected -- time is never invented.
* Floats render at fixed precision in the CSV so a diff of two reports is readable.

The verdict is computed by :func:`helmet_cnn_vit.stats.decide` from the
pre-registered rule and is embedded here verbatim, including its rationale string.
Nothing in this module can promote a tie to a claim.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from helmet_rtdetr.models import _Model

from .calibrate import TemperatureFit
from .metrics import ClassificationMetrics
from .stats import ComparisonVerdict


class SeedResult(_Model):
    """One trained model at one seed, evaluated on the shared test split."""

    seed: int
    run_dir: str
    lr: float
    epochs: int
    best_epoch: int
    val_macro_f1: float | None
    test: ClassificationMetrics
    test_calibrated: ClassificationMetrics | None
    temperature: TemperatureFit | None
    train_seconds: float


class SliceResult(_Model):
    """One model's metrics on one named slice of the test split."""

    slice_name: str
    bucket: str
    samples: int
    macro_f1: float | None
    balanced_accuracy: float | None
    accuracy: float


class ModelResult(_Model):
    """Everything measured about one model family."""

    model: str
    timm_id: str
    family: str
    parameters: int
    checkpoint_bytes: int
    selected_lr: float
    tuning_grid: dict[str, float | None]
    seeds: tuple[SeedResult, ...]
    slices: tuple[SliceResult, ...]
    benchmark: dict[str, object] | None

    @property
    def test_macro_f1_per_seed(self) -> tuple[float | None, ...]:
        return tuple(s.test.macro_f1 for s in self.seeds)


class Provenance(_Model):
    """What must be recorded for the run to be reproducible."""

    dataset_id: str
    dataset_version: str
    dataset_licence: str
    dataset_attribution: str
    split_manifest_sha256: str
    corpus_sha256: str
    sampling_policy: dict[str, object]
    git_sha: str | None
    torch_version: str
    timm_version: str | None
    device: str
    commands: tuple[str, ...]


class ExperimentReport(_Model):
    """The complete artifact: what was measured, and what may be claimed."""

    title: str
    generated_at: datetime | None
    preregistration: str
    provenance: Provenance
    models: dict[str, ModelResult]
    verdict: ComparisonVerdict
    deviations: tuple[str, ...]
    limitations: tuple[str, ...]


def git_sha(repo_root: Path) -> str | None:
    """The current commit, or ``None`` if git cannot answer (never a guess)."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def summary_table(report: ExperimentReport) -> str:
    """A compact Markdown table of the headline result, for the write-up."""

    lines = [
        "| model | family | params | test macro-F1 (mean ± std) | balanced acc | PR-AUC | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, model in sorted(report.models.items()):
        scores = [s for s in model.test_macro_f1_per_seed if s is not None]
        if scores:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
            spread = f"{mean:.4f} ± {var ** 0.5:.4f}"
        else:  # pragma: no cover - a run with no macro-F1 is already a failure
            spread = "n/a"
        first = model.seeds[0].test if model.seeds else None
        balanced = f"{first.balanced_accuracy:.4f}" if first and first.balanced_accuracy else "n/a"
        pr_auc = f"{first.pr_auc_no_helmet:.4f}" if first and first.pr_auc_no_helmet else "n/a"
        ece = (
            f"{first.expected_calibration_error:.4f}"
            if first and first.expected_calibration_error is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {model.family} | {model.parameters / 1e6:.1f}M | {spread} | "
            f"{balanced} | {pr_auc} | {ece} |"
        )
    return "\n".join(lines)


def slices_table(model: ModelResult, slice_name: str) -> str:
    """A Markdown table of one model's metrics across one slice's buckets."""

    rows = [s for s in model.slices if s.slice_name == slice_name]
    lines = [f"| {slice_name} | n | macro-F1 | balanced acc |", "|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: r.bucket):
        macro = f"{row.macro_f1:.4f}" if row.macro_f1 is not None else "undefined"
        balanced = (
            f"{row.balanced_accuracy:.4f}" if row.balanced_accuracy is not None else "undefined"
        )
        lines.append(f"| {row.bucket} | {row.samples} | {macro} | {balanced} |")
    return "\n".join(lines)


def save_report(report: ExperimentReport, output_dir: Path) -> dict[str, Path]:
    """Write ``results.json`` plus a human-readable ``summary.md``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    results = output_dir / "results.json"
    results.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written["results"] = results

    parts = [
        f"# {report.title}",
        "",
        f"Pre-registration: `{report.preregistration}`",
        "",
        "## Headline",
        "",
        summary_table(report),
        "",
        "## Verdict (pre-committed rule)",
        "",
        report.verdict.rationale,
        "",
        f"- difference claimed: **{report.verdict.difference_claimed}**",
        f"- winner: **{report.verdict.winner or 'none (tie)'}**",
        f"- per-seed deltas: {list(report.verdict.per_seed_delta)}",
        f"- bootstrap 95% CI on delta macro-F1: "
        f"[{report.verdict.interval.lower:+.4f}, {report.verdict.interval.upper:+.4f}]"
        f" over {report.verdict.interval.resamples} resamples",
        "",
        "## Deviations from architecture-review section 12",
        "",
        *(f"- {d}" for d in report.deviations),
        "",
        "## Limitations",
        "",
        *(f"- {limitation}" for limitation in report.limitations),
        "",
    ]
    summary = output_dir / "summary.md"
    summary.write_text("\n".join(parts), encoding="utf-8")
    written["summary"] = summary
    return written


def metrics_csv(models: Mapping[str, ModelResult]) -> str:
    """Per-seed test metrics as CSV, at fixed precision (H5's convention)."""

    header = "model,family,seed,macro_f1,balanced_accuracy,accuracy,pr_auc_no_helmet,ece"
    lines = [header]
    for name, model in sorted(models.items()):
        for seed in model.seeds:
            t = seed.test
            lines.append(
                ",".join(
                    (
                        name,
                        model.family,
                        str(seed.seed),
                        _fmt(t.macro_f1),
                        _fmt(t.balanced_accuracy),
                        _fmt(t.accuracy),
                        _fmt(t.pr_auc_no_helmet),
                        _fmt(t.expected_calibration_error),
                    )
                )
            )
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    """Fixed 6 dp; an undefined metric renders as an empty cell, never as 0."""

    return "" if value is None else f"{value:.6f}"


def mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation over the defined values."""

    present = [v for v in values if v is not None]
    if not present:
        return None, None
    mean = sum(present) / len(present)
    if len(present) < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in present) / (len(present) - 1)
    return mean, variance**0.5

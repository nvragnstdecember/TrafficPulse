"""Deterministic evaluation reports: evaluation.json, summary.json, metrics.csv (H5).

Determinism strategy (the same one H1–H4 use)
---------------------------------------------
* Every report component is a frozen pydantic model whose collections are built
  in canonical order before construction, so ``model_dump_json`` is a pure
  function of evaluation content.
* Wall-clock time enters only through an injectable clock: ``generated_at``
  stays ``None`` unless the caller supplied one — time is never fabricated
  (the repo-wide honesty rule).
* The CSV renders floats at a fixed 6 decimal places, undefined values as
  empty cells, and rows in a fixed order (overall first, then classes in
  model-label-id order) — byte-identical for identical metrics.

``evaluation.json`` is the complete :class:`EvaluationReport`;
``summary.json`` is the compact headline (:class:`EvaluationSummary`);
``metrics.csv`` is the overall + per-class table for spreadsheet consumption.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field

from ..models import NonEmptyStr, _Model
from .confusion import ConfusionMatrix
from .metrics import ClassMetrics, EvaluationMetrics
from .models import EvaluationConfig

EVALUATION_SCHEMA_VERSION = "1.0.0"

_CSV_COLUMNS = (
    "scope",
    "ap",
    "ap50",
    "ap75",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "false_negatives",
    "num_ground_truth",
    "num_predictions",
)


class DatasetSummary(_Model):
    """What was evaluated: the image universe and its ground truth."""

    split: str | None = None  # H3 split name when known; None for ad-hoc inputs
    num_images: int = Field(ge=0)
    num_ground_truth: int = Field(ge=0)
    ground_truth_per_class: dict[str, int]


class CheckpointProvenance(_Model):
    """Which trained checkpoint produced the predictions (H4A/H4B identity)."""

    checkpoint_id: NonEmptyStr
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    roles: tuple[str, ...]
    metric_name: str | None = None
    metric_value: float | None = None
    run_dir: NonEmptyStr  # as given by the caller; relative stays relative


class EvaluationReport(_Model):
    """The complete, self-describing result of one evaluation."""

    schema_version: str = EVALUATION_SCHEMA_VERSION
    config: EvaluationConfig
    dataset: DatasetSummary
    metrics: EvaluationMetrics
    confusion_matrix: ConfusionMatrix
    checkpoint: CheckpointProvenance | None = None  # None for pure-prediction runs
    generated_at: datetime | None = None  # honest: only set from an injected clock


class EvaluationSummary(_Model):
    """The headline numbers (summary.json)."""

    schema_version: str = EVALUATION_SCHEMA_VERSION
    mean_ap: float | None
    ap50: float | None
    ap75: float | None
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    num_images: int
    num_ground_truth: int
    num_predictions: int
    checkpoint_id: str | None = None

    @classmethod
    def from_report(cls, report: EvaluationReport) -> EvaluationSummary:
        metrics = report.metrics
        return cls(
            schema_version=report.schema_version,
            mean_ap=metrics.mean_ap,
            ap50=metrics.ap50,
            ap75=metrics.ap75,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            true_positives=metrics.true_positives,
            false_positives=metrics.false_positives,
            false_negatives=metrics.false_negatives,
            num_images=report.dataset.num_images,
            num_ground_truth=report.dataset.num_ground_truth,
            num_predictions=metrics.num_predictions,
            checkpoint_id=(
                report.checkpoint.checkpoint_id if report.checkpoint is not None else None
            ),
        )


def _cell(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _csv_row(scope: str, metrics: ClassMetrics | EvaluationMetrics) -> str:
    ap = metrics.mean_ap if isinstance(metrics, EvaluationMetrics) else metrics.ap
    num_ground_truth = metrics.num_ground_truth
    return ",".join(
        (
            scope,
            _cell(ap),
            _cell(metrics.ap50),
            _cell(metrics.ap75),
            _cell(metrics.precision),
            _cell(metrics.recall),
            _cell(metrics.f1),
            _cell(metrics.true_positives),
            _cell(metrics.false_positives),
            _cell(metrics.false_negatives),
            _cell(num_ground_truth),
            _cell(metrics.num_predictions),
        )
    )


def metrics_csv(report: EvaluationReport) -> str:
    """The overall + per-class metric table as deterministic CSV text."""

    lines = [",".join(_CSV_COLUMNS), _csv_row("overall", report.metrics)]
    lines.extend(
        _csv_row(class_metrics.label.value, class_metrics)
        for class_metrics in report.metrics.per_class
    )
    return "\n".join(lines) + "\n"


def save_report(report: EvaluationReport, directory: Path) -> dict[str, Path]:
    """Write evaluation.json + summary.json + metrics.csv under ``directory``.

    Creates the directory (idempotent); returns the written paths keyed by
    artifact name — the same shape H3's ``export_splits`` returns.
    """

    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    evaluation_path = directory / "evaluation.json"
    evaluation_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written["evaluation"] = evaluation_path

    summary_path = directory / "summary.json"
    summary_path.write_text(
        EvaluationSummary.from_report(report).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    written["summary"] = summary_path

    csv_path = directory / "metrics.csv"
    csv_path.write_text(metrics_csv(report), encoding="utf-8")
    written["metrics_csv"] = csv_path
    return written

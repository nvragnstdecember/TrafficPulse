"""Report models + deterministic serialization: JSON, summary, CSV (H5)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from _eval_helpers import pred, unified
from helmet_rtdetr.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    HelmetEvaluator,
    metrics_csv,
    save_report,
)
from helmet_rtdetr.unified import UnifiedClass


def _report(**evaluator_kwargs: object) -> EvaluationReport:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("b.png", UnifiedClass.NO_HELMET, box=(5, 5, 10, 10)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10)),
        pred("b.png", UnifiedClass.HELMET, score=0.6, box=(50, 50, 10, 10)),
    ]
    evaluator = HelmetEvaluator(**evaluator_kwargs)  # type: ignore[arg-type]
    return evaluator.evaluate(predictions, ground_truth)


# --- files ------------------------------------------------------------------------
def test_save_report_writes_all_three_artifacts(tmp_path: Path) -> None:
    written = save_report(_report(), tmp_path / "eval")
    assert set(written) == {"evaluation", "summary", "metrics_csv"}
    for path in written.values():
        assert path.is_file()


def test_saved_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    first = save_report(_report(), tmp_path / "one")
    second = save_report(_report(), tmp_path / "two")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def test_evaluation_json_round_trips() -> None:
    report = _report()
    restored = EvaluationReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_report_from_shuffled_inputs_is_byte_identical() -> None:
    ground_truth = [
        unified("a.png", UnifiedClass.HELMET, box=(0, 0, 10, 10)),
        unified("b.png", UnifiedClass.NO_HELMET, box=(5, 5, 10, 10)),
    ]
    predictions = [
        pred("a.png", UnifiedClass.HELMET, score=0.9, box=(0, 0, 10, 10)),
        pred("b.png", UnifiedClass.NO_HELMET, score=0.6, box=(5, 5, 10, 10)),
    ]
    forward = HelmetEvaluator().evaluate(predictions, ground_truth)
    backward = HelmetEvaluator().evaluate(reversed(predictions), reversed(ground_truth))
    assert forward.model_dump_json() == backward.model_dump_json()


# --- summary -----------------------------------------------------------------------
def test_summary_mirrors_the_report_headline() -> None:
    report = _report()
    summary = EvaluationSummary.from_report(report)
    assert summary.mean_ap == report.metrics.mean_ap
    assert summary.precision == report.metrics.precision
    assert summary.num_images == report.dataset.num_images
    assert summary.checkpoint_id is None  # pure-prediction run: no checkpoint


# --- CSV ---------------------------------------------------------------------------
def test_csv_has_overall_plus_one_row_per_class() -> None:
    lines = metrics_csv(_report()).strip().split("\n")
    assert len(lines) == 4  # header + overall + helmet + no_helmet
    assert lines[0].startswith("scope,ap,ap50,ap75,")
    assert lines[1].startswith("overall,")
    assert lines[2].startswith("helmet,")
    assert lines[3].startswith("no_helmet,")


def test_csv_renders_undefined_metrics_as_empty_cells() -> None:
    report = HelmetEvaluator().evaluate([pred("a.png", score=0.9)], [])  # no ground truth
    overall = metrics_csv(report).strip().split("\n")[1].split(",")
    assert overall[1] == ""  # mean_ap is undefined, never fabricated as 0


# --- time honesty -------------------------------------------------------------------
def test_generated_at_stays_none_without_a_clock() -> None:
    assert _report().generated_at is None


def test_generated_at_comes_only_from_an_injected_clock() -> None:
    instant = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
    report = _report(clock=lambda: instant)
    assert report.generated_at == instant

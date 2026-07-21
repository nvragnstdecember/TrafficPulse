"""Checkpoint evaluation against real (tiny) trained RT-DETR runs (H5).

Mirrors the H4B test discipline: one real tiny training run (module-scoped),
then genuine checkpoint loading + inference + metric computation on CPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _rtdetr_helpers import (
    HAVE_TORCH,
    TORCH_SKIP_REASON,
    make_rt_config,
    make_split_fixture,
)
from helmet_rtdetr.errors import (
    CheckpointNotFoundError,
    EvaluationDataError,
    InvalidEvaluationConfigError,
    PayloadNotFoundError,
)
from helmet_rtdetr.evaluation import (
    EvaluationConfig,
    HelmetEvaluator,
    evaluate_checkpoint,
    evaluate_checkpoints,
)
from helmet_rtdetr.split import SplitName
from helmet_rtdetr.training import MemoryLogSink

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_SKIP_REASON)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    """One real tiny training run (2 epochs, 2 retained checkpoints)."""

    from helmet_rtdetr.rtdetr import RTDETRTrainer

    tmp = tmp_path_factory.mktemp("h5-run")
    make_split_fixture(tmp)
    trainer = RTDETRTrainer(make_rt_config(tmp, epochs=2), sink=MemoryLogSink())
    trainer.fit()
    return tmp, trainer


def _eval_args(tmp: Path) -> dict:
    from helmet_rtdetr.rtdetr import DataConfig, RTDETRModelConfig

    return {
        "model": RTDETRModelConfig(checkpoint=None),  # the tiny architecture trained above
        "data": DataConfig(
            splits_dir=str(tmp / "splits"),
            image_root=str(tmp / "images"),
            image_height=64,
            image_width=64,
        ),
    }


def _run_dir(tmp: Path) -> Path:
    return tmp / "runs" / "run-a"


# --- selectors -------------------------------------------------------------------
def test_evaluate_latest_checkpoint(trained) -> None:
    tmp, trainer = trained
    report = evaluate_checkpoint(_run_dir(tmp), **_eval_args(tmp))
    assert report.checkpoint is not None
    assert (
        report.checkpoint.checkpoint_id
        == trainer.trainer.checkpoints.latest().record.checkpoint_id
    )
    assert report.checkpoint.run_dir == str(_run_dir(tmp))
    assert report.dataset.split == "test"
    assert report.dataset.num_images >= 1
    assert report.metrics.num_predictions > 0  # 10 queries per image at threshold 0


def test_evaluate_best_checkpoint(trained) -> None:
    tmp, trainer = trained
    report = evaluate_checkpoint(_run_dir(tmp), checkpoint="best", **_eval_args(tmp))
    best = trainer.trainer.checkpoints.best().record
    assert report.checkpoint is not None
    assert report.checkpoint.checkpoint_id == best.checkpoint_id
    assert "best" in report.checkpoint.roles


def test_evaluate_explicit_checkpoint_id(trained) -> None:
    tmp, trainer = trained
    checkpoint_id = trainer.trainer.checkpoints.checkpoint_ids()[0]
    report = evaluate_checkpoint(_run_dir(tmp), checkpoint=checkpoint_id, **_eval_args(tmp))
    assert report.checkpoint is not None
    assert report.checkpoint.checkpoint_id == checkpoint_id


def test_unknown_checkpoint_id_raises(trained) -> None:
    tmp, _ = trained
    with pytest.raises(CheckpointNotFoundError):
        evaluate_checkpoint(_run_dir(tmp), checkpoint="e9999-s99999999", **_eval_args(tmp))


# --- determinism ------------------------------------------------------------------
def test_repeat_evaluation_is_byte_identical(trained) -> None:
    tmp, _ = trained
    first = evaluate_checkpoint(_run_dir(tmp), **_eval_args(tmp))
    second = evaluate_checkpoint(_run_dir(tmp), **_eval_args(tmp))
    assert first.model_dump_json() == second.model_dump_json()


def test_operating_point_counts_are_internally_consistent(trained) -> None:
    tmp, _ = trained
    report = evaluate_checkpoint(_run_dir(tmp), **_eval_args(tmp))
    for class_metrics in report.metrics.per_class:
        # Every ground-truth box is either matched (TP) or missed (FN).
        assert (
            class_metrics.true_positives + class_metrics.false_negatives
            == class_metrics.num_ground_truth
        )
    assert report.metrics.num_predictions == sum(
        c.num_predictions for c in report.metrics.per_class
    )


# --- directory sweep ----------------------------------------------------------------
def test_evaluate_checkpoints_covers_all_retained(tmp_path: Path, trained) -> None:
    tmp, trainer = trained
    retained = trainer.trainer.checkpoints.checkpoint_ids()
    reports = evaluate_checkpoints(_run_dir(tmp), output_root=tmp_path, **_eval_args(tmp))
    assert [r.checkpoint.checkpoint_id for r in reports if r.checkpoint] == list(retained)
    for checkpoint_id in retained:
        assert (tmp_path / checkpoint_id / "evaluation.json").is_file()
        assert (tmp_path / checkpoint_id / "summary.json").is_file()
        assert (tmp_path / checkpoint_id / "metrics.csv").is_file()


def test_evaluate_checkpoints_of_an_unstarted_run_is_empty(tmp_path: Path, trained) -> None:
    tmp, _ = trained
    reports = evaluate_checkpoints(tmp_path / "no-such-run", **_eval_args(tmp))
    assert reports == ()


# --- failure modes -------------------------------------------------------------------
def test_missing_payload_fails_loudly(tmp_path: Path, trained) -> None:
    tmp, _ = trained
    run_copy = tmp_path / "run-copy"
    shutil.copytree(_run_dir(tmp), run_copy)
    for payload in (run_copy / "checkpoints").glob("ckpt-*.pt"):
        payload.unlink()
    with pytest.raises(PayloadNotFoundError, match="no weight payload"):
        evaluate_checkpoint(run_copy, **_eval_args(tmp))


def test_payload_without_model_weights_fails_loudly(tmp_path: Path, trained) -> None:
    from helmet_rtdetr.rtdetr import PayloadStore
    from helmet_rtdetr.training import CheckpointManager, CheckpointPolicy

    tmp, _ = trained
    run_copy = tmp_path / "run-copy"
    shutil.copytree(_run_dir(tmp), run_copy)
    manager = CheckpointManager(run_copy / "checkpoints", CheckpointPolicy())
    latest_id = manager.latest().record.checkpoint_id
    PayloadStore(run_copy / "checkpoints").save(latest_id, {"optimizer": {}})
    with pytest.raises(EvaluationDataError, match="no 'model' weights"):
        evaluate_checkpoint(run_copy, **_eval_args(tmp))


def test_empty_split_is_refused(tmp_path: Path, trained) -> None:
    from helmet_rtdetr.rtdetr import DataConfig

    tmp, _ = trained
    empty_splits = tmp_path / "splits"
    empty_splits.mkdir()
    (empty_splits / "test.jsonl").write_text("", encoding="utf-8")
    args = _eval_args(tmp)
    args["data"] = DataConfig(
        splits_dir=str(empty_splits),
        image_root=str(tmp / "images"),
        image_height=64,
        image_width=64,
    )
    with pytest.raises(EvaluationDataError, match="no objects"):
        evaluate_checkpoint(_run_dir(tmp), **args)


def test_missing_split_manifest_is_refused(tmp_path: Path, trained) -> None:
    from helmet_rtdetr.rtdetr import DataConfig

    tmp, _ = trained
    args = _eval_args(tmp)
    args["data"] = DataConfig(
        splits_dir=str(tmp_path / "never-exported"),
        image_root=str(tmp / "images"),
        image_height=64,
        image_width=64,
    )
    with pytest.raises(EvaluationDataError, match="not found"):
        evaluate_checkpoint(_run_dir(tmp), **args)


def test_cuda_request_without_cuda_fails_loudly(trained) -> None:
    import torch

    if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
        pytest.skip("CUDA present; the unavailable-CUDA branch cannot be exercised")
    tmp, _ = trained
    with pytest.raises(InvalidEvaluationConfigError, match="CUDA is not available"):
        evaluate_checkpoint(
            _run_dir(tmp), config=EvaluationConfig(device="cuda"), **_eval_args(tmp)
        )


# --- other splits + class-based API ---------------------------------------------------
def test_val_split_can_be_evaluated(trained) -> None:
    tmp, _ = trained
    report = evaluate_checkpoint(_run_dir(tmp), split=SplitName.VAL, **_eval_args(tmp))
    assert report.dataset.split == "val"


def test_evaluator_class_method_and_save_report(tmp_path: Path, trained) -> None:
    tmp, _ = trained
    evaluator = HelmetEvaluator(EvaluationConfig(device="cpu", batch_size=1))
    report = evaluator.evaluate_checkpoint(_run_dir(tmp), **_eval_args(tmp))
    written = evaluator.save_report(report, tmp_path / "artifacts")
    assert set(written) == {"evaluation", "summary", "metrics_csv"}
    assert all(path.is_file() for path in written.values())


def test_output_dir_of_the_helper_writes_artifacts(tmp_path: Path, trained) -> None:
    tmp, _ = trained
    evaluate_checkpoint(_run_dir(tmp), output_dir=tmp_path / "out", **_eval_args(tmp))
    assert (tmp_path / "out" / "evaluation.json").is_file()

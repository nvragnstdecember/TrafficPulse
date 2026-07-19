"""The full RT-DETR training loop on the H4A trainer (H4B). Real tiny training."""

from __future__ import annotations

from pathlib import Path

import pytest
from _rtdetr_helpers import (
    HAVE_TORCH,
    TORCH_SKIP_REASON,
    make_rt_config,
    make_split_fixture,
)
from helmet_rtdetr.errors import PayloadNotFoundError
from helmet_rtdetr.training import Callback, MemoryLogSink, RunPhase, TrainingState

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_SKIP_REASON)


class _InterruptAfterFirstEpoch(Callback):
    """Simulates an interruption: raises when the SECOND epoch begins.

    Raising at epoch start (not epoch end) guarantees epoch 1's checkpoint and
    weight payload were fully written before the "crash".
    """

    def on_epoch_start(self, state: TrainingState) -> None:
        if state.epoch >= 1:
            raise RuntimeError("simulated interruption")


def _fit(tmp_path: Path, **kwargs):
    from helmet_rtdetr.rtdetr import RTDETRTrainer

    trainer = RTDETRTrainer(make_rt_config(tmp_path, **kwargs), sink=MemoryLogSink())
    return trainer, trainer.fit()


# --- the happy path -----------------------------------------------------------
def test_fit_trains_validates_and_checkpoints(tmp_path: Path) -> None:
    make_split_fixture(tmp_path)
    trainer, final = _fit(tmp_path, epochs=2)

    assert final.phase is RunPhase.FINISHED
    assert final.epoch == 2
    assert final.global_step == 4  # 4 train images / batch 2 = 2 steps x 2 epochs
    assert trainer.metrics.names() == ("train/loss", "train/lr", "val/loss")
    assert len(trainer.metrics.history("val/loss")) == 2
    assert final.best_metric_name == "val/loss"


def test_payloads_exactly_mirror_retained_metadata(tmp_path: Path) -> None:
    make_split_fixture(tmp_path)
    trainer, _ = _fit(tmp_path, epochs=3, keep_last=1)

    checkpoints = trainer.trainer.layout.checkpoints
    metadata_ids = {p.stem.removeprefix("ckpt-") for p in checkpoints.glob("ckpt-*.json")}
    payload_ids = {p.stem.removeprefix("ckpt-") for p in checkpoints.glob("ckpt-*.pt")}
    assert payload_ids == metadata_ids  # retention mirrors the H4A manager exactly
    assert metadata_ids == set(trainer.trainer.checkpoints.checkpoint_ids())


def test_amp_falls_back_cleanly_on_cpu(tmp_path: Path) -> None:
    import torch

    make_split_fixture(tmp_path)
    trainer, final = _fit(tmp_path, epochs=1)  # config requests amp=True

    if not torch.cuda.is_available():
        assert trainer.amp_active is False  # requested, gracefully disabled
    assert final.epoch == 1  # and training still ran


# --- resume --------------------------------------------------------------------
def test_interrupted_run_resumes_with_restored_weights(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import PayloadStore, RTDETRTrainer, checkpoint_id_for

    make_split_fixture(tmp_path)
    first = RTDETRTrainer(
        make_rt_config(tmp_path, epochs=2),
        callbacks=(_InterruptAfterFirstEpoch(),),
        sink=MemoryLogSink(),
    )
    with pytest.raises(RuntimeError, match="interruption"):
        first.fit()
    assert first.state.epoch == 1  # epoch 1 completed + checkpointed before the crash

    # What the payload says the model weighed at the checkpoint:
    store = PayloadStore(first.trainer.layout.checkpoints)
    saved = store.load(checkpoint_id_for(first.state))

    peeked: list[float] = []

    class _PeekWeights(Callback):
        def on_epoch_start(self, state: TrainingState) -> None:
            if not peeked:
                model = resumed._model  # the restored module, before any training step
                total = sum(p.double().sum() for p in model.module.parameters())
                peeked.append(float(total))

    resumed = RTDETRTrainer(
        make_rt_config(tmp_path, epochs=2, resume=True),
        callbacks=(_PeekWeights(),),
        sink=MemoryLogSink(),
    )
    final = resumed.fit()

    assert resumed.trainer.resumed is True
    assert final.epoch == 2  # only the remaining epoch ran
    # Buffers (e.g. batch-norm running stats) are part of state_dict but not
    # parameters(); compare the parameter subset of the saved payload instead.
    param_names = {name for name, _ in resumed._model.module.named_parameters()}
    saved_param_sum = float(
        sum(v.double().sum() for k, v in saved["model"].items() if k in param_names)
    )
    assert peeked and peeked[0] == pytest.approx(saved_param_sum, rel=0, abs=1e-6)


def test_resume_restores_optimizer_and_continues_metrics(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import RTDETRTrainer

    make_split_fixture(tmp_path)
    first = RTDETRTrainer(
        make_rt_config(tmp_path, epochs=2),
        callbacks=(_InterruptAfterFirstEpoch(),),
        sink=MemoryLogSink(),
    )
    with pytest.raises(RuntimeError):
        first.fit()

    resumed = RTDETRTrainer(make_rt_config(tmp_path, epochs=2, resume=True))
    resumed.fit()

    # Metrics history spans BOTH epochs: epoch 1 from disk, epoch 2 freshly run.
    assert [p.epoch for p in resumed.metrics.history("val/loss")] == [0, 1]


def test_resume_with_missing_payload_fails_loudly(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import RTDETRTrainer, checkpoint_id_for

    make_split_fixture(tmp_path)
    first = RTDETRTrainer(
        make_rt_config(tmp_path, epochs=2),
        callbacks=(_InterruptAfterFirstEpoch(),),
        sink=MemoryLogSink(),
    )
    with pytest.raises(RuntimeError):
        first.fit()

    payload = first.trainer.layout.checkpoints / (
        f"ckpt-{checkpoint_id_for(first.state)}.pt"
    )
    payload.unlink()  # metadata remains; weights gone

    resumed = RTDETRTrainer(make_rt_config(tmp_path, epochs=2, resume=True))
    with pytest.raises(PayloadNotFoundError, match="no weight payload"):
        resumed.fit()  # never silently retrains from scratch


# --- configuration branches -----------------------------------------------------
def test_empty_train_split_fails_loudly(tmp_path: Path) -> None:
    from helmet_rtdetr.errors import InvalidTrainingConfigError
    from helmet_rtdetr.rtdetr import RTDETRTrainer

    make_split_fixture(tmp_path)
    (tmp_path / "splits" / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(InvalidTrainingConfigError, match="no images"):
        RTDETRTrainer(make_rt_config(tmp_path, epochs=1)).fit()


def test_fit_without_a_val_split_trains_train_only(tmp_path: Path) -> None:
    """No val.jsonl: training proceeds; only train metrics are recorded.

    Requires save_best=False — with no validation there is no val/loss to track,
    and the checkpoint policy would (correctly) refuse to pick a best on absent
    evidence.
    """

    make_split_fixture(tmp_path)
    (tmp_path / "splits" / "val.jsonl").unlink()
    trainer, final = _fit(tmp_path, epochs=1, save_best=False)

    assert final.epoch == 1
    assert trainer.metrics.names() == ("train/loss", "train/lr")


def test_explicit_cpu_device_is_honoured(tmp_path: Path) -> None:
    make_split_fixture(tmp_path)
    trainer, final = _fit(tmp_path, epochs=1, device="cpu")
    assert final.epoch == 1
    assert trainer.amp_active is False  # amp requested, but never active on cpu


def test_requesting_cuda_without_cuda_fails_loudly(tmp_path: Path) -> None:
    import torch

    if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
        pytest.skip("CUDA present; the unavailable-CUDA branch cannot be exercised")
    from helmet_rtdetr.errors import InvalidTrainingConfigError
    from helmet_rtdetr.rtdetr import RTDETRTrainer

    make_split_fixture(tmp_path)
    with pytest.raises(InvalidTrainingConfigError, match="CUDA is not available"):
        RTDETRTrainer(make_rt_config(tmp_path, epochs=1, device="cuda")).fit()


def test_epoch_granularity_scheduler_steps_per_epoch(tmp_path: Path) -> None:
    """A StepLR config drives the per-epoch stepping branch of the loop."""

    from helmet_rtdetr.training import StepSchedulerConfig

    make_split_fixture(tmp_path)
    trainer, final = _fit(
        tmp_path, epochs=2, scheduler=StepSchedulerConfig(step_size=1, gamma=0.5)
    )

    assert final.epoch == 2
    lrs = [p.value for p in trainer.metrics.history("train/lr")]
    assert lrs[0] == pytest.approx(1e-4 * 0.5)  # stepped once after epoch 1
    assert lrs[1] == pytest.approx(1e-4 * 0.25)  # and again after epoch 2


def test_incomplete_payload_on_resume_fails_loudly(tmp_path: Path) -> None:
    from helmet_rtdetr.errors import InvalidTrainingConfigError
    from helmet_rtdetr.rtdetr import PayloadStore, RTDETRTrainer, checkpoint_id_for

    make_split_fixture(tmp_path)
    first = RTDETRTrainer(
        make_rt_config(tmp_path, epochs=2),
        callbacks=(_InterruptAfterFirstEpoch(),),
        sink=MemoryLogSink(),
    )
    with pytest.raises(RuntimeError):
        first.fit()

    store = PayloadStore(first.trainer.layout.checkpoints)
    checkpoint_id = checkpoint_id_for(first.state)
    payload = store.load(checkpoint_id)
    del payload["scaler"]  # simulate a truncated / older-format payload
    store.save(checkpoint_id, payload)

    resumed = RTDETRTrainer(make_rt_config(tmp_path, epochs=2, resume=True))
    with pytest.raises(InvalidTrainingConfigError, match="missing components"):
        resumed.fit()


def test_write_payload_outside_fit_is_a_no_op(tmp_path: Path) -> None:
    """The payload callback guard: a checkpoint before any model exists persists nothing."""

    from helmet_rtdetr.rtdetr import RTDETRTrainer
    from helmet_rtdetr.training import CheckpointRecord, CheckpointRole

    make_split_fixture(tmp_path)
    trainer = RTDETRTrainer(make_rt_config(tmp_path, epochs=1))
    record = CheckpointRecord(
        checkpoint_id="e0000-s00000000",
        epoch=0,
        global_step=0,
        roles=(CheckpointRole.LATEST,),
        filename="ckpt-e0000-s00000000.json",
    )
    trainer._write_payload(record)  # model is None: must not raise, must not write
    assert not trainer.trainer.layout.checkpoints.exists() or not list(
        trainer.trainer.layout.checkpoints.glob("*.pt")
    )


def test_payload_prune_on_missing_directory_is_empty(tmp_path: Path) -> None:
    from helmet_rtdetr.rtdetr import PayloadStore

    assert PayloadStore(tmp_path / "never-created").prune(["x"]) == ()


# --- determinism ----------------------------------------------------------------
def test_identical_seeds_train_identically(tmp_path: Path) -> None:
    make_split_fixture(tmp_path / "a")
    make_split_fixture(tmp_path / "b")
    trainer_a, _ = _fit(tmp_path / "a", epochs=1, seed=7)
    trainer_b, _ = _fit(tmp_path / "b", epochs=1, seed=7)

    loss_a = trainer_a.metrics.latest("train/loss").value
    loss_b = trainer_b.metrics.latest("train/loss").value
    assert loss_a == loss_b  # same seed, same data -> bit-identical CPU training


def test_different_seeds_train_differently(tmp_path: Path) -> None:
    make_split_fixture(tmp_path / "a")
    make_split_fixture(tmp_path / "b")
    trainer_a, _ = _fit(tmp_path / "a", epochs=1, seed=7)
    trainer_b, _ = _fit(tmp_path / "b", epochs=1, seed=8)

    assert (
        trainer_a.metrics.latest("train/loss").value
        != trainer_b.metrics.latest("train/loss").value
    )

"""Checkpoint manager: save/load, latest/best, periodic, cleanup (H4A)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helmet_rtdetr.errors import CheckpointError, CheckpointNotFoundError
from helmet_rtdetr.training import (
    CheckpointManager,
    CheckpointPolicy,
    CheckpointRole,
    RunPhase,
    TrainingState,
)

POLICY_BEST = CheckpointPolicy(save_best=True, best_metric="val/ap", keep_last=2)


def state(epoch: int, step: int | None = None) -> TrainingState:
    return TrainingState(
        phase=RunPhase.RUNNING, epoch=epoch, global_step=step if step is not None else epoch * 10
    )


def manager(tmp_path: Path, policy: CheckpointPolicy = POLICY_BEST) -> CheckpointManager:
    return CheckpointManager(tmp_path / "checkpoints", policy)


# --- save + roles ------------------------------------------------------------
def test_first_save_is_latest_and_best(tmp_path: Path) -> None:
    record = manager(tmp_path).save(state(1), metrics={"val/ap": 0.5})
    assert CheckpointRole.LATEST in record.roles
    assert CheckpointRole.BEST in record.roles
    assert record.metric_value == 0.5


def test_improvement_updates_best(tmp_path: Path) -> None:
    m = manager(tmp_path)
    m.save(state(1), metrics={"val/ap": 0.5})
    improved = m.save(state(2), metrics={"val/ap": 0.7})
    assert CheckpointRole.BEST in improved.roles
    assert m.best().record.epoch == 2


def test_regression_keeps_the_old_best(tmp_path: Path) -> None:
    m = manager(tmp_path)
    m.save(state(1), metrics={"val/ap": 0.7})
    worse = m.save(state(2), metrics={"val/ap": 0.5})
    assert CheckpointRole.BEST not in worse.roles
    assert m.best().record.epoch == 1
    assert m.latest().record.epoch == 2


def test_min_mode_prefers_lower_values(tmp_path: Path) -> None:
    policy = CheckpointPolicy(save_best=True, best_metric="val/loss", best_mode="min")
    m = manager(tmp_path, policy)
    m.save(state(1), metrics={"val/loss": 1.0})
    better = m.save(state(2), metrics={"val/loss": 0.5})
    assert CheckpointRole.BEST in better.roles


def test_periodic_role_every_n_epochs(tmp_path: Path) -> None:
    policy = CheckpointPolicy(keep_last=1, every_n_epochs=2)
    m = manager(tmp_path, policy)
    roles = {e: m.save(state(e), metrics=None).roles for e in (1, 2, 3, 4)}
    assert CheckpointRole.PERIODIC not in roles[1]
    assert CheckpointRole.PERIODIC in roles[2]
    assert CheckpointRole.PERIODIC not in roles[3]
    assert CheckpointRole.PERIODIC in roles[4]


def test_save_without_best_tracking_needs_no_metrics(tmp_path: Path) -> None:
    m = manager(tmp_path, CheckpointPolicy(keep_last=3))
    record = m.save(state(1), metrics=None)
    assert record.roles == (CheckpointRole.LATEST,)
    with pytest.raises(CheckpointNotFoundError):
        m.best()


# --- guarded evidence --------------------------------------------------------
def test_missing_best_metric_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="absent"):
        manager(tmp_path).save(state(1), metrics={"train/loss": 1.0})


def test_non_finite_best_metric_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="finite"):
        manager(tmp_path).save(state(1), metrics={"val/ap": float("nan")})


# --- load / latest / best ----------------------------------------------------
def test_load_round_trips_state(tmp_path: Path) -> None:
    m = manager(tmp_path)
    record = m.save(state(1, step=42), metrics={"val/ap": 0.5})
    loaded = m.load(record.checkpoint_id)
    assert loaded.state.global_step == 42
    assert loaded.record == record


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError):
        manager(tmp_path).load("e9999-s00000000")


def test_latest_on_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError):
        manager(tmp_path).latest()


def test_corrupt_checkpoint_file_raises(tmp_path: Path) -> None:
    m = manager(tmp_path)
    record = m.save(state(1), metrics={"val/ap": 0.5})
    (tmp_path / "checkpoints" / record.filename).write_text("{ nope", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        m.load(record.checkpoint_id)


def test_corrupt_index_raises(tmp_path: Path) -> None:
    m = manager(tmp_path)
    m.save(state(1), metrics={"val/ap": 0.5})
    (tmp_path / "checkpoints" / "index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CheckpointError, match="index"):
        m.latest()


# --- cleanup -----------------------------------------------------------------
def test_cleanup_keeps_best_periodic_and_last_n(tmp_path: Path) -> None:
    policy = CheckpointPolicy(
        save_best=True, best_metric="val/ap", keep_last=1, every_n_epochs=3
    )
    m = manager(tmp_path, policy)
    # Best at epoch 1 (0.9), then declining; periodic at epoch 3.
    values = {1: 0.9, 2: 0.5, 3: 0.4, 4: 0.3}
    for epoch, value in values.items():
        m.save(state(epoch), metrics={"val/ap": value})

    retained = m.checkpoint_ids()
    assert "e0001-s00000010" in retained  # best, never cleaned up
    assert "e0003-s00000030" in retained  # periodic archive
    assert "e0004-s00000040" in retained  # last keep_last=1
    assert "e0002-s00000020" not in retained  # cleaned up
    assert not (tmp_path / "checkpoints" / "ckpt-e0002-s00000020.json").exists()


def test_retained_files_stay_loadable_after_cleanup(tmp_path: Path) -> None:
    m = manager(tmp_path, CheckpointPolicy(save_best=True, best_metric="val/ap", keep_last=1))
    m.save(state(1), metrics={"val/ap": 0.9})
    m.save(state(2), metrics={"val/ap": 0.1})
    m.save(state(3), metrics={"val/ap": 0.2})

    assert m.best().record.epoch == 1
    assert m.latest().record.epoch == 3


def test_duplicate_save_does_not_duplicate_history(tmp_path: Path) -> None:
    m = manager(tmp_path, CheckpointPolicy(keep_last=5))
    m.save(state(1), metrics=None)
    m.save(state(1), metrics=None)  # replayed save of the same epoch/step
    assert m.checkpoint_ids().count("e0001-s00000010") == 1


# --- determinism -------------------------------------------------------------
def test_identical_sequences_write_byte_identical_files(tmp_path: Path) -> None:
    for sub in ("a", "b"):
        m = manager(tmp_path / sub)
        m.save(state(1), metrics={"val/ap": 0.5})
        m.save(state(2), metrics={"val/ap": 0.7})

    for name in ("ckpt-e0001-s00000010.json", "ckpt-e0002-s00000020.json", "index.json"):
        a = (tmp_path / "a" / "checkpoints" / name).read_bytes()
        b = (tmp_path / "b" / "checkpoints" / name).read_bytes()
        assert a == b

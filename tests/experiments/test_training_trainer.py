"""Trainer lifecycle, callbacks, resume, seeding, layout (H4A)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from _training_helpers import Recorder, make_clock, make_config
from helmet_rtdetr.errors import (
    DuplicateExperimentError,
    ResumeError,
    TrainerStateError,
)
from helmet_rtdetr.training import (
    DEFAULT_RUNS_ROOT,
    LogEventKind,
    MemoryLogSink,
    RunLayout,
    RunPhase,
    Trainer,
    TrainingState,
    apply_seed_plan,
    derive_seed_plan,
)


def run_one_epoch(trainer: Trainer, value: float = 0.5, batches: int = 2) -> None:
    trainer.begin_epoch()
    for _ in range(batches):
        trainer.record_batch()
    trainer.end_epoch({"val/ap": value})


# --- happy path --------------------------------------------------------------
def test_full_lifecycle_event_sequence(tmp_path: Path) -> None:
    sink = MemoryLogSink()
    trainer = Trainer(make_config(tmp_path), sink=sink)
    trainer.begin()
    run_one_epoch(trainer, 0.5)
    run_one_epoch(trainer, 0.7)
    trainer.end()

    kinds = [e.kind for e in sink.events]
    assert kinds == [
        LogEventKind.EXPERIMENT_START,
        LogEventKind.EPOCH_START,
        LogEventKind.EPOCH_END,
        LogEventKind.CHECKPOINT,
        LogEventKind.EPOCH_START,
        LogEventKind.EPOCH_END,
        LogEventKind.CHECKPOINT,
        LogEventKind.EXPERIMENT_FINISH,
    ]
    assert [e.sequence for e in sink.events] == list(range(len(sink.events)))


def test_state_advances_and_tracks_best(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    run_one_epoch(trainer, 0.5, batches=3)
    run_one_epoch(trainer, 0.7, batches=2)
    state = trainer.end()

    assert state.phase is RunPhase.FINISHED
    assert state.epoch == 2
    assert state.global_step == 5
    assert state.best_metric_value == 0.7
    assert len(state.checkpoint_history) == 2


def test_run_directory_artifacts_are_written(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    run_one_epoch(trainer)
    trainer.end()

    layout = trainer.layout
    assert layout.config_path.is_file()
    assert layout.seed_plan_path.is_file()
    assert layout.metrics_path.is_file()
    final = TrainingState.model_validate_json(layout.state_path.read_text(encoding="utf-8"))
    assert final.phase is RunPhase.FINISHED


def test_callbacks_fire_in_registration_order(tmp_path: Path) -> None:
    log: list[str] = []
    trainer = Trainer(
        make_config(tmp_path),
        callbacks=(Recorder(log, "a"), Recorder(log, "b")),
    )
    trainer.begin()
    trainer.begin_epoch()
    trainer.record_batch()
    trainer.end_epoch({"val/ap": 0.5})
    trainer.end()

    assert log == [
        "a:train_start", "b:train_start",
        "a:epoch_start", "b:epoch_start",
        "a:batch_start", "b:batch_start",
        "a:batch_end", "b:batch_end",
        "a:epoch_end", "b:epoch_end",
        "a:checkpoint", "b:checkpoint",
        "a:train_end", "b:train_end",
    ]


def test_metrics_are_recorded_per_epoch(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    run_one_epoch(trainer, 0.5)
    run_one_epoch(trainer, 0.7)

    assert [p.value for p in trainer.metrics.history("val/ap")] == [0.5, 0.7]
    assert trainer.metrics.epoch_summary(1) == {"val/ap": 0.7}


# --- lifecycle guards --------------------------------------------------------
def test_begin_twice_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    with pytest.raises(TrainerStateError, match="first lifecycle call"):
        trainer.begin()


def test_epoch_calls_before_begin_raise(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    with pytest.raises(TrainerStateError, match="begin"):
        trainer.begin_epoch()
    with pytest.raises(TrainerStateError):
        trainer.end_epoch({"val/ap": 0.5})
    with pytest.raises(TrainerStateError):
        trainer.end()


def test_end_epoch_without_begin_epoch_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    with pytest.raises(TrainerStateError, match="inside an epoch"):
        trainer.end_epoch({"val/ap": 0.5})


def test_end_mid_epoch_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    trainer.begin_epoch()
    with pytest.raises(TrainerStateError, match="outside an epoch"):
        trainer.end()


def test_record_batch_outside_epoch_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    with pytest.raises(TrainerStateError):
        trainer.record_batch()


def test_epoch_budget_is_enforced(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path, epochs=1))
    trainer.begin()
    run_one_epoch(trainer)
    with pytest.raises(TrainerStateError, match="configured epochs"):
        trainer.begin_epoch()


# --- duplicate experiments ---------------------------------------------------
def test_duplicate_experiment_name_raises(tmp_path: Path) -> None:
    Trainer(make_config(tmp_path)).begin()
    with pytest.raises(DuplicateExperimentError, match="already exists"):
        Trainer(make_config(tmp_path)).begin()


# --- resume ------------------------------------------------------------------
def test_resume_restores_state_and_metrics(tmp_path: Path) -> None:
    first = Trainer(make_config(tmp_path))
    first.begin()
    run_one_epoch(first, 0.5, batches=2)  # interrupted: no end()

    sink = MemoryLogSink()
    resumed = Trainer(make_config(tmp_path, resume=True), sink=sink)
    resumed.begin()

    assert resumed.resumed is True
    assert resumed.state.epoch == 1
    assert resumed.state.global_step == 2
    assert resumed.metrics.names() == ("val/ap",)  # history reloaded from disk
    assert [e.kind for e in sink.events] == [LogEventKind.RESUME]

    run_one_epoch(resumed, 0.7)
    assert resumed.end().epoch == 2


def test_resume_from_best_checkpoint(tmp_path: Path) -> None:
    first = Trainer(make_config(tmp_path))
    first.begin()
    run_one_epoch(first, 0.9)  # best
    run_one_epoch(first, 0.4)  # latest, worse

    config = make_config(tmp_path, resume=True).model_copy(
        update={
            "resume": make_config(tmp_path, resume=True).resume.model_copy(
                update={"from_checkpoint": "best"}
            )
        }
    )
    resumed = Trainer(config)
    resumed.begin()
    assert resumed.state.epoch == 1  # the best checkpoint, not the latest


def test_resume_with_mismatched_config_raises(tmp_path: Path) -> None:
    Trainer(make_config(tmp_path, epochs=3)).begin()
    with pytest.raises(ResumeError, match="does not match"):
        Trainer(make_config(tmp_path, epochs=5, resume=True)).begin()


def test_resume_of_finished_experiment_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path, epochs=1))
    trainer.begin()
    run_one_epoch(trainer)
    trainer.end()

    with pytest.raises(ResumeError, match="finished"):
        Trainer(make_config(tmp_path, epochs=1, resume=True)).begin()


def test_resume_with_corrupt_stored_config_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    trainer.layout.config_path.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ResumeError, match="unreadable"):
        Trainer(make_config(tmp_path, resume=True)).begin()


def test_resume_with_corrupt_stored_state_raises(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    run_one_epoch(trainer)
    trainer.layout.state_path.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ResumeError, match="state.json"):
        Trainer(make_config(tmp_path, resume=True)).begin()


def test_trainer_exposes_its_checkpoint_manager(tmp_path: Path) -> None:
    trainer = Trainer(make_config(tmp_path))
    trainer.begin()
    run_one_epoch(trainer, 0.5)
    assert trainer.checkpoints.latest().record.epoch == 1


def test_base_callback_hooks_are_no_ops(tmp_path: Path) -> None:
    """A bare Callback must be safely attachable without overriding anything."""

    from helmet_rtdetr.training import Callback, CallbackList

    trainer = Trainer(make_config(tmp_path, epochs=1), callbacks=(Callback(),))
    trainer.begin()
    run_one_epoch(trainer)
    trainer.end()  # every hook fired against the no-op base without error
    assert len(CallbackList((Callback(), Callback()))) == 2


def test_checkpoint_record_requires_a_role() -> None:
    from helmet_rtdetr.training import CheckpointRecord
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CheckpointRecord(
            checkpoint_id="e0001-s00000001",
            epoch=1,
            global_step=1,
            roles=(),
            filename="ckpt-x.json",
        )


def test_manager_exposes_its_directory(tmp_path: Path) -> None:
    from helmet_rtdetr.training import CheckpointManager, CheckpointPolicy

    manager = CheckpointManager(tmp_path / "ck", CheckpointPolicy())
    assert manager.directory == tmp_path / "ck"


def test_resume_before_any_checkpoint_starts_fresh(tmp_path: Path) -> None:
    Trainer(make_config(tmp_path)).begin()  # initialised, never checkpointed

    resumed = Trainer(make_config(tmp_path, resume=True))
    resumed.begin()
    assert resumed.resumed is True
    assert resumed.state.epoch == 0


# --- determinism + time ------------------------------------------------------
def test_same_seed_reproduces_python_random(tmp_path: Path) -> None:
    Trainer(make_config(tmp_path / "a", seed=7)).begin()
    first = random.random()
    Trainer(make_config(tmp_path / "b", seed=7)).begin()
    second = random.random()
    Trainer(make_config(tmp_path / "c", seed=8)).begin()
    third = random.random()

    assert first == second
    assert first != third


def test_seed_plan_is_deterministic_with_distinct_components() -> None:
    plan = derive_seed_plan(7)
    assert plan == derive_seed_plan(7)
    assert len({plan.python_seed, plan.numpy_seed, plan.torch_seed}) == 3


def test_apply_seed_plan_reports_torch_as_deferred() -> None:
    applied = apply_seed_plan(derive_seed_plan(7))
    assert applied.applied == ("python", "numpy")
    assert "torch" in applied.deferred  # design-only: no ML import in H4A


def test_without_a_clock_time_is_never_fabricated(tmp_path: Path) -> None:
    sink = MemoryLogSink()
    trainer = Trainer(make_config(tmp_path), sink=sink)
    trainer.begin()
    run_one_epoch(trainer)
    state = trainer.end()

    assert state.elapsed_seconds is None
    assert all(e.at is None for e in sink.events)


def test_injected_clock_produces_elapsed_and_timestamps(tmp_path: Path) -> None:
    sink = MemoryLogSink()
    trainer = Trainer(make_config(tmp_path), sink=sink, clock=make_clock())
    trainer.begin()
    run_one_epoch(trainer)
    state = trainer.end()

    assert state.elapsed_seconds is not None and state.elapsed_seconds > 0.0
    assert all(e.at is not None for e in sink.events)


# --- layout ------------------------------------------------------------------
def test_layout_paths_are_deterministic(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "exp-a")
    assert layout.run_dir == tmp_path / "exp-a"
    assert layout.checkpoints == tmp_path / "exp-a" / "checkpoints"
    assert layout.metrics_path == tmp_path / "exp-a" / "metrics" / "metrics.json"


def test_layout_create_is_idempotent(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "exp-a")
    layout.create()
    layout.create()
    for sub in (layout.checkpoints, layout.logs, layout.metrics_dir, layout.artifacts):
        assert sub.is_dir()


def test_layout_initialization_flag(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "exp-a")
    assert layout.is_initialized() is False
    layout.create()
    layout.config_path.write_text("{}", encoding="utf-8")
    assert layout.is_initialized() is True


def test_default_runs_root_is_under_the_gitignored_runs_tree() -> None:
    assert DEFAULT_RUNS_ROOT.parent.name == "runs"
    assert DEFAULT_RUNS_ROOT.name == "helmet_rtdetr"

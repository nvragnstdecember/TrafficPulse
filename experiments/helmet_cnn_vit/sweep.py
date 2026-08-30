"""Tuning sweep and final seed runs for the CNN-vs-ViT comparison (P4-U5).

Implements the pre-registered budget from `PREREGISTRATION.md` section 7, and
implements it *symmetrically*: the same learning-rate grid, the same epoch counts,
and the same seeds are applied to both families by the same loop, so the "equal
tuning budget" that evaluation-protocol §8 demands is a property of the code rather
than of the operator's discipline.

Two phases:

``tune``
    3 configs per family (LR in {1e-4, 3e-4, 1e-3}) for 6 epochs. Selection is by
    **validation** macro-F1. Runs land under ``<runs>/tune``.

``final``
    The selected config per family, retrained for 12 epochs at seeds 0, 1, 2.
    Runs land under ``<runs>/final``.

Selection is written to ``<runs>/selection.json`` before any final run starts, so
the record of *what was chosen and why* cannot be back-filled after the test
numbers are known.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from helmet_rtdetr.models import _Model

from .models import MODEL_SPECS
from .train import TrainConfig, TrainResult, train_one

#: The pre-registered learning-rate grid (section 7). Equal for both families.
LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3)

#: Pre-registered epoch budgets.
TUNE_EPOCHS = 6
FINAL_EPOCHS = 12

#: Pre-registered seeds for the final runs.
FINAL_SEEDS: tuple[int, ...] = (0, 1, 2)

#: The seed used for every tuning run, so the grid is compared under one draw.
TUNE_SEED = 0


class Selection(_Model):
    """Which config won a family's tuning sweep, and on what evidence."""

    model: str
    chosen_lr: float
    val_macro_f1: float
    grid: dict[str, float | None]
    epochs: int


class SweepRecord(_Model):
    """The complete sweep outcome, written before the final runs begin."""

    lr_grid: tuple[float, ...]
    tune_epochs: int
    final_epochs: int
    final_seeds: tuple[int, ...]
    selections: dict[str, Selection]


def tune_family(
    model: str, *, crop_dir: Path, runs_root: Path, num_workers: int, epochs: int = TUNE_EPOCHS
) -> Selection:
    """Run the LR grid for one family and select on validation macro-F1."""

    grid: dict[str, float | None] = {}
    best_lr: float | None = None
    best_score: float | None = None

    for lr in LR_GRID:
        config = TrainConfig(
            model=model,
            seed=TUNE_SEED,
            epochs=epochs,
            lr=lr,
            num_workers=num_workers,
        )
        result = train_one(config, crop_dir=crop_dir, runs_root=runs_root / "tune")
        score = result.best_val_macro_f1
        grid[f"{lr:g}"] = score
        print(f"  [tune] {model} lr={lr:g} -> val macro-F1 {score} ({result.train_seconds:.0f}s)")
        # Strict >: a tie keeps the first (lower) learning rate, so selection is
        # total and reproducible rather than dependent on float noise.
        if score is not None and (best_score is None or score > best_score):
            best_score, best_lr = score, lr

    if best_lr is None or best_score is None:
        raise RuntimeError(f"no tuning run for {model} produced a macro-F1")
    return Selection(
        model=model, chosen_lr=best_lr, val_macro_f1=best_score, grid=grid, epochs=epochs
    )


def run_final(
    selection: Selection,
    *,
    crop_dir: Path,
    runs_root: Path,
    num_workers: int,
    seeds: Sequence[int] = FINAL_SEEDS,
    epochs: int = FINAL_EPOCHS,
) -> list[TrainResult]:
    """Retrain a family's selected config at each pre-registered seed."""

    results: list[TrainResult] = []
    for seed in seeds:
        config = TrainConfig(
            model=selection.model,
            seed=seed,
            epochs=epochs,
            lr=selection.chosen_lr,
            num_workers=num_workers,
        )
        result = train_one(config, crop_dir=crop_dir, runs_root=runs_root / "final")
        print(
            f"  [final] {selection.model} seed={seed} -> val macro-F1 "
            f"{result.best_val_macro_f1} ({result.train_seconds:.0f}s)"
        )
        results.append(result)
    return results


def run_sweep(
    *, crop_dir: Path, runs_root: Path, num_workers: int = 4
) -> tuple[SweepRecord, dict[str, list[TrainResult]]]:
    """Tune both families, record the selection, then run the final seeds."""

    runs_root.mkdir(parents=True, exist_ok=True)
    selections: dict[str, Selection] = {}
    for model in sorted(MODEL_SPECS):
        print(f"[tune] {model}")
        selections[model] = tune_family(
            model, crop_dir=crop_dir, runs_root=runs_root, num_workers=num_workers
        )

    record = SweepRecord(
        lr_grid=LR_GRID,
        tune_epochs=TUNE_EPOCHS,
        final_epochs=FINAL_EPOCHS,
        final_seeds=FINAL_SEEDS,
        selections=selections,
    )
    # Written BEFORE any final run, so selection cannot be revised in hindsight.
    (runs_root / "selection.json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"[selection] {json.dumps({k: v.chosen_lr for k, v in selections.items()})}")

    finals: dict[str, list[TrainResult]] = {}
    for model, selection in sorted(selections.items()):
        print(f"[final] {model} lr={selection.chosen_lr:g}")
        finals[model] = run_final(
            selection, crop_dir=crop_dir, runs_root=runs_root, num_workers=num_workers
        )
    return record, finals

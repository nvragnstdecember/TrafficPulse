"""Prediction/ground-truth accumulation + split-manifest loading (H5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _eval_helpers import pred, unified
from helmet_rtdetr.errors import EvaluationDataError, InvalidPredictionError
from helmet_rtdetr.evaluation import PredictionAccumulator, load_ground_truth
from helmet_rtdetr.unified import ObjectProvenance, UnifiedClass, UnifiedObject


# --- load_ground_truth -----------------------------------------------------------
def test_load_ground_truth_reads_a_split_manifest(tmp_path: Path) -> None:
    objects = [unified("a.png", UnifiedClass.HELMET), unified("b.png", UnifiedClass.NO_HELMET)]
    path = tmp_path / "test.jsonl"
    path.write_text(
        "\n".join(o.model_dump_json() for o in objects) + "\n\n", encoding="utf-8"
    )
    loaded = load_ground_truth(path)
    assert loaded == tuple(objects)  # manifest order, blank lines skipped


def test_load_ground_truth_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvaluationDataError, match="not found"):
        load_ground_truth(tmp_path / "absent.jsonl")


def test_load_ground_truth_malformed_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text('{"not": "a unified object"}\n', encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="line 1"):
        load_ground_truth(path)


# --- accumulation ------------------------------------------------------------------
def test_evaluable_ground_truth_registers_and_counts() -> None:
    accumulator = PredictionAccumulator()
    assert accumulator.add_ground_truth(unified("a.png", UnifiedClass.HELMET)) is True
    assert accumulator.num_images == 1
    assert len(accumulator.ground_truth()) == 1


def test_ignored_and_motorcycle_objects_register_their_image_only() -> None:
    accumulator = PredictionAccumulator()
    assert accumulator.add_ground_truth(unified("a.png", UnifiedClass.HELMET, ignore=True)) is False
    assert accumulator.add_ground_truth(unified("b.png", UnifiedClass.MOTORCYCLE)) is False
    assert accumulator.image_ids() == ("a.png", "b.png")  # genuine negatives
    assert accumulator.ground_truth() == ()


def test_duplicate_ground_truth_is_refused() -> None:
    accumulator = PredictionAccumulator()
    accumulator.add_ground_truth(unified("a.png", UnifiedClass.HELMET))
    with pytest.raises(EvaluationDataError, match="duplicate"):
        accumulator.add_ground_truth(unified("a.png", UnifiedClass.HELMET))


def test_conflicting_objects_with_one_content_id_are_refused() -> None:
    # Same (image, box, label) => same object_id, but different provenance.
    accumulator = PredictionAccumulator()
    original = unified("a.png", UnifiedClass.HELMET)
    conflicting = UnifiedObject(
        image_path=original.image_path,
        bbox=original.bbox,
        label=original.label,
        provenance=ObjectProvenance(
            dataset_id="other-set", dataset_version="9", adapter="coco", source_label="y"
        ),
    )
    accumulator.add_ground_truth(original)
    with pytest.raises(EvaluationDataError, match="conflicting"):
        accumulator.add_ground_truth(conflicting)


def test_prediction_on_unknown_image_is_refused() -> None:
    accumulator = PredictionAccumulator()
    with pytest.raises(InvalidPredictionError, match="unknown image"):
        accumulator.add_prediction(pred("nowhere.png"))


def test_prediction_on_registered_image_is_accepted() -> None:
    accumulator = PredictionAccumulator()
    accumulator.add_image("a.png")
    assert accumulator.add_predictions([pred("a.png"), pred("a.png", score=0.5)]) == 2
    assert len(accumulator.predictions()) == 2


def test_empty_image_id_is_refused() -> None:
    with pytest.raises(EvaluationDataError, match="non-empty"):
        PredictionAccumulator().add_image("")


def test_outputs_are_canonically_ordered() -> None:
    accumulator = PredictionAccumulator()
    accumulator.add_ground_truths(
        [unified("b.png", UnifiedClass.HELMET), unified("a.png", UnifiedClass.HELMET)]
    )
    accumulator.add_predictions([pred("b.png", score=0.5), pred("a.png", score=0.9)])
    assert accumulator.image_ids() == ("a.png", "b.png")
    assert [o.image_path for o in accumulator.ground_truth()] == ["a.png", "b.png"]
    assert [p.score for p in accumulator.predictions()] == [0.9, 0.5]

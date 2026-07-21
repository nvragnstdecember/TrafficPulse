"""Prediction/ground-truth accumulation for one evaluation (H5).

The accumulator is the strict front door of an evaluation: it fixes the image
universe, filters ground truth by the same rule the H4B data layer applies to
training targets (``ignore``-flagged and non-detector-class objects contribute
no box but still register their image — a genuine negative sample), refuses
duplicate ground truth (same content-derived ``object_id``), and refuses
predictions on images outside the universe (a symptom of mismatched inputs
that must never be silently scored as false positives on a phantom image).

Ordering: every output tuple is canonically ordered (images sorted; ground
truth and predictions via ``matching.py``'s content-derived orders), so an
evaluation over the same content is identical regardless of insertion order.

:func:`load_ground_truth` reads an H3 split manifest (``train/val/test.jsonl``,
one ``UnifiedObject`` per line — the exact format ``export_splits`` writes and
``RTDETRDataset`` trains from), so evaluation consumes the same artifact the
trainer did.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from ..errors import EvaluationDataError, InvalidPredictionError
from ..rtdetr.data import LABEL_IDS
from ..unified import UnifiedObject
from .matching import ground_truth_order, prediction_order
from .models import Prediction


def load_ground_truth(split_path: Path) -> tuple[UnifiedObject, ...]:
    """Parse one H3 split manifest into unified objects (manifest order)."""

    if not split_path.is_file():
        raise EvaluationDataError(f"split manifest not found: {split_path}")
    objects: list[UnifiedObject] = []
    for number, line in enumerate(
        split_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            objects.append(UnifiedObject.model_validate_json(line))
        except ValidationError as exc:
            raise EvaluationDataError(
                f"{split_path.name} line {number} is not a valid unified object: {exc}"
            ) from exc
    return tuple(objects)


class PredictionAccumulator:
    """Collects one evaluation's images, ground truth, and predictions."""

    def __init__(self) -> None:
        self._images: set[str] = set()
        self._ground_truth: dict[str, UnifiedObject] = {}  # keyed by object_id
        self._predictions: list[Prediction] = []

    # --- building ---------------------------------------------------------------
    def add_image(self, image_id: str) -> None:
        """Register one image in the universe (idempotent)."""

        if not image_id:
            raise EvaluationDataError("image_id must be non-empty")
        self._images.add(image_id)

    def add_ground_truth(self, obj: UnifiedObject) -> bool:
        """Register one annotation; return whether it is evaluable.

        The image joins the universe either way — an object skipped for being
        ``ignore``-flagged or a non-detector class still proves its image was
        annotated (a genuine negative), mirroring the H4B training dataset.
        """

        self._images.add(obj.image_path)
        if obj.ignore or obj.label not in LABEL_IDS:
            return False
        existing = self._ground_truth.get(obj.object_id)
        if existing is not None:
            if existing == obj:
                raise EvaluationDataError(
                    f"duplicate ground-truth object {obj.object_id} "
                    f"({obj.image_path}, {obj.label.value})"
                )
            # Same content hash but different metadata (provenance etc.) would
            # be an H2 corpus invariant violation; refuse it identically.
            raise EvaluationDataError(
                f"conflicting ground-truth objects share id {obj.object_id}"
            )
        self._ground_truth[obj.object_id] = obj
        return True

    def add_ground_truths(self, objects: Iterable[UnifiedObject]) -> int:
        """Register many annotations; return how many are evaluable."""

        return sum(1 for obj in objects if self.add_ground_truth(obj))

    def add_prediction(self, prediction: Prediction) -> None:
        """Register one prediction; its image must already be in the universe."""

        if prediction.image_id not in self._images:
            raise InvalidPredictionError(
                f"prediction references unknown image {prediction.image_id!r}; "
                "register images (or ground truth) before predictions"
            )
        self._predictions.append(prediction)

    def add_predictions(self, predictions: Iterable[Prediction]) -> int:
        """Register many predictions; return how many were added."""

        count = 0
        for prediction in predictions:
            self.add_prediction(prediction)
            count += 1
        return count

    # --- reading (canonical orders) ----------------------------------------------
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._images))

    def ground_truth(self) -> tuple[UnifiedObject, ...]:
        return ground_truth_order(self._ground_truth.values())

    def predictions(self) -> tuple[Prediction, ...]:
        return prediction_order(self._predictions)

    @property
    def num_images(self) -> int:
        return len(self._images)

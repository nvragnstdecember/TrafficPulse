"""The helmet-detector evaluator + checkpoint-evaluation helpers (H5).

Two entry paths, one metric core
--------------------------------
* :meth:`HelmetEvaluator.evaluate` — pure computation over already-made
  predictions and ground truth. No ML framework is touched; this is what the
  bulk of the test suite exercises.
* :meth:`HelmetEvaluator.evaluate_checkpoint` — loads a trained H4B checkpoint
  (H4A metadata via :class:`CheckpointManager`, weights via
  :class:`PayloadStore`), runs deterministic inference over one H3 split, and
  feeds the decoded predictions through the same pure core. torch/transformers
  are imported lazily at call time (the package-wide P1-U7 discipline);
  without the ``rtdetr`` extra this raises the typed
  ``BackendUnavailableError``.

Everything H4B already provides is reused, never rebuilt: checkpoint
resolution (``latest``/``best``/explicit id) is the H4A manager's index;
weight payloads are the H4B store's ``ckpt-<id>.pt`` convention; the dataset
is the H4B ``RTDETRDataset`` over H3 manifests; decoding is the model's own
``post_process_object_detection`` path.

Inference determinism assumptions (documented, tested on CPU)
-------------------------------------------------------------
* The DataLoader runs ``shuffle=False, num_workers=0`` — single-process,
  sequential index order, so decoded batches align with ``dataset.image_path``
  indices by construction.
* ``eval()`` mode + ``no_grad`` on CPU is bit-deterministic; CUDA inference
  may differ at float ulp level between kernels — CPU is the deterministic
  reference for byte-identical reports.
* Decoded boxes (pixel xyxy at the **original** image size, obtained by
  passing per-image target sizes read from the image headers) are clamped to
  the image bounds; a box degenerate after clamping (zero width/height) is
  dropped — it cannot be represented as a valid ``BBox`` and cannot match any
  ground truth. A decoded label id outside the helmet label space raises
  :class:`InvalidPredictionError` (a wrong-headed checkpoint must fail loudly).

The run directory records only the H4A ``ExperimentConfig`` — not the H4B
model/data configs — so the caller supplies :class:`RTDETRModelConfig` and
:class:`DataConfig` explicitly, exactly as it did for training.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import EvaluationDataError, InvalidEvaluationConfigError, InvalidPredictionError
from ..rtdetr.data import DataConfig, RTDETRDataset, build_dataloader
from ..rtdetr.model import RTDETRModel, RTDETRModelConfig, require_torch
from ..rtdetr.payload import PayloadStore
from ..split import SplitName
from ..training.checkpoint import CheckpointManager
from ..training.config import CheckpointPolicy
from ..training.run_layout import RunLayout
from ..training.state import CheckpointRecord
from ..unified import BBox, UnifiedObject
from .accumulator import PredictionAccumulator, load_ground_truth
from .confusion import build_confusion_matrix
from .metrics import compute_metrics
from .models import EVAL_CLASSES, ID_TO_CLASS, EvaluationConfig, Prediction
from .reports import (
    CheckpointProvenance,
    DatasetSummary,
    EvaluationReport,
    save_report,
)

# The two symbolic checkpoint selectors; anything else is an explicit id. Real
# ids (``e####-s########``) can never collide with these names.
_SELECTORS = ("latest", "best")


class HelmetEvaluator:
    """Deterministic COCO-compatible evaluation of the binary helmet detector."""

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config if config is not None else EvaluationConfig()
        self._clock = clock

    @property
    def config(self) -> EvaluationConfig:
        return self._config

    # --- pure evaluation ---------------------------------------------------------
    def evaluate(
        self,
        predictions: Iterable[Prediction],
        ground_truth: Iterable[UnifiedObject],
        *,
        image_ids: Iterable[str] | None = None,
    ) -> EvaluationReport:
        """Evaluate already-made predictions against unified ground truth.

        ``image_ids`` declares the image universe explicitly (needed to count
        annotation-free negative images, and to reject predictions on images
        the dataset does not contain). When ``None``, the universe is inferred
        as the union of ground-truth and prediction images — every input is
        accepted. Ground-truth images always join the universe either way.
        """

        return self._report(
            tuple(predictions),
            tuple(ground_truth),
            image_ids=None if image_ids is None else tuple(image_ids),
            split=None,
            checkpoint=None,
        )

    # --- checkpoint evaluation ----------------------------------------------------
    def evaluate_checkpoint(
        self,
        run_dir: Path,
        *,
        model: RTDETRModelConfig,
        data: DataConfig,
        checkpoint: str = "latest",
        split: SplitName = SplitName.TEST,
    ) -> EvaluationReport:
        """Run inference with one trained checkpoint over one H3 split; evaluate.

        ``checkpoint`` is ``"latest"``, ``"best"``, or an explicit checkpoint id
        (resolution and errors are the H4A manager's). ``run_dir`` is the H4A
        run directory (``<output_root>/<experiment-name>``).
        """

        torch = require_torch()
        layout = RunLayout(run_dir.parent, run_dir.name)
        record = _resolve_checkpoint(layout, checkpoint)
        payload = PayloadStore(layout.checkpoints).load(record.checkpoint_id)
        if "model" not in payload:
            raise EvaluationDataError(
                f"checkpoint {record.checkpoint_id!r} payload has no 'model' weights"
            )

        model_obj = RTDETRModel.build(model)
        model_obj.load_state_dict(payload["model"])
        device = self._resolve_device(torch)
        model_obj.to(device)
        model_obj.eval()

        split_path = Path(data.splits_dir) / f"{split.value}.jsonl"
        objects = load_ground_truth(split_path)
        if not objects:
            raise EvaluationDataError(
                f"split manifest {split_path.name} contains no objects; refusing to "
                "evaluate against an empty split"
            )
        dataset = RTDETRDataset(
            split_path,
            image_root=Path(data.image_root),
            image_height=data.image_height,
            image_width=data.image_width,
        )
        predictions = self._predict(
            model_obj, dataset, image_root=Path(data.image_root), device=device, torch=torch
        )

        return self._report(
            predictions,
            objects,
            image_ids=tuple(dataset.image_path(i) for i in range(len(dataset))),
            split=split.value,
            checkpoint=CheckpointProvenance(
                checkpoint_id=record.checkpoint_id,
                epoch=record.epoch,
                global_step=record.global_step,
                roles=tuple(role.value for role in record.roles),
                metric_name=record.metric_name,
                metric_value=record.metric_value,
                run_dir=str(run_dir),
            ),
        )

    # --- persistence ---------------------------------------------------------------
    def save_report(self, report: EvaluationReport, directory: Path) -> dict[str, Path]:
        """Write evaluation.json + summary.json + metrics.csv; return the paths."""

        return save_report(report, directory)

    # --- internals -------------------------------------------------------------------
    def _report(
        self,
        predictions: Sequence[Prediction],
        ground_truth: Sequence[UnifiedObject],
        *,
        image_ids: Sequence[str] | None,
        split: str | None,
        checkpoint: CheckpointProvenance | None,
    ) -> EvaluationReport:
        accumulator = PredictionAccumulator()
        if image_ids is not None:
            for image_id in image_ids:
                accumulator.add_image(image_id)
        accumulator.add_ground_truths(ground_truth)
        if image_ids is None:  # inferred universe: predictions register their images
            for prediction in predictions:
                accumulator.add_image(prediction.image_id)
        accumulator.add_predictions(predictions)

        ordered_gt = accumulator.ground_truth()
        ordered_predictions = accumulator.predictions()
        per_class = {label.value: 0 for label in EVAL_CLASSES}
        for obj in ordered_gt:
            per_class[obj.label.value] += 1

        return EvaluationReport(
            config=self._config,
            dataset=DatasetSummary(
                split=split,
                num_images=accumulator.num_images,
                num_ground_truth=len(ordered_gt),
                ground_truth_per_class=per_class,
            ),
            metrics=compute_metrics(ordered_predictions, ordered_gt, self._config),
            confusion_matrix=build_confusion_matrix(
                ordered_predictions, ordered_gt, self._config
            ),
            checkpoint=checkpoint,
            generated_at=self._clock() if self._clock is not None else None,
        )

    def _resolve_device(self, torch: Any) -> Any:
        # Mirrors the H4B training loop's resolution, but raises the
        # evaluation-typed error — an eval caller never sees a training error.
        if self._config.device == "cpu":
            return torch.device("cpu")
        if self._config.device == "cuda":
            if not torch.cuda.is_available():
                raise InvalidEvaluationConfigError(
                    "device='cuda' was requested but CUDA is not available"
                )
            return torch.device("cuda")  # pragma: no cover - needs CUDA hardware
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _predict(
        self,
        model: RTDETRModel,
        dataset: RTDETRDataset,
        *,
        image_root: Path,
        device: Any,
        torch: Any,
    ) -> tuple[Prediction, ...]:
        from PIL import Image

        # Original (height, width) per sample: decode maps normalized boxes back
        # to the ORIGINAL pixel space, which is where ground-truth boxes live.
        sizes: list[tuple[int, int]] = []
        for index in range(len(dataset)):
            with Image.open(image_root / dataset.image_path(index)) as handle:
                sizes.append((handle.height, handle.width))

        loader = build_dataloader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=False,  # sequential order keeps batch->index alignment exact
            seed=0,  # unused with shuffle=False; the builder requires a value
            num_workers=0,  # single-process: deterministic, and eval loads are light
        )
        predictions: list[Prediction] = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                outputs = model.forward(pixel_values=pixel_values)
                count = int(pixel_values.shape[0])
                decoded = model.decode(
                    outputs,
                    target_sizes=torch.tensor(sizes[offset : offset + count]),
                    threshold=self._config.decode_threshold,
                )
                for in_batch, image_result in enumerate(decoded):
                    image_id = dataset.image_path(offset + in_batch)
                    height, width = sizes[offset + in_batch]
                    for score, label_id, box in zip(
                        image_result["scores"].tolist(),
                        image_result["labels"].tolist(),
                        image_result["boxes"].tolist(),
                        strict=True,
                    ):
                        prediction = _decoded_prediction(
                            image_id,
                            score=float(score),
                            label_id=int(label_id),
                            xyxy=tuple(float(v) for v in box),
                            width=width,
                            height=height,
                        )
                        if prediction is not None:
                            predictions.append(prediction)
                offset += count
        return tuple(predictions)


def _decoded_prediction(
    image_id: str,
    *,
    score: float,
    label_id: int,
    xyxy: tuple[float, ...],
    width: int,
    height: int,
) -> Prediction | None:
    """One decoded detection as a validated Prediction (None when degenerate)."""

    if label_id not in ID_TO_CLASS:
        raise InvalidPredictionError(
            f"decoded label id {label_id} is outside the helmet label space "
            f"{sorted(ID_TO_CLASS)}; the checkpoint does not match the binary detector"
        )
    x1, y1, x2, y2 = xyxy
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None  # degenerate after clamping: unrepresentable and unmatchable
    return Prediction(
        image_id=image_id,
        label=ID_TO_CLASS[label_id],
        score=score,
        bbox=BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
    )


def _resolve_checkpoint(layout: RunLayout, checkpoint: str) -> CheckpointRecord:
    """Resolve latest/best/explicit-id via the H4A manager's own index.

    The manager needs a policy only when *saving*; reads never consult it, so a
    default policy makes it a pure reader here.
    """

    manager = CheckpointManager(layout.checkpoints, CheckpointPolicy())
    if checkpoint == "latest":
        return manager.latest().record
    if checkpoint == "best":
        return manager.best().record
    return manager.load(checkpoint).record


def evaluate_checkpoint(
    run_dir: Path,
    *,
    model: RTDETRModelConfig,
    data: DataConfig,
    checkpoint: str = "latest",
    split: SplitName = SplitName.TEST,
    config: EvaluationConfig | None = None,
    output_dir: Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> EvaluationReport:
    """Evaluate one checkpoint (``latest``/``best``/explicit id) of one run.

    When ``output_dir`` is given the report artifacts are also written there.
    """

    evaluator = HelmetEvaluator(config, clock=clock)
    report = evaluator.evaluate_checkpoint(
        run_dir, model=model, data=data, checkpoint=checkpoint, split=split
    )
    if output_dir is not None:
        evaluator.save_report(report, output_dir)
    return report


def evaluate_checkpoints(
    run_dir: Path,
    *,
    model: RTDETRModelConfig,
    data: DataConfig,
    split: SplitName = SplitName.TEST,
    config: EvaluationConfig | None = None,
    output_root: Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> tuple[EvaluationReport, ...]:
    """Evaluate every retained checkpoint of one run, oldest first.

    The id order is the H4A manager's retained history. A run with no
    checkpoints yields an empty tuple (nothing to evaluate is a valid answer
    for a directory sweep — asking for a *specific* checkpoint that is absent
    still raises). With ``output_root``, each report lands in
    ``<output_root>/<checkpoint_id>/``.
    """

    layout = RunLayout(run_dir.parent, run_dir.name)
    manager = CheckpointManager(layout.checkpoints, CheckpointPolicy())
    reports: list[EvaluationReport] = []
    for checkpoint_id in manager.checkpoint_ids():
        reports.append(
            evaluate_checkpoint(
                run_dir,
                model=model,
                data=data,
                checkpoint=checkpoint_id,
                split=split,
                config=config,
                output_dir=None if output_root is None else output_root / checkpoint_id,
                clock=clock,
            )
        )
    return tuple(reports)

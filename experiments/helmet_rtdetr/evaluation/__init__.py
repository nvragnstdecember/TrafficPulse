"""Deterministic evaluation framework for the RT-DETR helmet detector (H5).

Public API of the evaluation subsystem, exposed as ``helmet_rtdetr.evaluation``
(mirroring ``helmet_rtdetr.training`` and ``helmet_rtdetr.rtdetr``): the data
pipeline (H1–H3), training (H4A/H4B), and evaluation stay separable namespaces.

This unit **evaluates trained checkpoints**; it neither trains nor performs
runtime inference for the ``trafficpulse`` pipeline. The metric core (matching,
AP, confusion, reports) is pure Python — importing this package touches no ML
framework; only :meth:`HelmetEvaluator.evaluate_checkpoint` needs the optional
``rtdetr`` extra, lazily, and raises the typed ``BackendUnavailableError``
without it.
"""

from __future__ import annotations

from ..errors import (
    EvaluationDataError,
    EvaluationError,
    InvalidEvaluationConfigError,
    InvalidPredictionError,
)
from .accumulator import PredictionAccumulator, load_ground_truth
from .confusion import BACKGROUND_LABEL, ConfusionMatrix, build_confusion_matrix
from .evaluator import HelmetEvaluator, evaluate_checkpoint, evaluate_checkpoints
from .matching import (
    MatchResult,
    cap_per_image,
    ground_truth_order,
    iou,
    match_greedy,
    prediction_order,
)
from .metrics import (
    ClassMetrics,
    EvaluationMetrics,
    compute_metrics,
    evaluable_ground_truth,
    interpolated_average_precision,
)
from .models import (
    COCO_IOU_THRESHOLDS,
    EVAL_CLASSES,
    ID_TO_CLASS,
    EvaluationConfig,
    Prediction,
)
from .reports import (
    EVALUATION_SCHEMA_VERSION,
    CheckpointProvenance,
    DatasetSummary,
    EvaluationReport,
    EvaluationSummary,
    metrics_csv,
    save_report,
)

__all__ = [
    # evaluator
    "HelmetEvaluator",
    "evaluate_checkpoint",
    "evaluate_checkpoints",
    # prediction model + configuration
    "Prediction",
    "EvaluationConfig",
    "EVAL_CLASSES",
    "ID_TO_CLASS",
    "COCO_IOU_THRESHOLDS",
    # accumulation
    "PredictionAccumulator",
    "load_ground_truth",
    # matching
    "MatchResult",
    "match_greedy",
    "iou",
    "prediction_order",
    "ground_truth_order",
    "cap_per_image",
    # metrics
    "EvaluationMetrics",
    "ClassMetrics",
    "compute_metrics",
    "evaluable_ground_truth",
    "interpolated_average_precision",
    # confusion matrix
    "ConfusionMatrix",
    "build_confusion_matrix",
    "BACKGROUND_LABEL",
    # reports
    "EvaluationReport",
    "EvaluationSummary",
    "DatasetSummary",
    "CheckpointProvenance",
    "EVALUATION_SCHEMA_VERSION",
    "metrics_csv",
    "save_report",
    # errors
    "EvaluationError",
    "InvalidEvaluationConfigError",
    "InvalidPredictionError",
    "EvaluationDataError",
]

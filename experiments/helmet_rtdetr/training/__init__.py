"""Reusable, model-agnostic training infrastructure (H4A).

The public API of the training subsystem. Deliberately exposed as
``helmet_rtdetr.training`` rather than folded into the package's top-level
namespace: the H1–H3 data-pipeline API and the training API are separable
subsystems, and a consumer of one should not import the other's world.

Nothing in this subpackage imports torch, transformers, or any ML framework; the
H4B unit plugs the actual RT-DETR loop into these seams.
"""

from __future__ import annotations

from ..errors import (
    CheckpointError,
    CheckpointNotFoundError,
    DuplicateExperimentError,
    InvalidMetricNameError,
    InvalidMetricValueError,
    InvalidTrainingConfigError,
    MetricNotFoundError,
    ResumeError,
    TrainerStateError,
    TrainingError,
)
from .callbacks import Callback, CallbackList
from .checkpoint import CheckpointManager
from .config import (
    AdamWConfig,
    CheckpointPolicy,
    CosineSchedulerConfig,
    ExperimentConfig,
    LoggingConfig,
    OneCycleSchedulerConfig,
    OptimizerConfig,
    ResumeConfig,
    SchedulerConfig,
    SgdConfig,
    StepSchedulerConfig,
)
from .events import (
    JsonlLogSink,
    LogEvent,
    LogEventKind,
    LogSink,
    MemoryLogSink,
    NullLogSink,
    build_sink,
)
from .metrics import (
    METRIC_NAME_RE,
    MetricPoint,
    MetricsDump,
    MetricSeries,
    MetricsStore,
    validate_metric_name,
)
from .run_layout import DEFAULT_RUNS_ROOT, RunLayout
from .seeding import AppliedSeeds, SeedPlan, apply_seed_plan, derive_seed_plan
from .state import (
    CheckpointFile,
    CheckpointRecord,
    CheckpointRole,
    RunPhase,
    TrainingState,
)
from .trainer import Trainer

__all__ = [
    # trainer + state
    "Trainer",
    "TrainingState",
    "RunPhase",
    "CheckpointFile",
    "CheckpointRecord",
    "CheckpointRole",
    # configuration
    "ExperimentConfig",
    "OptimizerConfig",
    "AdamWConfig",
    "SgdConfig",
    "SchedulerConfig",
    "CosineSchedulerConfig",
    "StepSchedulerConfig",
    "OneCycleSchedulerConfig",
    "CheckpointPolicy",
    "LoggingConfig",
    "ResumeConfig",
    # checkpointing
    "CheckpointManager",
    # metrics
    "MetricsStore",
    "MetricPoint",
    "MetricSeries",
    "MetricsDump",
    "METRIC_NAME_RE",
    "validate_metric_name",
    # callbacks
    "Callback",
    "CallbackList",
    # events / logging
    "LogEvent",
    "LogEventKind",
    "LogSink",
    "MemoryLogSink",
    "JsonlLogSink",
    "NullLogSink",
    "build_sink",
    # seeding
    "SeedPlan",
    "AppliedSeeds",
    "derive_seed_plan",
    "apply_seed_plan",
    # layout
    "RunLayout",
    "DEFAULT_RUNS_ROOT",
    # errors
    "TrainingError",
    "InvalidTrainingConfigError",
    "DuplicateExperimentError",
    "TrainerStateError",
    "ResumeError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "InvalidMetricNameError",
    "InvalidMetricValueError",
    "MetricNotFoundError",
]

"""Real-time inference engine for TrafficPulse (H6).

The composition layer that runs the architecture's full flow -- video ->
RT-DETR inference -> IoU tracking -> rule reasoning -> hypothesis lifecycle ->
``ConfirmedEvent`` -> ``EvidenceManifest`` -> persistence -- over the shipped,
independently-tested seams (P1-U5 ingestion, P1-U6/U7 detection, P1-U8/U9
tracking, P1-U3/U4 + P2-U4 + P4-U5 reasoning, P3-U2 orchestration core, P1-U11
persistence). It adds the real-time envelope -- frame sources, deterministic
scheduling with back-pressure, batched/timed inference, multi-rule composition,
evidence frame references, metrics, structured logging -- and **no** new
reasoning.

Importing this package pulls in no ML framework: real backends are built only
by the :func:`build_engine` / :func:`~trafficpulse.engine.runner.build_detector`
composition roots, lazily. Everything in the engine's decision path is
deterministic; wall-clock and resource measurements exist only under injected
clocks/probes.
"""

from .config import (
    EngineConfig,
    EngineTrackerConfig,
    EvidenceConfig,
    IllegalStoppingRuleConfig,
    InferenceConfig,
    NoHelmetRuleConfig,
    RuleConfig,
    SchedulerConfig,
    WrongWayRuleConfig,
)
from .engine import EngineRunResult, InferenceEngine, build_engine
from .errors import (
    EngineConfigurationError,
    EngineError,
    FrameSourceError,
    RunCancelledError,
    UnsupportedRuleError,
)
from .evidence import FrameStamp, build_engine_manifest, media_seconds, pick_evidence_frames
from .logs import (
    EngineLogEvent,
    EngineLogEventKind,
    EngineLogSink,
    JsonlLogSink,
    MemoryLogSink,
    NullLogSink,
)
from .metrics import (
    EngineMetrics,
    LatencyKind,
    LatencySummary,
    MetricsRecorder,
    torch_cuda_memory_probe,
)
from .rules import (
    BuiltRule,
    CompositeFrameObserver,
    MultiRuleFinalize,
    build_rules,
    require_shipped,
)
from .runner import (
    DetectorRunner,
    InstrumentedTracker,
    SupportsBatchDetect,
    build_detector,
    detector_adapter_config,
    detector_model_ref,
    resolve_device,
)
from .scheduler import FrameScheduler, ScheduleDecision
from .sources import (
    FileFrameSource,
    FrameSource,
    IterableFrameSource,
    frame_record_from_array,
)

__all__ = [
    # engine
    "InferenceEngine",
    "EngineRunResult",
    "build_engine",
    # configuration
    "EngineConfig",
    "SchedulerConfig",
    "InferenceConfig",
    "EngineTrackerConfig",
    "EvidenceConfig",
    "RuleConfig",
    "WrongWayRuleConfig",
    "IllegalStoppingRuleConfig",
    "NoHelmetRuleConfig",
    # sources
    "FrameSource",
    "FileFrameSource",
    "IterableFrameSource",
    "frame_record_from_array",
    # scheduling
    "FrameScheduler",
    "ScheduleDecision",
    # runner
    "DetectorRunner",
    "InstrumentedTracker",
    "SupportsBatchDetect",
    "build_detector",
    "detector_adapter_config",
    "detector_model_ref",
    "resolve_device",
    # rules
    "BuiltRule",
    "build_rules",
    "require_shipped",
    "MultiRuleFinalize",
    "CompositeFrameObserver",
    # evidence
    "FrameStamp",
    "build_engine_manifest",
    "pick_evidence_frames",
    "media_seconds",
    # metrics
    "EngineMetrics",
    "MetricsRecorder",
    "LatencyKind",
    "LatencySummary",
    "torch_cuda_memory_probe",
    # logging
    "EngineLogEvent",
    "EngineLogEventKind",
    "EngineLogSink",
    "MemoryLogSink",
    "JsonlLogSink",
    "NullLogSink",
    # errors
    "EngineError",
    "EngineConfigurationError",
    "UnsupportedRuleError",
    "FrameSourceError",
    "RunCancelledError",
]

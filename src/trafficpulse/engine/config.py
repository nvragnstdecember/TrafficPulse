"""Typed configuration for the real-time inference engine (H6).

Four configuration surfaces, all frozen + strict pydantic models (the same
posture as ``DetectorConfig`` / ``TrackerConfig`` / the domain contracts):

* :class:`SchedulerConfig` -- how frames flow: stride/FPS decimation and the
  bounded back-pressure queue.
* :class:`InferenceConfig` -- how the **real** RT-DETR backend is built by the
  composition root (checkpoint, device with ``auto`` fallback, thresholds,
  label map). Constructing this model loads nothing and imports no ML
  framework; only :func:`~trafficpulse.engine.runner.build_detector` realises
  it, lazily.
* :class:`RuleConfig` -- a discriminated union naming which shipped reasoning
  slices run and their per-rule options. Only violations with an existing
  reasoner are representable; unshipped ones fail loudly in the rule registry.
* :class:`EngineConfig` -- the whole engine: scheduler + rules + evidence
  margins + batching, plus the backend declarations (``inference`` /
  ``tracker``) that **only** the :func:`~trafficpulse.engine.engine.build_engine`
  composition root consumes. The ``InferenceEngine`` constructor itself takes
  injected ``Detector`` / ``Tracker`` seams and ignores those two blocks, so
  the engine class stays backend-free.

Validation split (consistent with the rest of the runtime): field-level bounds
raise pydantic ``ValidationError``; cross-field semantic rules raise the typed
:class:`~trafficpulse.engine.errors.EngineConfigurationError`.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import ModelRef, ObjectClass
from ..contracts.primitives import Confidence
from ..observations.stationary import STATIONARY_EPSILON_PX, STATIONARY_WINDOW
from ..tracking.iou_tracker import IouTrackerConfig
from .errors import EngineConfigurationError

# ``auto`` resolves to cuda-if-available at detector build time; the explicit
# forms match the P1-U7 backend's own device validation.
_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


class _EngineModel(BaseModel):
    """Frozen + strict base for every engine configuration model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- frame scheduling ----------------------------------------------------------
class SchedulerConfig(_EngineModel):
    """Deterministic frame-flow policy: decimation + bounded back-pressure.

    * ``frame_stride`` keeps every N-th *read* frame (1 = keep all).
    * ``target_fps`` additionally decimates by **media time** (PTS): a frame is
      admitted only when at least ``1/target_fps`` media-seconds have elapsed
      since the last admitted frame. ``None`` disables the FPS gate. Media time
      -- never wall-clock -- drives this, so scheduling is replayable.
    * ``queue_capacity`` bounds the pending queue between ``submit`` and
      ``drain``; a submit onto a full queue **drops the incoming frame** (the
      queued frames are older and already admitted -- dropping the newest keeps
      the admitted stream's frame order strictly monotonic) and counts it.
    """

    frame_stride: int = Field(default=1, ge=1)
    target_fps: float | None = Field(default=None, gt=0.0)
    queue_capacity: int = Field(default=64, ge=1)


# --- real-backend declaration (composition root only) ---------------------------
class InferenceConfig(_EngineModel):
    """Declaration of the real RT-DETR detector the composition root builds.

    ``checkpoint`` is an operator-supplied HuggingFace id or local directory --
    the H4B/H5 training pipeline's ``save_pretrained`` export is exactly such a
    directory, which is how *trained* helmet weights load here. ``device``
    accepts ``auto`` (cuda when available, else cpu -- the GPU/CPU fallback),
    or the explicit ``cpu`` / ``cuda[:N]`` forms the backend validates.
    ``source_model`` overrides the stamped provenance; when ``None`` a truthful
    provisional ref is derived from the checkpoint (mirroring the P1-U12
    runner). ``local_files_only`` keeps loading offline by default.
    """

    checkpoint: str = Field(min_length=1)
    label_map: dict[str, ObjectClass]
    device: str = "auto"
    score_threshold: Confidence = 0.5
    local_files_only: bool = True
    source_model: ModelRef | None = None

    @model_validator(mode="after")
    def _valid_device(self) -> Self:
        if not _DEVICE_RE.match(self.device):
            raise EngineConfigurationError(
                f"device must be 'auto', 'cpu', or 'cuda[:N]', got {self.device!r}"
            )
        if not self.label_map:
            raise EngineConfigurationError(
                "label_map must map at least one detector-native label to an ObjectClass"
            )
        return self


class EngineTrackerConfig(_EngineModel):
    """Declaration of the IoU tracker the composition root builds.

    Wraps the existing backend knobs (:class:`IouTrackerConfig`: iou_threshold /
    max_age / min_hits -- the configurable birth/update/lost/dead lifecycle
    thresholds) plus the provenance ``ModelRef`` stamped onto every
    ``TrackState.tracker``.
    """

    backend: IouTrackerConfig = IouTrackerConfig()
    tracker_ref: ModelRef | None = ModelRef(name="iou-tracker", version="0.1.0-provisional")


# --- rule declarations -----------------------------------------------------------
class WrongWayRuleConfig(_EngineModel):
    """Run the wrong-way slice; ``direction_id`` picks the governing direction
    when the scene declares more than one."""

    kind: Literal["wrong_way"] = "wrong_way"
    direction_id: str | None = None


class IllegalStoppingRuleConfig(_EngineModel):
    """Run the illegal-stopping slice with the provisional pixel-space
    stationarity parameters (defaults are the P2-U3 module defaults)."""

    kind: Literal["illegal_stopping"] = "illegal_stopping"
    stationary_window: int = Field(default=STATIONARY_WINDOW, ge=2)
    stationary_epsilon_px: float = Field(default=STATIONARY_EPSILON_PX, gt=0.0)


class NoHelmetRuleConfig(_EngineModel):
    """Run the no-helmet slice; requires an injected ``HelmetClassifier``
    (fail-fast in the rule registry when absent)."""

    kind: Literal["no_helmet"] = "no_helmet"


class TripleRidingRuleConfig(_EngineModel):
    """Run the triple-riding slice (v1.1 U3).

    Pure geometry over the shipped perception + association seams -- it needs no
    classifier. Its temporal parameters (min_persistence, rider_count_threshold,
    max_observation_gap) are read from the scene's ``triple_riding`` block."""

    kind: Literal["triple_riding"] = "triple_riding"


RuleConfig: TypeAlias = Annotated[
    WrongWayRuleConfig | IllegalStoppingRuleConfig | NoHelmetRuleConfig | TripleRidingRuleConfig,
    Field(discriminator="kind"),
]


# --- evidence ---------------------------------------------------------------------
class EvidenceConfig(_EngineModel):
    """Before/after context margins (media-seconds) for evidence frame picking."""

    before_seconds: float = Field(default=1.0, ge=0.0)
    after_seconds: float = Field(default=1.0, ge=0.0)


# --- the engine -------------------------------------------------------------------
class EngineConfig(_EngineModel):
    """Everything one engine run needs, declared and validated up front.

    ``batch_size`` groups admitted frames for the detector runner: a detector
    that implements the optional batch protocol receives whole batches; any
    other detector is driven frame-by-frame with identical results. ``rules``
    must name at least one shipped rule. ``inference`` / ``tracker`` are read
    **only** by the ``build_engine`` composition root (the engine class takes
    injected seams); ``inference`` may stay ``None`` when the caller always
    injects a detector.
    """

    rules: tuple[RuleConfig, ...]
    scheduler: SchedulerConfig = SchedulerConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    batch_size: int = Field(default=1, ge=1)
    inference: InferenceConfig | None = None
    tracker: EngineTrackerConfig = EngineTrackerConfig()

    @model_validator(mode="after")
    def _at_least_one_rule(self) -> Self:
        if not self.rules:
            raise EngineConfigurationError(
                "an engine must be configured with at least one rule"
            )
        return self

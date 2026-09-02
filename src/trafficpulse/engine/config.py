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
from ..contracts.enums import SignalState
from ..contracts.primitives import Confidence
from ..observations.helmet_stability import HelmetStabilizationConfig
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

    acknowledge_turban_blind: bool = False
    """Permit building this rule on a backend that has **declared** it cannot emit
    ``turban``.

    The rule exempts a rider whose predominant observation is ``turban``
    (``rules.no_helmet.exempt_riders``). A binary backend never produces one, so the
    exemption silently becomes dead code and turban-wearing riders are confirmed as
    violations -- a systematic false-positive class against a religious group, and a
    reversal of the H8 real-footage fix. Nothing raises on its own, which is exactly
    why the rule registry refuses the combination by default.

    Setting this to ``True`` does **not** make the consequence go away; it records
    that an operator accepted it deliberately, so the choice is auditable rather than
    accidental. Leave it ``False`` unless you have read
    :mod:`trafficpulse.classifier.capabilities` and mean it."""


class TripleRidingRuleConfig(_EngineModel):
    """Run the triple-riding slice (v1.1 U3).

    Pure geometry over the shipped perception + association seams -- it needs no
    classifier. Its temporal parameters (min_persistence, rider_count_threshold,
    max_observation_gap) are read from the scene's ``triple_riding`` block."""

    kind: Literal["triple_riding"] = "triple_riding"


class SignalPhaseSpec(_EngineModel):
    """One declared phase of a run's signal schedule: a state from a media offset.

    ``at_seconds`` is measured from the start of the clip -- the number an operator
    reads off a player's scrub bar -- and is converted to a media-time instant when
    the rule is built. Seconds rather than an absolute datetime because that is the
    form the fact is actually known in, and converting once at the boundary is
    safer than asking every caller to anchor the epoch correctly."""

    at_seconds: float = Field(ge=0.0)
    state: SignalState


class RedLightRuleConfig(_EngineModel):
    """Run the red-light-jumping slice (H13).

    Carries the **per-run signal schedule**, which is the one thing about this rule
    that cannot live in the scene: a phase names a media-time instant, and media
    time belongs to one clip, whereas a ``SceneConfig`` is per-camera and shared
    across many. The scene stays camera geometry (stop line, junction zone, signal
    group); this carries the timing.

    ``stop_line_id`` / ``zone_id`` select the governing approach when the scene
    declares more than one -- the same shape as ``WrongWayRuleConfig.direction_id``.

    A schedule is **required**: with no phases every instant resolves to
    ``SignalState.UNKNOWN`` and the rule could never confirm, so accepting an empty
    one would ship a rule that silently does nothing. The rule registry refuses it
    at build time, exactly as it refuses a ``no_helmet`` rule with no classifier."""

    kind: Literal["red_light_jumping"] = "red_light_jumping"
    schedule: tuple[SignalPhaseSpec, ...] = ()
    stop_line_id: str | None = None
    zone_id: str | None = None

    @model_validator(mode="after")
    def _monotonic_schedule(self) -> Self:
        # A plain ValueError, not the typed EngineConfigurationError the sibling
        # configs raise: this is a constraint *within* one field rather than a
        # cross-field semantic rule, and -- unlike the backend declarations -- this
        # config is parsed straight from a client request body, where pydantic's
        # ValidationError is what turns a malformed schedule into a clean 422
        # instead of an unhandled error.
        offsets = [phase.at_seconds for phase in self.schedule]
        if offsets != sorted(offsets):
            raise ValueError(
                "signal schedule phases must be declared in non-decreasing "
                "at_seconds order so the step function is unambiguous"
            )
        return self


RuleConfig: TypeAlias = Annotated[
    WrongWayRuleConfig
    | IllegalStoppingRuleConfig
    | NoHelmetRuleConfig
    | TripleRidingRuleConfig
    | RedLightRuleConfig,
    Field(discriminator="kind"),
]


# --- analyses (perception without enforcement) ------------------------------------
class HelmetAnalysisConfig(_EngineModel):
    """Run helmet classification as **analysis only**: perception, never enforcement.

    Why this is not a rule
    ----------------------
    Helmet *classification* and the no-helmet *violation decision* are different
    claims, and until now the only way to obtain the first was to configure the
    second. That coupling is what makes a turban-blind backend an all-or-nothing
    choice: either run a violation rule whose exemption can never fire, or see no
    helmet state at all.

    An analysis produces **no** ``ConfirmedEvent``, has no reasoner, reads no rule
    parameters from the scene, and is invisible to the event store and the evidence
    manifest. It registers the existing P4-U4 :class:`HelmetFrameObserver`, so the
    classifier runs and its per-rider output is available for inspection and for the
    overlay framework -- and nothing downstream can mistake a label for a violation,
    because no violation is ever minted.

    That distinction is exactly what the P4-U9/P4-U10 evidence requires
    (``docs/helmet-runtime-evaluation.md``): the classifier is measurable and
    demonstrable on runtime crops, while the violation rule is not currently safe to
    run on a backend that cannot express the turban exemption. This declaration lets
    a deployment say "classify, do not enforce" **without** touching the turban
    capability guard, which continues to refuse the rule.

    Scene-independent by construction: a scene needs no ``no_helmet`` parameter block
    to support analysis, because there is no persistence window to resolve.
    """

    kind: Literal["helmet_analysis"] = "helmet_analysis"

    stabilization: HelmetStabilizationConfig = HelmetStabilizationConfig()
    """Temporal smoothing policy for the labels this analysis *reports*.

    Presentation only, and deliberately scoped to an analysis rather than offered to
    the ``no_helmet`` rule: the rule's temporal-run semantics are its own and have
    never been evaluated against smoothed input, so feeding it smoothed labels would
    silently change a reasoner nobody has re-validated. See
    :mod:`trafficpulse.observations.helmet_stability` -- the window is a legibility
    choice, not a tuned parameter, and no accuracy claim rests on it."""


AnalysisConfig: TypeAlias = Annotated[
    HelmetAnalysisConfig,
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
    analysis: tuple[AnalysisConfig, ...] = ()
    """Perception-only declarations that run beside the rules and emit no events.

    Separate from ``rules`` deliberately: everything in ``rules`` can mint a
    ``ConfirmedEvent``, and nothing in ``analysis`` ever can. Keeping them in one
    list would put that difference in a comment instead of in the type."""

    scheduler: SchedulerConfig = SchedulerConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    batch_size: int = Field(default=1, ge=1)
    inference: InferenceConfig | None = None
    tracker: EngineTrackerConfig = EngineTrackerConfig()

    @model_validator(mode="after")
    def _at_least_one_rule(self) -> Self:
        # An analysis-only engine is legitimate: it observes and classifies without
        # asserting any violation. What is refused is an engine asked to do nothing
        # at all, which is a configuration mistake either way.
        if not self.rules and not self.analysis:
            raise EngineConfigurationError(
                "an engine must be configured with at least one rule or analysis"
            )
        return self

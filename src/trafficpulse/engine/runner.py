"""The RT-DETR runner: instrumentation, batched inference, real-backend build (H6).

Three concerns, all behind the existing P1-U6/P1-U8 seams -- nothing here alters
what crosses a seam, only *when* inference runs and what gets measured:

* :class:`DetectorRunner` -- the engine's detector decorator. It implements
  ``Detector`` (transparent to the pipeline), counts every raw detection, and
  times every **real inference call** through the injected ``perf`` probe (one
  latency sample per call -- a batch call is one sample; no probe, no timing,
  never fabricated). When the wrapped backend implements the optional
  :class:`SupportsBatchDetect` capability, the engine ``prefetch``\\ es one
  admitted batch -- a single ``detect_batch`` call -- and the pipeline's
  subsequent per-frame ``detect`` calls are served from those results by frame
  identity. A backend without the capability (the current ``RTDetrDetector``,
  every stub) is driven frame-by-frame; for a deterministic backend the two
  paths yield identical detections by construction, which the tests assert.
* :class:`InstrumentedTracker` -- the tracker analogue: times each ``update``
  and counts emitted track states; behaviour is pass-through.
* :func:`build_detector` -- the composition root that realises an
  :class:`InferenceConfig` as the **real** RT-DETR backend (P1-U7), resolving
  ``device="auto"`` to CUDA-when-available (the GPU/CPU fallback). torch and
  the backend module are imported lazily inside the functions, so importing
  ``trafficpulse.engine`` keeps the runtime ML-free.

Loading *trained* weights: the H4B/H5 helmet training pipeline exports a
HuggingFace-layout directory (``save_pretrained``); pointing
:attr:`InferenceConfig.checkpoint` at that directory loads the trained detector
through the same offline path as any cached checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..contracts import Detection, ModelRef, TrackState
from ..detector.config import DetectorConfig
from ..detector.frame import Frame
from ..detector.interface import Detector
from ..detector.raw import RawDetection
from ..tracking.interface import Tracker
from .config import InferenceConfig
from .metrics import LatencyKind, MetricsRecorder


@runtime_checkable
class SupportsBatchDetect(Protocol):
    """Optional capability: run inference over several frames in one call.

    ``detect_batch`` must return one ``RawDetection`` sequence per input frame,
    in input order, and must be equivalent to calling ``detect`` per frame (a
    deterministic backend's batching may not change its per-frame output).
    """

    def detect_batch(self, frames: Sequence[Frame]) -> Sequence[Sequence[RawDetection]]:
        """Return per-frame detections for ``frames``, in input order."""
        ...


class DetectorRunner(Detector):
    """Timed, batch-capable detector decorator (see module docstring).

    Frame identity for prefetched results is ``(camera_id, frame_index)`` --
    the same identity the tracker seam orders by. ``detect`` falls through to
    the wrapped backend for any frame that was not prefetched, so correctness
    never depends on prefetching having happened.
    """

    def __init__(self, inner: Detector, recorder: MetricsRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self._prefetched: dict[tuple[str, int], tuple[RawDetection, ...]] = {}

    @property
    def inner(self) -> Detector:
        return self._inner

    def prefetch(self, frames: Sequence[Frame]) -> bool:
        """Batch-infer ``frames`` when the backend can; report whether it did.

        A no-op (``False``) for single-frame batches -- nothing to gain -- and
        for backends without :class:`SupportsBatchDetect`.
        """

        if len(frames) < 2 or not isinstance(self._inner, SupportsBatchDetect):
            return False
        perf = self._recorder.perf
        started = perf() if perf is not None else None
        results = self._inner.detect_batch(frames)
        if perf is not None and started is not None:
            self._recorder.observe_latency(LatencyKind.INFERENCE, perf() - started)
        for frame, raw in zip(frames, results, strict=True):
            self._prefetched[(frame.camera_id, frame.frame_index)] = tuple(raw)
        return True

    def detect(self, frame: Frame) -> Sequence[RawDetection]:
        key = (frame.camera_id, frame.frame_index)
        if key in self._prefetched:
            raw: Sequence[RawDetection] = self._prefetched.pop(key)
        else:
            perf = self._recorder.perf
            started = perf() if perf is not None else None
            raw = self._inner.detect(frame)
            if perf is not None and started is not None:
                self._recorder.observe_latency(LatencyKind.INFERENCE, perf() - started)
        self._recorder.detections += len(raw)
        return raw

    def clear(self) -> None:
        """Drop any prefetched results (engine reset)."""

        self._prefetched.clear()


class InstrumentedTracker(Tracker):
    """Times and counts an inner tracker's updates; behaviour is pass-through."""

    def __init__(self, inner: Tracker, recorder: MetricsRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def update(self, detections: Sequence[Detection]) -> Sequence[TrackState]:
        perf = self._recorder.perf
        started = perf() if perf is not None else None
        states = self._inner.update(detections)
        if perf is not None and started is not None:
            self._recorder.observe_latency(LatencyKind.TRACKING, perf() - started)
        self._recorder.track_states += len(states)
        return states

    def reset(self) -> None:
        self._inner.reset()


def detector_model_ref(config: InferenceConfig) -> ModelRef:
    """The truthful provenance ref for a built detector.

    The explicit ``source_model`` when configured; otherwise derived from the
    checkpoint with a provisional version marker (no pinned model version is
    claimed and no weights hash is fabricated) -- the P1-U12 convention.
    """

    if config.source_model is not None:
        return config.source_model
    return ModelRef(name=config.checkpoint, version="provisional")


def detector_adapter_config(config: InferenceConfig) -> DetectorConfig:
    """The P1-U6 adapter configuration an :class:`InferenceConfig` implies."""

    return DetectorConfig(
        label_map=dict(config.label_map),
        score_threshold=config.score_threshold,
        source_model=detector_model_ref(config),
    )


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to ``cuda``-when-available, else ``cpu`` (lazy torch).

    Explicit devices pass through unchanged (the backend validates them). When
    torch is absent, ``auto`` resolves to ``cpu`` -- the backend will then fail
    fast with its own typed dependency error if construction proceeds, which is
    the accurate diagnosis.
    """

    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch present in this env
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_detector(config: InferenceConfig) -> Detector:
    """Realise the declared **real** RT-DETR backend (composition root).

    Imports the P1-U7 backend lazily; fails fast with its typed errors when the
    optional ``rtdetr`` extra or the checkpoint is unavailable. The backend
    pre-filter threshold is set to the adapter's authoritative gate (never
    above it), the documented two-threshold discipline.
    """

    from ..detector.rtdetr import RTDetrConfig, RTDetrDetector

    return RTDetrDetector(
        RTDetrConfig(
            checkpoint=config.checkpoint,
            device=resolve_device(config.device),
            local_files_only=config.local_files_only,
            threshold=config.score_threshold,
        )
    )

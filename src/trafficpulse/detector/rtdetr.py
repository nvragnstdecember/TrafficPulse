"""The first real detector backend: RT-DETR via HuggingFace Transformers (P1-U7).

This is the smallest real RT-DETR inference path behind the frozen P1-U6
``Detector`` boundary. It integrates the **Apache-2.0** RT-DETR port shipped in
HuggingFace ``transformers`` -- the permissive-only direction ADR-001 mandates
(permissive code path only; **no Ultralytics / AGPL**; RT-DETR primary). The
detector produces framework-neutral ``RawDetection`` values; the existing
``DetectionAdapter`` remains the sole authority that stamps frozen U2
``Detection`` contracts. Nothing framework-native -- tensors, processors, model
handles, device objects, model-native result dicts -- escapes this module.

Where the ML dependency lives
-----------------------------
``torch`` and ``transformers`` are an **optional** extra (``trafficpulse[rtdetr]``)
and are imported **lazily**, only inside :meth:`_TransformersRTDetrEngine.load`
and :meth:`_TransformersRTDetrEngine.infer`. Consequences, all tested:

* importing this module (or ``trafficpulse.detector``) pulls in **no** ML
  framework -- the P1-U6 boundary invariant is preserved;
* constructing :class:`RTDetrConfig` loads nothing and downloads nothing;
* the base install and every unit test stay ML-free and network-free.

Internal inference seam
-----------------------
:class:`RTDetrInferenceEngine` is a small **framework-neutral** protocol: it
returns :class:`EngineDetection` values carrying only plain ``int``/``float``
Python types -- never a tensor. :class:`_TransformersRTDetrEngine` is the real
implementation (the only code that touches torch/transformers). Because the seam
is framework-neutral, :class:`RTDetrDetector` is fully testable with a fake engine
and no ML dependency, and no framework object can leak through it.

Input image assumptions
-----------------------
``frame.image`` is the ingestion/U6 payload: an RGB ``uint8`` array of shape
``(height, width, 3)`` (see P1-U5 ``FrameRecord`` and P1-U6 ``Frame``). It is
passed **as-is** to the transformers image processor -- no colour-space change
(transformers image processors consume RGB, matching ingestion output), no copy,
no in-place mutation, no resize by this module. A ``frame.image is None`` (identity
carried without pixels, e.g. the stub path) raises :class:`MissingFrameImageError`,
because a real detector needs pixels.

Preprocessing / coordinate-conversion boundary
-----------------------------------------------
All preprocessing (resize, normalisation, tensor packing) happens **inside** the
transformers image processor, and the inverse coordinate transform happens inside
``post_process_object_detection(target_sizes=(height, width))``. That call returns
boxes already mapped back to **original-frame** ``(x1, y1, x2, y2)`` **pixel**
coordinates (top-left origin, ``+x`` right, ``+y`` down). Preprocessing-space,
normalised, or centre-width-height coordinates never escape the engine. RT-DETR is
NMS-free, so no non-maximum-suppression step is applied here.

RT-DETR (like most detectors) can predict coordinates fractionally **outside** the
frame, which the frozen ``BoundingBox`` contract (non-negative, ``x2>x1``) rejects.
The detector therefore **clips** every emitted box to the original image rectangle
``[0, width] x [0, height]`` so the coordinates stay in original-image pixel space
and satisfy the contract; a box with **no in-frame area** after clipping is entirely
off-screen and is **dropped**. This is the detector's own coordinate step
(framework-neutral, unit-tested with a fake engine) -- not the engine's.

Native labels
-------------
The engine maps the model's integer class id to the model's **native** label
string via ``model.config.id2label`` and emits it unchanged. This module invents
**no** TrafficPulse semantic class: the native label travels in
``RawDetection.label``, and the existing ``DetectorConfig.label_map`` (consumed by
``DetectionAdapter``) decides which native labels become ``ObjectClass`` values.
An unmapped native label retains the P1-U6 behaviour: the adapter drops it.

Two confidence thresholds (distinct, both documented)
-----------------------------------------------------
* :attr:`RTDetrConfig.threshold` is a backend **efficiency pre-filter**. RT-DETR
  always emits a fixed set of object queries; ``post_process_object_detection``
  requires a threshold to prune near-zero-score boxes before they are ever
  materialised as ``RawDetection``. It is *not* the authoritative gate.
* ``DetectorConfig.score_threshold`` (on the adapter) remains the **authoritative**
  confidence gate for the produced ``Detection``. Set the backend threshold
  **at or below** the adapter threshold so the pre-filter never hides a detection
  the adapter would have kept.

Model acquisition
-----------------
:class:`RTDetrConfig.checkpoint` is an operator-supplied HuggingFace model id or a
local directory path; this unit ships, blesses, and defaults to **no** checkpoint.
``local_files_only`` defaults to ``True`` (offline: load only already-cached / local
artifacts; never download). Constructing :class:`RTDetrDetector` **without** an
injected engine loads the checkpoint (fail-fast). If the optional deps are absent
it raises :class:`BackendDependencyError`; if the checkpoint is not locally
available it raises :class:`ModelArtifactUnavailableError`. Per ADR-001, weight and
pretraining-data provenance is a **per-artifact** review (U4 registry), not implied
by the Apache-2.0 code licence -- so TrafficPulse redistributes no weights here.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from ..contracts.primitives import Confidence, NonEmptyStr
from .errors import DetectorError
from .frame import Frame
from .interface import Detector
from .raw import RawDetection

_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


# --- backend error taxonomy --------------------------------------------------
class RTDetrBackendError(DetectorError):
    """Base class for RT-DETR backend errors (a :class:`DetectorError` subclass).

    Every backend failure surfaces as one of these stable TrafficPulse errors, so
    callers never see a raw ``torch`` / ``transformers`` exception cross the
    detector boundary. Originating framework exceptions are chained as ``__cause__``.
    """


class BackendDependencyError(RTDetrBackendError):
    """The optional ``torch`` / ``transformers`` backend dependencies are missing."""


class ModelArtifactUnavailableError(RTDetrBackendError):
    """The requested RT-DETR checkpoint is not available locally.

    Raised when ``from_pretrained`` cannot resolve the checkpoint (e.g.
    ``local_files_only=True`` and nothing is cached). The backend never downloads
    silently to recover.
    """


class InvalidDeviceError(RTDetrBackendError):
    """A device was requested that this environment cannot provide (e.g. CUDA)."""


class MissingFrameImageError(RTDetrBackendError):
    """A real detector was asked to run on a frame with no ``image`` payload."""


class MalformedBackendOutputError(RTDetrBackendError):
    """The inference engine returned a structurally invalid detection.

    A wrong-arity box, or a class id absent from the model's label map. Numeric
    range / box-geometry validation is deliberately **not** duplicated here -- that
    remains the ``DetectionAdapter``'s authoritative contract check.
    """


class BackendInferenceError(RTDetrBackendError):
    """A framework-level failure occurred during inference or post-processing."""


# --- configuration -----------------------------------------------------------
class RTDetrConfig(BaseModel):
    """Backend-specific runtime configuration for the RT-DETR detector.

    Deliberately separate from the framework-neutral ``DetectorConfig`` (which
    configures the *adapter*): these fields are meaningless to any other backend
    and must not leak into the shared seam. Frozen + strict like the domain
    contracts. Exposes **no** framework-native object (no ``torch.device``): the
    device is a validated string.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: NonEmptyStr
    """RT-DETR checkpoint identity: a HuggingFace model id or a local directory
    path with the weights + config. No default -- the operator chooses the artifact,
    whose weight/pretraining-data provenance is reviewed per ADR-001 (U4 registry)."""

    device: str = "cpu"
    """Execution device: ``"cpu"`` (default), ``"cuda"``, or ``"cuda:N"``. CUDA is
    an explicit opt-in and is only honoured if the backend reports it available."""

    local_files_only: bool = True
    """Offline by default: load only already-local/cached artifacts, never download.
    Set ``False`` to allow ``transformers`` to fetch the checkpoint (explicit)."""

    threshold: Confidence = 0.5
    """Backend efficiency pre-filter passed to ``post_process_object_detection`` to
    prune RT-DETR's fixed near-zero-score queries. The adapter's ``score_threshold``
    remains the authoritative gate; keep this at or below it."""

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_RE.match(value):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N', got {value!r}")
        return value


# --- internal, framework-neutral inference seam ------------------------------
@dataclass(frozen=True)
class EngineDetection:
    """One post-processed detection at the internal engine seam (framework-neutral).

    Carries only plain Python scalars -- never a tensor. ``box`` is already in
    **original-frame** ``(x1, y1, x2, y2)`` pixel coordinates and ``label_id`` is
    the model's integer class id (resolved to a native label string by the engine).
    """

    label_id: int
    score: float
    box: tuple[float, float, float, float]


class RTDetrInferenceEngine(Protocol):
    """The framework-neutral inference seam :class:`RTDetrDetector` depends on.

    The real engine (:class:`_TransformersRTDetrEngine`) is the only object that
    touches torch/transformers; a fake engine satisfies this protocol in tests with
    no ML dependency. Only plain Python types cross this seam.
    """

    def label_name(self, label_id: int) -> str | None:
        """Return the model's native label string for ``label_id`` (or ``None``)."""
        ...

    def infer(
        self, image: NDArray[np.uint8], *, threshold: float
    ) -> Sequence[EngineDetection]:
        """Run inference on one RGB ``uint8`` image and return original-pixel boxes."""
        ...


# --- detector ----------------------------------------------------------------
class RTDetrDetector(Detector):
    """A real RT-DETR :class:`~trafficpulse.detector.interface.Detector`.

    Satisfies the exact P1-U6 interface: ``detect(frame) -> Sequence[RawDetection]``.
    Construct with only a :class:`RTDetrConfig` to load the real transformers engine
    (fail-fast), or inject an :class:`RTDetrInferenceEngine` (tests / advanced
    embedding). The injected-engine path touches no ML dependency.
    """

    def __init__(
        self, config: RTDetrConfig, *, engine: RTDetrInferenceEngine | None = None
    ) -> None:
        self._config = config
        self._engine: RTDetrInferenceEngine = (
            engine if engine is not None else _TransformersRTDetrEngine.load(config)
        )

    @property
    def config(self) -> RTDetrConfig:
        return self._config

    def detect(self, frame: Frame) -> Sequence[RawDetection]:
        """Run RT-DETR on ``frame.image`` and return framework-neutral detections.

        Raises:
            MissingFrameImageError: ``frame.image`` is ``None``.
            MalformedBackendOutputError: the engine returned a structurally invalid
                detection (wrong-arity box, or an unmapped class id).
            BackendInferenceError: a framework-level inference failure.
        """

        image = frame.image
        if image is None:
            raise MissingFrameImageError(
                "RT-DETR requires pixels, but frame.image is None "
                f"(camera_id={frame.camera_id!r}, frame_index={frame.frame_index})"
            )
        height, width = int(image.shape[0]), int(image.shape[1])
        engine_detections = self._engine.infer(image, threshold=self._config.threshold)
        raws: list[RawDetection] = []
        for det in engine_detections:
            raw = self._to_raw(det, width=width, height=height)
            if raw is not None:  # None == box entirely outside the frame (dropped)
                raws.append(raw)
        return tuple(raws)

    def _to_raw(self, det: EngineDetection, *, width: int, height: int) -> RawDetection | None:
        box = det.box
        if len(box) != 4:
            raise MalformedBackendOutputError(
                f"engine emitted a box with {len(box)} coordinate(s), expected 4: {box!r}"
            )
        label = self._engine.label_name(det.label_id)
        if label is None:
            raise MalformedBackendOutputError(
                f"engine emitted class id {det.label_id!r} absent from the model label map"
            )
        # Coerce to builtin float so no framework-native scalar (e.g. a numpy/torch
        # scalar) escapes the backend through RawDetection.
        coords = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if not all(math.isfinite(c) for c in coords):
            # Non-finite predictions are not clippable; leave them for the adapter to
            # reject as malformed rather than fabricating a finite box here.
            return RawDetection(label=label, score=float(det.score), box=coords)
        # Clip to the original image rectangle so coordinates satisfy the frozen
        # BoundingBox contract; drop a box that has no area left inside the frame.
        x1 = _clip(coords[0], 0.0, float(width))
        y1 = _clip(coords[1], 0.0, float(height))
        x2 = _clip(coords[2], 0.0, float(width))
        y2 = _clip(coords[3], 0.0, float(height))
        if x2 <= x1 or y2 <= y1:
            return None
        return RawDetection(label=label, score=float(det.score), box=(x1, y1, x2, y2))


# --- real transformers engine (the only torch/transformers-touching code) ----
class _TransformersRTDetrEngine:
    """RT-DETR inference on top of ``torch`` + HuggingFace ``transformers``.

    Constructed via :meth:`load`. Holds the model/processor/device handles privately;
    they never escape (only :class:`EngineDetection` values do). torch/transformers
    are imported lazily inside :meth:`load` / :meth:`infer` so importing this module
    never imports them.
    """

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: Any,
        id2label: dict[int, str],
        torch: Any,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._id2label = id2label
        self._torch = torch

    @classmethod
    def load(cls, config: RTDetrConfig) -> "_TransformersRTDetrEngine":
        """Load the model + processor for ``config.checkpoint`` (may acquire artifacts).

        Raises:
            BackendDependencyError: torch/transformers are not installed.
            InvalidDeviceError: CUDA was requested but is unavailable.
            ModelArtifactUnavailableError: the checkpoint cannot be resolved locally.
        """

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise BackendDependencyError(
                "the RT-DETR backend needs the optional 'rtdetr' dependencies "
                "(torch, transformers); install with: pip install 'trafficpulse[rtdetr]'"
            ) from exc

        device = _resolve_device(config.device, torch)
        try:
            processor = AutoImageProcessor.from_pretrained(
                config.checkpoint, local_files_only=config.local_files_only
            )
            model = AutoModelForObjectDetection.from_pretrained(
                config.checkpoint, local_files_only=config.local_files_only
            )
        except OSError as exc:
            raise ModelArtifactUnavailableError(
                f"RT-DETR checkpoint {config.checkpoint!r} is not available "
                f"(local_files_only={config.local_files_only}); provide a locally "
                "cached checkpoint or set local_files_only=False to allow download"
            ) from exc

        model.to(device)
        model.eval()
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        return cls(
            model=model, processor=processor, device=device, id2label=id2label, torch=torch
        )

    def label_name(self, label_id: int) -> str | None:
        return self._id2label.get(label_id)

    def infer(
        self, image: NDArray[np.uint8], *, threshold: float
    ) -> Sequence[EngineDetection]:
        torch = self._torch
        height, width = int(image.shape[0]), int(image.shape[1])
        try:
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([[height, width]], device=self._device)
            processed = self._processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )
        except (RuntimeError, ValueError) as exc:
            raise BackendInferenceError(f"RT-DETR inference failed: {exc}") from exc

        result = processed[0]
        scores = result["scores"].tolist()
        labels = result["labels"].tolist()
        boxes = result["boxes"].tolist()
        detections: list[EngineDetection] = []
        for score, label_id, box in zip(scores, labels, boxes, strict=True):
            x1, y1, x2, y2 = box
            detections.append(
                EngineDetection(
                    label_id=int(label_id),
                    score=float(score),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]`` (both finite)."""

    return max(low, min(high, value))


def _resolve_device(device: str, torch: Any) -> Any:
    """Resolve a validated device string into a framework device (never escapes)."""

    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise InvalidDeviceError(
            f"device {device!r} was requested but CUDA is not available in this environment"
        )
    return torch.device(device)

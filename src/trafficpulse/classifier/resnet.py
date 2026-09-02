"""The trained ResNet-50 helmet-state backend (P4-U5 winner), behind the P4-U2 seam.

This backend runs the **frozen P4-U5 ResNet-50 checkpoint** -- the model that won the
pre-registered CNN-vs-ViT comparison (``docs/cnn-vs-vit-results.md``, ADR-005) -- as a
:class:`~trafficpulse.classifier.interface.HelmetClassifier`. It satisfies the same
``classify(crops) -> Sequence[RawHelmetPrediction]`` contract as the zero-shot backend, so
selecting it is a configuration change and nothing downstream can tell the difference.

**It is not the production default and adopting it is not authorised by this module.**
ADR-005 decision 3 adopts no backend, and decision 4 gates adoption on evaluating the
candidate *on derived head crops* -- the input the runtime actually produces. This module
makes that evaluation possible; it does not pre-empt it.

What P4-U5 does and does not establish
--------------------------------------
P4-U5 measured **whole-motorcycle crops** cut from HELMET's own annotation. The runtime
supplies the **top 30% of an RT-DETR person box at full width**
(``observations.helmet.extract_head_region``). Those are different objects, not two views of
one distribution, so the experiment's test macro-F1 of 0.92881 says **nothing** about this
backend's runtime accuracy and must never be quoted as if it did.

Preprocessing: replicated deliberately, and where it cannot be, said so
----------------------------------------------------------------------
Preprocessing is *model-specific*, so it belongs here rather than in the caller -- the same
reason the zero-shot backend delegates to its own ``AutoProcessor``. The P4-U5 training
pipeline is reproduced exactly in the parts that matter:

* **centred square pad with black** ``(0, 0, 0)`` (``extract.py:square_pad_resize``);
  edge replication was rejected there because it invents texture the annotation cannot
  support, and the same argument holds here;
* **PIL BILINEAR** resize to **224** -- the interpolation the stored crops were actually
  built with. (``pretrained_cfg`` advertises ``bicubic``, and the eval transform requests
  it, but on an already-224 image that resize is a no-op, so bilinear is what the weights
  were fitted through.);
* ImageNet normalisation **mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)** --
  resolved from ``resnet50.a1_in1k``'s own ``pretrained_cfg`` and hard-coded here so the
  runtime needs no ``timm``.

**One difference is deliberately NOT replicated:** training crops made a **JPEG q95**
round-trip on their way to disk, and this backend classifies raw arrays. Re-encoding every
crop through JPEG to imitate a storage artifact would be inventing a preprocessing step to
chase a number, so it is left out and recorded instead. It is a real residual domain gap and
belongs in any honest reading of this backend's runtime scores.

Calibration
-----------
``temperature`` divides the logits before softmax. P4-U5 fitted it on the **validation**
split (seed 0: T = 2.2298, ECE 0.0355 -> 0.0065), so a *calibrated* posterior crosses the
seam rather than the raw over-confident one. It defaults to ``1.0`` (no scaling): a
temperature is a property of a specific fitted checkpoint, so it is configuration, never a
constant baked into code. Note the fitted value was measured on **whole-motorcycle** crops
and carries the same caveat as the accuracy figure.

Abstention
----------
A two-class softmax cannot express "I don't know", but the rule layer depends on
``uncertain`` to *bridge* a gap rather than contradict a run
(``rules.no_helmet`` step semantics). ``abstain_below`` gives the backend an explicit,
configured floor: when the winning probability does not clear it, the backend emits its
``uncertain`` label instead of forcing a binary call. The threshold is **configuration and
must be pre-committed**, never tuned against observed test outputs -- doing so would leak
the test split into the model's operating point.

Why torchvision and not timm
----------------------------
The checkpoint is a plain ResNet-50 ``state_dict`` (``conv1``/``bn1``/``layer1..4``/``fc``,
head ``[2, 2048]``), verified to load into :func:`torchvision.models.resnet50` with
``strict=True`` and no missing or unexpected keys. ``timm`` built it, but ``timm`` is not
needed to run it -- and ``pyproject.toml`` is explicit that the ``cnnvit`` extra is dev-time
research tooling that never ships in the wheel. ``torchvision`` is already required by the
``rtdetr`` extra, so this backend adds **no new runtime dependency**.

Turban: what this backend cannot say
------------------------------------
This is a **binary** classifier. It cannot emit ``turban``, and it declares that through
:attr:`~trafficpulse.classifier.interface.HelmetClassifier.supported_labels` so a
composition root can refuse a configuration whose exemption policy depends on a label that
can never arrive. See :mod:`trafficpulse.classifier.capabilities`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .crop import Crop
from .errors import HelmetClassifierError
from .interface import HelmetClassifier
from .raw import RawHelmetPrediction

_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")

#: Input resolution the P4-U5 checkpoints were trained and evaluated at (§12).
IMAGE_SIZE = 224

#: Constant square-pad colour, matching the experiment's extraction geometry.
PAD_RGB: tuple[int, int, int] = (0, 0, 0)

#: ImageNet normalisation, as resolved from ``resnet50.a1_in1k``'s ``pretrained_cfg``.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Native class labels, **index-aligned with the trained head**. P4-U5 fixed this order in
#: ``helmet_cnn_vit.datasets.CLASS_INDEX`` (``helmet`` -> 0, ``no_helmet`` -> 1); getting it
#: backwards would invert every prediction while still "working", so it is pinned here and
#: asserted by the tests rather than left implicit.
NATIVE_LABELS: tuple[str, ...] = ("helmet", "no_helmet")

#: The label emitted when no class clears ``abstain_below``. Maps to ``HelmetState.UNCERTAIN``
#: through the P4-U4 adapter's label map, which the rule layer treats as a bridged gap.
ABSTAIN_LABEL = "uncertain"


# --- backend error taxonomy --------------------------------------------------
class ResNetBackendError(HelmetClassifierError):
    """Base class for ResNet backend errors (a :class:`HelmetClassifierError`).

    Every backend failure surfaces as one of these stable TrafficPulse errors, so callers
    never see a raw ``torch`` exception cross the classifier boundary. Originating
    framework exceptions are chained as ``__cause__``.
    """


class ResNetDependencyError(ResNetBackendError):
    """The optional ``torch`` / ``torchvision`` dependencies are missing."""


class CheckpointUnavailableError(ResNetBackendError):
    """The configured checkpoint file does not exist or cannot be read."""


class MalformedCheckpointError(ResNetBackendError):
    """The checkpoint is not a P4-U5 helmet checkpoint of the expected shape."""


class ResNetInvalidDeviceError(ResNetBackendError):
    """A device was requested that this environment cannot provide (e.g. CUDA)."""


class ResNetMissingCropImageError(ResNetBackendError):
    """A real classifier was asked to run on a crop with no ``image`` payload."""


class MalformedResNetOutputError(ResNetBackendError):
    """The inference engine returned a structurally invalid probability matrix."""


class ResNetInferenceError(ResNetBackendError):
    """A framework-level failure occurred during inference or post-processing."""


# --- configuration -----------------------------------------------------------
class ResNetHelmetConfig(BaseModel):
    """Runtime configuration for the trained ResNet-50 helmet backend.

    Frozen and strict like the domain contracts, and exposing no framework-native object
    (no ``torch.device``): the device is a validated string.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: Path
    """Filesystem path to the P4-U5 ``best.pt``. No default: the operator chooses the
    artifact, whose provenance is a per-artifact review (ADR-001). Never downloaded."""

    device: str = "cpu"
    """``cpu``, ``cuda``, or ``cuda:N``."""

    temperature: float = Field(default=1.0, gt=0.0)
    """Logit temperature fitted on **validation** (P4-U5 seed 0: 2.2298). ``1.0`` disables
    scaling. Belongs to a specific checkpoint, so it is configuration, not a constant."""

    abstain_below: float | None = Field(default=None, ge=0.0, le=1.0)
    """Winning-probability floor below which the backend emits ``uncertain``. ``None``
    always forces a binary call. **Pre-commit this value**; tuning it on test outputs would
    leak the split into the operating point."""

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_RE.match(value):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N'; got {value!r}")
        return value

    @property
    def declared_labels(self) -> frozenset[str]:
        """Everything a backend built from this config can emit (see the classifier).

        Lives on the **config** so a composition root can ask what a backend *would*
        be able to say without constructing one -- constructing one loads torch and
        reads a checkpoint, which an application deciding which rules to offer must
        not have to do. :attr:`ResNetHelmetClassifier.supported_labels` returns exactly
        this, so there is one definition and the two cannot drift.

        ``turban`` is absent under every configuration: the P4-U5 model is binary.
        """

        labels = set(NATIVE_LABELS)
        if self.abstain_below is not None:
            labels.add(ABSTAIN_LABEL)
        return frozenset(labels)


# --- internal, framework-neutral inference seam ------------------------------
class ResNetInferenceEngine(Protocol):
    """Framework-neutral scoring seam (no tensor crosses it)."""

    def infer(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[Sequence[float]]:
        """Score every image; one row per image, one probability per class.

        Rows are in input order and hold one probability per entry of
        :data:`NATIVE_LABELS`, in that order, summing to ~1.0.
        """
        ...


def square_pad_resize(image: NDArray[np.uint8], *, size: int = IMAGE_SIZE) -> NDArray[np.uint8]:
    """Centre-pad to a square with :data:`PAD_RGB`, then BILINEAR-resize to ``size``.

    A NumPy/PIL reimplementation of the experiment's ``extract.square_pad_resize``, kept
    byte-faithful to it: same pad colour, same centring arithmetic (``(side - w) // 2``),
    same BILINEAR filter. It lives here rather than in the caller because it is part of
    *this model's* input contract, not a property of the crop.
    """

    try:
        from PIL import Image as PilImage
    except ImportError as exc:  # pragma: no cover - exercised via the dependency test
        raise ResNetDependencyError(
            "the ResNet helmet backend needs Pillow (the optional 'rtdetr' extra); "
            "install with: pip install 'trafficpulse[rtdetr]'"
        ) from exc

    if image.ndim != 3 or image.shape[2] != 3:
        raise MalformedResNetOutputError(
            f"expected an RGB (H, W, 3) uint8 crop; got shape {image.shape!r}"
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    if height == 0 or width == 0:
        raise MalformedResNetOutputError("cannot preprocess a zero-area crop")

    pil = PilImage.fromarray(image, mode="RGB")
    side = max(width, height)
    if width != height:
        canvas = PilImage.new("RGB", (side, side), PAD_RGB)
        canvas.paste(pil, ((side - width) // 2, (side - height) // 2))
        pil = canvas
    if pil.size != (size, size):
        pil = pil.resize((size, size), PilImage.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


class _TorchResNetEngine:
    """The real ``torch`` + ``torchvision`` engine. Constructed only by :meth:`load`."""

    def __init__(self, model: Any, torch_module: Any, device: Any, temperature: float) -> None:
        self._model = model
        self._torch = torch_module
        self._device = device
        self._temperature = temperature

    @classmethod
    def load(cls, config: ResNetHelmetConfig) -> _TorchResNetEngine:
        """Load the checkpoint into a torchvision ResNet-50. Lazy, and fail-fast.

        ``torch``/``torchvision`` are imported **here**, not at module import, so importing
        ``trafficpulse.classifier`` pulls in no ML framework.
        """

        try:
            import torch
            from torchvision.models import resnet50  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ResNetDependencyError(
                "the ResNet helmet backend needs the optional 'rtdetr' dependencies "
                "(torch, torchvision); install with: pip install 'trafficpulse[rtdetr]'"
            ) from exc

        path = config.checkpoint
        if not path.is_file():
            raise CheckpointUnavailableError(
                f"helmet checkpoint not found: {path}. The backend never downloads weights; "
                "point 'checkpoint' at a locally available P4-U5 best.pt."
            )

        try:
            device = torch.device(config.device)
        except (RuntimeError, AssertionError, ValueError) as exc:
            raise ResNetInvalidDeviceError(f"invalid device {config.device!r}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ResNetInvalidDeviceError(
                f"device {config.device!r} requested but CUDA is not available here"
            )

        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - any torch load failure is a config fact
            raise CheckpointUnavailableError(f"could not read checkpoint {path}: {exc}") from exc

        if not isinstance(payload, dict) or "model" not in payload:
            raise MalformedCheckpointError(
                f"checkpoint {path} has no 'model' state_dict; expected a P4-U5 payload "
                "of {'model': ..., 'epoch': ..., 'config': ...}"
            )

        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(NATIVE_LABELS))
        try:
            model.load_state_dict(payload["model"], strict=True)
        except (RuntimeError, KeyError) as exc:
            raise MalformedCheckpointError(
                f"checkpoint {path} is not a {len(NATIVE_LABELS)}-class ResNet-50 helmet "
                f"checkpoint: {exc}"
            ) from exc

        model.to(device)
        model.eval()
        return cls(model, torch, device, config.temperature)

    def infer(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[Sequence[float]]:
        """Preprocess, run the model, temperature-scale, and return plain floats."""

        torch = self._torch
        try:
            prepared = [square_pad_resize(image) for image in images]
            batch = np.stack(prepared).astype(np.float32) / 255.0
            mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
            std = np.asarray(IMAGENET_STD, dtype=np.float32)
            batch = (batch - mean) / std
            # (N, H, W, C) -> (N, C, H, W), the layout torchvision expects.
            tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
            tensor = tensor.to(self._device)
            with torch.inference_mode():
                logits = self._model(tensor).float()
                if self._temperature != 1.0:
                    logits = logits / self._temperature
                probabilities = torch.softmax(logits, dim=-1)
            rows = probabilities.detach().cpu().tolist()
        except ResNetBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - framework failure must not escape the seam
            raise ResNetInferenceError(f"ResNet helmet inference failed: {exc}") from exc
        return [[float(value) for value in row] for row in rows]


class ResNetHelmetClassifier(HelmetClassifier):
    """The trained P4-U5 ResNet-50 as a :class:`HelmetClassifier`.

    Construct with only a :class:`ResNetHelmetConfig` to load the real torch engine
    (fail-fast), or inject a :class:`ResNetInferenceEngine` for tests. The injected-engine
    path touches no ML dependency.
    """

    def __init__(
        self, config: ResNetHelmetConfig, *, engine: ResNetInferenceEngine | None = None
    ) -> None:
        self._config = config
        self._engine: ResNetInferenceEngine = (
            engine if engine is not None else _TorchResNetEngine.load(config)
        )

    @property
    def config(self) -> ResNetHelmetConfig:
        return self._config

    @property
    def supported_labels(self) -> frozenset[str]:
        """Everything this backend can emit -- and, by omission, what it cannot.

        ``turban`` is **absent**: the P4-U5 model is binary and no configuration of it can
        produce that label. Declaring the set is what lets a composition root refuse a
        turban-dependent policy loudly instead of running one that can never fire.
        """

        return self._config.declared_labels

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        """Classify each crop; one prediction per crop, in input order.

        Raises:
            ResNetMissingCropImageError: a crop carries no ``image``.
            MalformedResNetOutputError: the engine returned a structurally invalid matrix.
            ResNetInferenceError: a framework-level inference failure.
        """

        if not crops:
            return ()  # empty in, empty out: never touch the model (P4-U2 contract)

        images: list[NDArray[np.uint8]] = []
        for crop in crops:
            if crop.image is None:
                raise ResNetMissingCropImageError(
                    f"crop for track {crop.track_id!r} at frame {crop.frame_index} carries "
                    "no image; a real backend cannot classify an empty crop"
                )
            images.append(crop.image)

        rows = self._engine.infer(images)
        if len(rows) != len(crops):
            raise MalformedResNetOutputError(
                f"engine returned {len(rows)} rows for {len(crops)} crops"
            )

        predictions: list[RawHelmetPrediction] = []
        for row in rows:
            if len(row) != len(NATIVE_LABELS):
                raise MalformedResNetOutputError(
                    f"engine returned {len(row)} scores for {len(NATIVE_LABELS)} classes"
                )
            best_index = 0
            for index in range(1, len(row)):
                # Strict '>' so an exact tie keeps the earlier-declared label: deterministic.
                if row[index] > row[best_index]:
                    best_index = index
            score = row[best_index]
            if not np.isfinite(score):
                raise MalformedResNetOutputError(f"engine returned a non-finite score: {score!r}")
            score = min(1.0, max(0.0, float(score)))
            threshold = self._config.abstain_below
            if threshold is not None and score < threshold:
                # Abstain with the *real* winning probability. The score is not discarded
                # and not rounded: the rule layer is told how weak the call actually was.
                predictions.append(RawHelmetPrediction(label=ABSTAIN_LABEL, score=score))
            else:
                predictions.append(
                    RawHelmetPrediction(label=NATIVE_LABELS[best_index], score=score)
                )
        return tuple(predictions)

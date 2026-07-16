"""The first real helmet backend: zero-shot vision-language classification (P4-U3).

The smallest real inference path behind the frozen P4-U2 ``HelmetClassifier``
boundary. It scores each head crop against a set of natural-language prompts using
a CLIP-family model from the **Apache-2.0** HuggingFace ``transformers`` port, and
emits the best-matching prompt's native label. Nothing framework-native -- tensors,
processors, model handles, device objects -- escapes this module.

Why zero-shot, and not a fine-tuned classifier
----------------------------------------------
Because the alternative is not currently reachable, and pretending otherwise would
be dishonest. A fine-tuned helmet classifier needs a helmet dataset; the project's
only registered candidate (``registry/datasets/helmet-myanmar.yaml``) records
``licensing.status: unknown`` and ``local_acquisition_status: not_downloaded``, and
the dataset policy forbids acquisition before that gate resolves. A zero-shot model
needs **no dataset and no training**: it is the only real backend reachable under
the project's own rules today.

It is also not a detour. The pre-registered CNN-vs-ViT experiment
(architecture-review §12) remains the intended production backend and lands behind
**this same seam** with no change to any consumer -- which is precisely what P4-U2
exists to guarantee. This backend is the demonstrable floor, not the ceiling.

Licence posture (ADR-001 permissive-only)
-----------------------------------------
This module is written against ``AutoModel`` / ``AutoProcessor``, so it supports any
CLIP-family checkpoint without code change -- notably CLIP (MIT) and SigLIP
(Apache-2.0). **No AGPL model code is involved**: Ultralytics (hence YOLO11/YOLO12
and their trained weights) remains excluded from the integrated path by ADR-001.

This unit ships, blesses, and defaults to **no checkpoint**, mirroring P1-U7. A
permissive *code* licence does not grant rights to any particular weight file: per
ADR-001 the exact artifact's weight and pretraining-data terms are a **per-artifact
review** (U4 registry) before reliance or distribution. The operator names the
checkpoint; this module downloads nothing by default
(``local_files_only=True``).

Where the ML dependency lives
-----------------------------
``torch`` and ``transformers`` are imported **lazily**, only inside
:meth:`_TransformersZeroShotEngine.load` / :meth:`~_TransformersZeroShotEngine.infer`
-- the same discipline as ``detector/rtdetr.py``. Consequences, all tested:

* importing this module (or ``trafficpulse.classifier``) pulls in **no** ML
  framework -- the P4-U2 boundary invariant is preserved;
* constructing :class:`ZeroShotHelmetConfig` loads nothing and downloads nothing;
* the base install and every unit test stay ML-free and network-free.

No new dependency is added: the required packages are the existing optional
``rtdetr`` extra (torch, transformers, pillow), which this backend reuses.

Internal inference seam
-----------------------
:class:`ZeroShotInferenceEngine` is a small **framework-neutral** protocol
returning plain ``float`` scores -- never a tensor. :class:`_TransformersZeroShotEngine`
is the real implementation (the only code that touches torch/transformers). Because
the seam is framework-neutral, :class:`ZeroShotHelmetClassifier` is fully testable
with a fake engine and no ML dependency, and no framework object can leak through it.

Scores are relative, NOT calibrated probabilities
-------------------------------------------------
The engine softmaxes the image-text similarity logits **across the configured
prompt set**, so a score is "how well this prompt matched relative to the other
prompts offered" -- it is not a calibrated probability of the label being correct,
and must never be relabelled as one (architecture-review §13). Adding, removing, or
rewording a prompt changes every score. A near-tie surfaces as a score near
``1/len(prompts)``, which is the signal a downstream quality gate (P4-U4) uses to
route a crop to ``uncertain``; this backend applies **no** such gate itself and
emits no ``uncertain`` label of its own -- abstention policy is the rule layer's
(P4-U5), per the frozen ontology.

No accuracy is claimed for this backend by this unit. Zero-shot performance on
small, blurred CCTV head crops is unvalidated, and P4-U1 measured a median rider
head crop of ~30px on real footage.
"""

import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .crop import Crop
from .errors import HelmetClassifierError
from .interface import HelmetClassifier
from .raw import RawHelmetPrediction

_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


# --- backend error taxonomy --------------------------------------------------
class ZeroShotBackendError(HelmetClassifierError):
    """Base class for zero-shot backend errors (a :class:`HelmetClassifierError`).

    Every backend failure surfaces as one of these stable TrafficPulse errors, so
    callers never see a raw ``torch`` / ``transformers`` exception cross the
    classifier boundary. Originating framework exceptions are chained as
    ``__cause__``.
    """


class BackendDependencyError(ZeroShotBackendError):
    """The optional ``torch`` / ``transformers`` backend dependencies are missing."""


class ModelArtifactUnavailableError(ZeroShotBackendError):
    """The requested checkpoint is not available locally.

    Raised when ``from_pretrained`` cannot resolve the checkpoint (e.g.
    ``local_files_only=True`` and nothing is cached). The backend never downloads
    silently to recover.
    """


class InvalidDeviceError(ZeroShotBackendError):
    """A device was requested that this environment cannot provide (e.g. CUDA)."""


class MissingCropImageError(ZeroShotBackendError):
    """A real classifier was asked to run on a crop with no ``image`` payload."""


class MalformedBackendOutputError(ZeroShotBackendError):
    """The inference engine returned a structurally invalid score matrix.

    Wrong arity (not one score row per crop, or not one score per prompt), or a
    non-finite score. This is a *structural* check on the engine seam; the
    authoritative contract validation of the resulting observation remains the
    P4-U4 adapter's job.
    """


class BackendInferenceError(ZeroShotBackendError):
    """A framework-level failure occurred during inference or post-processing."""


# --- prompts -----------------------------------------------------------------
# A PROVISIONAL starting prompt set: native label -> natural-language prompt.
#
# These are untuned and carry no accuracy claim. Prompt wording is this backend's
# entire "training", so it is exactly the kind of parameter the project marks
# provisional until measured on held-out data. The keys are the backend's *native*
# vocabulary; mapping them onto the frozen four-label ontology is the P4-U4
# adapter's job (the keys merely happen to read like ontology ids, as the stub's do).
#
# Deliberately no "uncertain" prompt: uncertainty is not a visual class to match,
# it is the *absence* of a confident match, surfaced by a near-tie score and
# decided by the rule layer (D4: uncertain -> abstain). Prompting for it would let
# the model assert abstention as a positive finding.
DEFAULT_HELMET_PROMPTS: dict[str, str] = {
    "helmet": "a close-up photo of a motorcycle rider wearing a safety helmet",
    "no_helmet": "a close-up photo of a motorcycle rider with a bare head, not wearing a helmet",
    "turban": "a close-up photo of a motorcycle rider wearing a turban",
}


# --- configuration -----------------------------------------------------------
class ZeroShotHelmetConfig(BaseModel):
    """Backend-specific runtime configuration for the zero-shot helmet classifier.

    Deliberately separate from anything the seam or the adapter consumes: these
    fields are meaningless to another backend and must not leak into shared
    configuration. Frozen + strict like the domain contracts. Exposes **no**
    framework-native object (no ``torch.device``): the device is a validated string.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: str
    """CLIP-family checkpoint identity: a HuggingFace model id or a local directory.
    No default -- the operator chooses the artifact, whose weight/pretraining-data
    provenance is reviewed per ADR-001 (U4 registry)."""

    prompts: dict[str, str] = DEFAULT_HELMET_PROMPTS
    """Native label -> prompt text. At least two are required: a zero-shot score is
    meaningful only *relative* to competing prompts, so a single-prompt set would
    softmax to a constant 1.0 and assert its label unconditionally."""

    device: str = "cpu"
    """Execution device: ``"cpu"`` (default), ``"cuda"``, or ``"cuda:N"``. CUDA is
    an explicit opt-in and is only honoured if the backend reports it available."""

    local_files_only: bool = True
    """Offline by default: load only already-local/cached artifacts, never download.
    Set ``False`` to allow ``transformers`` to fetch the checkpoint (explicit)."""

    @field_validator("checkpoint")
    @classmethod
    def _non_empty_checkpoint(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checkpoint must be a non-empty model id or local path")
        return value

    @field_validator("prompts")
    @classmethod
    def _validate_prompts(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) < 2:
            raise ValueError(
                "prompts must offer at least two competing labels: a zero-shot score "
                "is relative to the prompt set, so a single prompt always scores 1.0"
            )
        if any(not label.strip() for label in value):
            raise ValueError("prompt labels (the native vocabulary) must be non-empty")
        if any(not text.strip() for text in value.values()):
            raise ValueError("prompt texts must be non-empty")
        return value

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not _DEVICE_RE.match(value):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N', got {value!r}")
        return value


# --- internal, framework-neutral inference seam ------------------------------
class ZeroShotInferenceEngine(Protocol):
    """Framework-neutral zero-shot scoring seam (no tensor crosses it)."""

    def infer(
        self, images: Sequence[NDArray[np.uint8]], prompts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """Score every image against every prompt.

        Returns one row per image, each holding one score per prompt in prompt
        order, as plain floats summing to ~1.0 across the row.
        """
        ...


class ZeroShotHelmetClassifier(HelmetClassifier):
    """A real zero-shot :class:`~trafficpulse.classifier.interface.HelmetClassifier`.

    Satisfies the exact P4-U2 interface: ``classify(crops) -> Sequence[RawHelmetPrediction]``.
    Construct with only a :class:`ZeroShotHelmetConfig` to load the real transformers
    engine (fail-fast), or inject a :class:`ZeroShotInferenceEngine` (tests / advanced
    embedding). The injected-engine path touches no ML dependency.
    """

    def __init__(
        self, config: ZeroShotHelmetConfig, *, engine: ZeroShotInferenceEngine | None = None
    ) -> None:
        self._config = config
        # Frozen prompt order: dict preserves insertion order, so the label list and
        # the prompt list stay index-aligned with the engine's score columns, and an
        # argmax tie always resolves to the first-declared label (deterministic).
        self._labels: tuple[str, ...] = tuple(config.prompts)
        self._prompts: tuple[str, ...] = tuple(config.prompts[label] for label in self._labels)
        self._engine: ZeroShotInferenceEngine = (
            engine if engine is not None else _TransformersZeroShotEngine.load(config)
        )

    @property
    def config(self) -> ZeroShotHelmetConfig:
        return self._config

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        """Score each crop against the prompt set; return the best match per crop.

        Raises:
            MissingCropImageError: a crop carries no ``image``.
            MalformedBackendOutputError: the engine returned a structurally invalid
                score matrix.
            BackendInferenceError: a framework-level inference failure.
        """

        if not crops:
            return ()  # empty in, empty out: never touch the model (P4-U2 contract)

        images: list[NDArray[np.uint8]] = []
        for crop in crops:
            if crop.image is None:
                raise MissingCropImageError(
                    "the zero-shot backend requires pixels, but crop.image is None "
                    f"(camera_id={crop.camera_id!r}, frame_index={crop.frame_index}, "
                    f"track_id={crop.track_id!r})"
                )
            images.append(crop.image)

        rows = self._engine.infer(images, self._prompts)
        if len(rows) != len(crops):
            raise MalformedBackendOutputError(
                f"engine returned {len(rows)} score rows for {len(crops)} crops; "
                "the seam requires exactly one prediction per crop, in input order"
            )
        return tuple(self._best(row) for row in rows)

    def _best(self, row: Sequence[float]) -> RawHelmetPrediction:
        if len(row) != len(self._labels):
            raise MalformedBackendOutputError(
                f"engine returned {len(row)} scores for {len(self._labels)} prompts"
            )
        best_index = -1
        best_score = float("-inf")
        for index, score in enumerate(row):
            value = float(score)
            if not np.isfinite(value):
                raise MalformedBackendOutputError(
                    f"engine returned a non-finite score ({value!r}) for prompt "
                    f"{self._labels[index]!r}"
                )
            if value > best_score:  # strict >: an exact tie keeps the earlier label
                best_score = value
                best_index = index
        return RawHelmetPrediction(
            label=self._labels[best_index],
            # Clamp: softmax is mathematically in [0, 1] but float error can emit
            # 1.0000000001, which the P4-U4 adapter's Confidence bound would reject.
            score=max(0.0, min(1.0, best_score)),
        )


# --- real transformers engine (the only torch/transformers-touching code) ----
class _TransformersZeroShotEngine:
    """CLIP-family zero-shot scoring on ``torch`` + HuggingFace ``transformers``.

    Constructed via :meth:`load`. Holds the model/processor/device handles privately;
    they never escape (only plain floats do). torch/transformers are imported lazily
    inside :meth:`load` / :meth:`infer` so importing this module never imports them.

    Works with any checkpoint exposing the CLIP-family image-text contract
    (``logits_per_image``) through ``AutoModel``/``AutoProcessor`` -- CLIP and SigLIP
    both do -- so swapping the model is configuration, not code.
    """

    def __init__(self, *, model: Any, processor: Any, device: Any, torch: Any) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._torch = torch

    @classmethod
    def load(cls, config: ZeroShotHelmetConfig) -> "_TransformersZeroShotEngine":
        """Load the model + processor for ``config.checkpoint`` (may acquire artifacts).

        Raises:
            BackendDependencyError: torch/transformers are not installed.
            InvalidDeviceError: CUDA was requested but is unavailable.
            ModelArtifactUnavailableError: the checkpoint cannot be resolved locally.
        """

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise BackendDependencyError(
                "the zero-shot helmet backend needs the optional 'rtdetr' "
                "dependencies (torch, transformers, pillow); install with: "
                "pip install 'trafficpulse[rtdetr]'"
            ) from exc

        device = _resolve_device(config.device, torch)
        try:
            processor = AutoProcessor.from_pretrained(
                config.checkpoint, local_files_only=config.local_files_only
            )
            model = AutoModel.from_pretrained(
                config.checkpoint, local_files_only=config.local_files_only
            )
        except OSError as exc:
            raise ModelArtifactUnavailableError(
                f"checkpoint {config.checkpoint!r} is not available "
                f"(local_files_only={config.local_files_only}); provide a locally "
                "cached checkpoint or set local_files_only=False to allow download. "
                "Weight provenance is reviewed per artifact (ADR-001, U4 registry)."
            ) from exc

        model.to(device)
        model.eval()
        return cls(model=model, processor=processor, device=device, torch=torch)

    def infer(
        self, images: Sequence[NDArray[np.uint8]], prompts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """One batched forward pass; return per-image softmax over the prompts."""

        torch = self._torch
        try:
            inputs = self._processor(
                text=list(prompts),
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Relative, not calibrated: softmax across the prompt set (see docstring).
            probabilities = outputs.logits_per_image.softmax(dim=-1)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise BackendInferenceError(f"zero-shot inference failed: {exc}") from exc

        # .tolist() is the tensor->plain-float boundary: nothing framework-native
        # travels past this return.
        rows = probabilities.tolist()
        return [[float(value) for value in row] for row in rows]


def _resolve_device(device: str, torch: Any) -> Any:
    """Resolve a validated device string into a framework device (never escapes)."""

    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise InvalidDeviceError(
            f"device {device!r} was requested but CUDA is not available in this environment"
        )
    return torch.device(device)

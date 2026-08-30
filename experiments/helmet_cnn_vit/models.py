"""The two model families under comparison (P4-U5).

Architecture-review §12 locks the pair: **ResNet-50** (CNN) versus **DeiT-Small**
(ViT), both ImageNet-1k pretrained, both from ``timm``, both at 224x224.

Why one factory for both
------------------------
Every difference between the two runs must be the architecture. Both models are
built by the same :func:`create_model` call, trained by the same loop, fed by the
same :mod:`helmet_cnn_vit.datasets` pipeline, and evaluated by the same
:mod:`helmet_cnn_vit.metrics` core. ``timm`` is what makes that possible: it
exposes both families behind one ``create_model``/``num_classes`` interface, so
the head replacement, the optimiser, and the schedule are literally shared code
rather than two implementations that are merely intended to match.

Normalisation is the one deliberate per-family difference: each checkpoint is
evaluated under the mean/std it was pretrained with, resolved from the model's own
``pretrained_cfg``. Forcing a shared normalisation would feed at least one model
inputs it was never trained for, which would be a handicap disguised as fairness.

``timm`` and ``torch`` are imported lazily, so this module is importable -- and its
specs are testable -- in a CI environment that has neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from helmet_rtdetr.models import _Model

from .errors import CnnVitError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

#: Number of output classes: helmet / no_helmet (see :mod:`helmet_cnn_vit.labels`).
NUM_CLASSES = 2

#: The input resolution both families are trained and evaluated at (§12).
IMAGE_SIZE = 224


class BackendUnavailableError(CnnVitError):
    """``timm``/``torch`` are not installed (the optional ``cnnvit`` extra)."""


class ModelSpec(_Model):
    """One competitor in the comparison."""

    name: str
    #: The exact pretrained checkpoint, pinned by tag so a re-run gets the same weights.
    timm_id: str
    family: Literal["cnn", "vit"]
    licence: str
    #: Approximate parameter count, for the report's cost table. The measured value
    #: is recorded at run time; this is only a sanity check.
    approx_params_m: float


#: The locked §12 pair. Adding a third entry here would be an ablation, not a
#: change to the mandated comparison.
MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet50": ModelSpec(
        name="resnet50",
        timm_id="resnet50.a1_in1k",
        family="cnn",
        licence="Apache-2.0",
        approx_params_m=25.6,
    ),
    "deit_small": ModelSpec(
        name="deit_small",
        timm_id="deit_small_patch16_224.fb_in1k",
        family="vit",
        licence="Apache-2.0",
        approx_params_m=22.1,
    ),
}


def require_backend() -> None:
    """Raise a typed error if the optional training backend is absent.

    Mirrors :func:`helmet_rtdetr.rtdetr.model.require_torch` -- an absent optional
    dependency is a configuration fact, not an ``ImportError`` traceback.
    """

    import importlib.util

    missing = [name for name in ("torch", "timm") if importlib.util.find_spec(name) is None]
    if missing:
        raise BackendUnavailableError(
            f"the CNN-vs-ViT experiment needs {missing}; install the 'cnnvit' extra "
            f"(pip install timm, plus a CUDA torch build) in the experiment venv"
        )


def spec_for(name: str) -> ModelSpec:
    """Look up a competitor by short name, failing loudly on a typo."""

    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise CnnVitError(
            f"unknown model {name!r}; the §12 pair is {sorted(MODEL_SPECS)}"
        ) from None


def create_model(
    spec: ModelSpec, *, pretrained: bool = True, num_classes: int = NUM_CLASSES
) -> torch.nn.Module:
    """Build one competitor with a fresh ``num_classes`` head.

    ``pretrained=False`` builds the same architecture with random weights, which is
    what the tests use so they never touch the network.
    """

    require_backend()
    import timm

    return timm.create_model(spec.timm_id, pretrained=pretrained, num_classes=num_classes)


def normalisation_for(model: torch.nn.Module) -> dict[str, Any]:
    """The mean/std/interpolation the checkpoint was pretrained with."""

    require_backend()
    import timm

    return dict(timm.data.resolve_data_config({}, model=model))


def parameter_count(model: torch.nn.Module) -> int:
    """Total trainable parameters -- part of the §12 cost table."""

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

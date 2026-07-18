"""Centralised deterministic seeding (H4A).

One base seed fans out into stable per-component seeds via SHA-256 (a
process-independent hash — never Python's salted ``hash()``), so components are
decorrelated but the whole plan is reproducible from the single configured seed.

Torch is **designed for, not executed**: this unit is ML-free, so
:func:`apply_seed_plan` seeds ``random`` and NumPy now and returns the torch seed
as *deferred* — the H4B training loop applies ``torch.manual_seed(plan.torch_seed)``
(and cuDNN determinism per ``plan.cudnn_deterministic``) when torch actually
loads. NumPy is imported inside the function so importing this module stays
dependency-light.
"""

from __future__ import annotations

import hashlib
import random

from pydantic import Field

from ..models import _Model


def _component_seed(base_seed: int, component: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{component}".encode()).digest()
    return int.from_bytes(digest[:4], "big")  # 32-bit, valid for every consumer


class SeedPlan(_Model):
    """The derived per-component seeds for one experiment."""

    base_seed: int = Field(ge=0)
    python_seed: int = Field(ge=0)
    numpy_seed: int = Field(ge=0)
    torch_seed: int = Field(ge=0)
    cudnn_deterministic: bool = True


class AppliedSeeds(_Model):
    """What :func:`apply_seed_plan` actually did vs deliberately deferred."""

    plan: SeedPlan
    applied: tuple[str, ...]
    deferred: tuple[str, ...]


def derive_seed_plan(base_seed: int) -> SeedPlan:
    """Derive the deterministic per-component seed plan for ``base_seed``."""

    return SeedPlan(
        base_seed=base_seed,
        python_seed=_component_seed(base_seed, "python"),
        numpy_seed=_component_seed(base_seed, "numpy"),
        torch_seed=_component_seed(base_seed, "torch"),
    )


def apply_seed_plan(plan: SeedPlan) -> AppliedSeeds:
    """Seed ``random`` and NumPy now; report torch as deferred (see module docs)."""

    random.seed(plan.python_seed)

    import numpy  # local import: keeps module import dependency-light (base dep)

    numpy.random.seed(plan.numpy_seed)
    return AppliedSeeds(
        plan=plan,
        applied=("python", "numpy"),
        deferred=("torch", "torch.cuda"),
    )

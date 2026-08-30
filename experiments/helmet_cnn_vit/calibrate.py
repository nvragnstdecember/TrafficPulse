"""Temperature scaling for the CNN-vs-ViT comparison (P4-U5).

Architecture-review §12 requires calibration to be reported "via ECE + reliability
diagrams, temperature scaling on validation reported pre/post". This module fits
the single temperature parameter; :mod:`helmet_cnn_vit.metrics` computes the ECE
either side of it.

Calibration is not decoration here. The runtime's `no_helmet` rule aggregates
per-frame classifier confidences over a track, so a systematically over-confident
classifier does not merely look wrong on a reliability diagram -- it shifts the
temporal aggregator's decision threshold. That is the "result system consequences"
§12 cites as part of why this task was chosen for the mandatory experiment.

Why the fit runs on stored probabilities
----------------------------------------
Each run stores the model's `no_helmet` probability per crop rather than raw
logits. For a two-class softmax those are equivalent: the probability is
``sigmoid(z)`` where ``z`` is the logit *difference*, so ``z = log(p / (1 - p))``
recovers everything temperature scaling needs. Probabilities are clipped away from
0 and 1 before the log, because a saturated fp16 probability of exactly 1.0 would
otherwise produce an infinite logit and poison the optimisation.

**The temperature is fitted on validation only** and then applied unchanged to
test, per the dataset-policy rule that tuning never touches the test split.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from helmet_rtdetr.models import _Model

from .errors import EmptyEvaluationError, MismatchedPredictionsError
from .metrics import POSITIVE_CLASS

#: Probabilities are clipped to this distance from 0 and 1 before the logit.
_EPS = 1e-6

#: Search bounds for the temperature. T > 1 softens, T < 1 sharpens.
_MIN_T = 0.05
_MAX_T = 10.0


class TemperatureFit(_Model):
    """The fitted temperature and the negative log-likelihood either side of it."""

    temperature: float
    nll_before: float
    nll_after: float

    @property
    def improved(self) -> bool:
        """Whether scaling actually reduced the validation NLL."""

        return self.nll_after < self.nll_before


def to_logits(scores: Sequence[float]) -> np.ndarray:
    """Recover the binary logit difference from stored `no_helmet` probabilities."""

    p = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Rescale probabilities by ``temperature`` (a no-op at ``T = 1``)."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise MismatchedPredictionsError(f"temperature must be positive, got {temperature!r}")
    scaled = 1.0 / (1.0 + np.exp(-to_logits(scores) / temperature))
    return tuple(float(v) for v in scaled)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Binary cross-entropy of the temperature-scaled logits, in nats per sample."""

    z = logits / temperature
    # log(1 + exp(-|z|)) + max(-z, 0) is the numerically stable softplus form.
    log_p = -(np.logaddexp(0.0, -z))
    log_1mp = -(np.logaddexp(0.0, z))
    return float(-np.mean(targets * log_p + (1.0 - targets) * log_1mp))


def fit_temperature(
    truth: Sequence[str], scores: Sequence[float], *, steps: int = 200
) -> TemperatureFit:
    """Fit a single temperature on the **validation** split by golden-section search.

    A one-dimensional, unimodal, cheap objective: a deterministic bracketed search
    is used rather than gradient descent so the fit is reproducible to the last bit
    and carries no optimiser state or learning rate of its own.
    """

    if len(truth) != len(scores):
        raise MismatchedPredictionsError(f"{len(truth)} labels but {len(scores)} scores")
    if not truth:
        raise EmptyEvaluationError("cannot calibrate on an empty split")

    logits = to_logits(scores)
    targets = np.array([1.0 if t == POSITIVE_CLASS else 0.0 for t in truth], dtype=np.float64)

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = _MIN_T, _MAX_T
    c = high - invphi * (high - low)
    d = low + invphi * (high - low)
    for _ in range(steps):
        if _nll(logits, targets, c) < _nll(logits, targets, d):
            high = d
        else:
            low = c
        c = high - invphi * (high - low)
        d = low + invphi * (high - low)
        if abs(high - low) < 1e-6:
            break

    temperature = (low + high) / 2.0
    return TemperatureFit(
        temperature=temperature,
        nll_before=_nll(logits, targets, 1.0),
        nll_after=_nll(logits, targets, temperature),
    )

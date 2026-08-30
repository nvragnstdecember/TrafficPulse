"""The pre-committed decision rule for the CNN-vs-ViT comparison (P4-U5).

Architecture-review §12 fixes, before any training, what counts as a difference:

    a difference is claimed only if it is **sign-consistent across all three seeds**
    **and** a pooled bootstrap 95% CI on ΔmacroF1 excludes zero -- otherwise it is
    reported as a tie, interpreted through the accuracy/latency/VRAM tradeoff.

That rule is implemented here as :func:`decide`, so the verdict is computed from
the numbers rather than chosen while looking at them. The functions are pure and
seeded; nothing here imports torch.

Why McNemar
-----------
Both models are evaluated on the *same* test crops, so their errors are paired. An
unpaired test would discard that pairing and lose power. McNemar's exact test looks
only at the discordant pairs -- crops one model gets right and the other gets wrong
-- which is exactly the evidence that distinguishes them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from helmet_rtdetr.models import _Model

from .errors import EmptyEvaluationError, MismatchedPredictionsError
from .metrics import CLASS_ORDER

#: Bootstrap resamples for the CI on ΔmacroF1 (§12 pre-registration).
DEFAULT_RESAMPLES = 10_000

#: The confidence level the claim rule uses.
DEFAULT_CONFIDENCE = 0.95


class McNemarResult(_Model):
    """Exact paired test over the discordant crops."""

    #: Crops the first model got right and the second got wrong.
    only_first_correct: int
    #: Crops the second model got right and the first got wrong.
    only_second_correct: int
    both_correct: int
    both_wrong: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_first_correct + self.only_second_correct


class BootstrapInterval(_Model):
    """A percentile bootstrap interval on a paired difference."""

    observed: float
    lower: float
    upper: float
    confidence: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


class ComparisonVerdict(_Model):
    """The pre-committed verdict: a claimed difference, or an honest tie."""

    #: ``True`` only when both §12 conditions hold.
    difference_claimed: bool
    winner: str | None
    sign_consistent: bool
    per_seed_delta: tuple[float, ...]
    interval: BootstrapInterval
    mcnemar: tuple[McNemarResult, ...]
    rationale: str


def mcnemar(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> McNemarResult:
    """Exact (binomial) McNemar test over paired per-crop outcomes.

    The exact form is used rather than the chi-squared approximation because the
    discordant count can be small on per-site and per-corruption slices, where the
    approximation is unreliable.
    """

    if len(first_correct) != len(second_correct):
        raise MismatchedPredictionsError(
            f"{len(first_correct)} vs {len(second_correct)} paired outcomes"
        )
    if not first_correct:
        raise EmptyEvaluationError("cannot run a paired test over an empty set")

    both = only_first = only_second = neither = 0
    for a, b in zip(first_correct, second_correct, strict=True):
        if a and b:
            both += 1
        elif a:
            only_first += 1
        elif b:
            only_second += 1
        else:
            neither += 1

    n = only_first + only_second
    if n == 0:
        # The models are indistinguishable on every crop: no evidence either way.
        p_value = 1.0
    else:
        k = min(only_first, only_second)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
        p_value = min(1.0, 2.0 * tail)
    return McNemarResult(
        only_first_correct=only_first,
        only_second_correct=only_second,
        both_correct=both,
        both_wrong=neither,
        p_value=p_value,
    )


def _macro_f1_from_indices(
    truth: np.ndarray, predicted: np.ndarray, indices: np.ndarray
) -> float:
    """macro-F1 over a resampled index set, vectorised for the bootstrap loop.

    Classes absent from a resample are skipped, matching
    :func:`~helmet_cnn_vit.metrics.compute_metrics`'s ``None`` convention.
    """

    t = truth[indices]
    p = predicted[indices]
    scores = []
    for label in range(len(CLASS_ORDER)):
        support = int(np.count_nonzero(t == label))
        if support == 0:
            continue
        true_positive = int(np.count_nonzero((t == label) & (p == label)))
        predicted_count = int(np.count_nonzero(p == label))
        recall = true_positive / support
        precision = (true_positive / predicted_count) if predicted_count else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
    return float(sum(scores) / len(scores)) if scores else 0.0


def bootstrap_delta_macro_f1(
    truth: Sequence[str],
    first_predicted: Sequence[str],
    second_predicted: Sequence[str],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap CI on ``macroF1(first) - macroF1(second)``.

    Crops are resampled **as pairs**, so each resample scores both models on the
    same crops and the interval reflects the paired difference rather than the sum
    of two independent sampling errors. The RNG is seeded, so the interval is a
    reproducible function of its inputs.
    """

    if not (len(truth) == len(first_predicted) == len(second_predicted)):
        raise MismatchedPredictionsError("truth and both prediction sequences must align")
    if not truth:
        raise EmptyEvaluationError("cannot bootstrap an empty evaluation set")

    encode = {label: index for index, label in enumerate(CLASS_ORDER)}
    unknown = sorted((set(truth) | set(first_predicted) | set(second_predicted)) - set(encode))
    if unknown:
        raise MismatchedPredictionsError(f"labels outside the class space: {unknown}")

    t = np.array([encode[x] for x in truth], dtype=np.int8)
    a = np.array([encode[x] for x in first_predicted], dtype=np.int8)
    b = np.array([encode[x] for x in second_predicted], dtype=np.int8)

    total = len(t)
    every = np.arange(total)
    observed = _macro_f1_from_indices(t, a, every) - _macro_f1_from_indices(t, b, every)

    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        indices = rng.integers(0, total, size=total)
        deltas[i] = _macro_f1_from_indices(t, a, indices) - _macro_f1_from_indices(
            t, b, indices
        )

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(deltas, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        observed=float(observed),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        resamples=resamples,
    )


def bootstrap_delta_macro_f1_pooled(
    truth: Sequence[str],
    first_predicted: Sequence[Sequence[str]],
    second_predicted: Sequence[Sequence[str]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> BootstrapInterval:
    """Pooled paired bootstrap on ΔmacroF1 across all seeds (the §12 form).

    §12 asks for a *pooled* interval. Bootstrapping a single seed would describe
    only that seed's run and would understate the uncertainty, because seed-to-seed
    variation is part of what the claim has to survive. So within each resample of
    crops, each family's macro-F1 is averaged over its seeds and the difference of
    those two averages is recorded. The interval therefore reflects both sources of
    variation: which crops were sampled, and which seed was drawn.

    Every seed is scored on the *same* resampled crops, preserving the pairing that
    makes this a paired test.
    """

    if not truth:
        raise EmptyEvaluationError("cannot bootstrap an empty evaluation set")
    if not first_predicted or not second_predicted:
        raise MismatchedPredictionsError("each family needs at least one seed's predictions")
    for predictions in (*first_predicted, *second_predicted):
        if len(predictions) != len(truth):
            raise MismatchedPredictionsError("every seed must predict the whole test split")

    encode = {label: index for index, label in enumerate(CLASS_ORDER)}
    unknown = sorted(
        (set(truth) | {p for row in first_predicted for p in row}
         | {p for row in second_predicted for p in row})
        - set(encode)
    )
    if unknown:
        raise MismatchedPredictionsError(f"labels outside the class space: {unknown}")

    t = np.array([encode[x] for x in truth], dtype=np.int8)
    a_seeds = [np.array([encode[x] for x in row], dtype=np.int8) for row in first_predicted]
    b_seeds = [np.array([encode[x] for x in row], dtype=np.int8) for row in second_predicted]

    total = len(t)
    every = np.arange(total)

    def pooled_delta(indices: np.ndarray) -> float:
        a = sum(_macro_f1_from_indices(t, seed_pred, indices) for seed_pred in a_seeds)
        b = sum(_macro_f1_from_indices(t, seed_pred, indices) for seed_pred in b_seeds)
        return a / len(a_seeds) - b / len(b_seeds)

    observed = pooled_delta(every)
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        deltas[i] = pooled_delta(rng.integers(0, total, size=total))

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(deltas, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        observed=float(observed),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        resamples=resamples,
    )


def decide(
    *,
    first_name: str,
    second_name: str,
    per_seed_delta: Sequence[float],
    interval: BootstrapInterval,
    mcnemar_results: Sequence[McNemarResult] = (),
) -> ComparisonVerdict:
    """Apply the §12 claim rule. ``per_seed_delta`` is macroF1(first) - macroF1(second).

    A difference is claimed only when every seed agrees on the sign **and** the
    bootstrap interval excludes zero. Anything else is a tie -- including the case
    where the interval excludes zero but the seeds disagree, which is precisely the
    situation the sign-consistency clause exists to catch.
    """

    if not per_seed_delta:
        raise EmptyEvaluationError("no per-seed deltas to decide on")

    positive = all(d > 0 for d in per_seed_delta)
    negative = all(d < 0 for d in per_seed_delta)
    sign_consistent = positive or negative
    claimed = sign_consistent and interval.excludes_zero

    if claimed:
        winner = first_name if positive else second_name
        rationale = (
            f"macro-F1 differs by {interval.observed:+.4f} "
            f"(95% CI [{interval.lower:+.4f}, {interval.upper:+.4f}], excludes zero) and all "
            f"{len(per_seed_delta)} seeds agree on the sign, so the difference is claimed "
            f"for {winner}."
        )
    else:
        winner = None
        reasons = []
        if not sign_consistent:
            reasons.append("the per-seed deltas do not agree on a sign")
        if not interval.excludes_zero:
            reasons.append(
                f"the 95% CI [{interval.lower:+.4f}, {interval.upper:+.4f}] includes zero"
            )
        rationale = (
            f"Reported as a TIE because {' and '.join(reasons)}. Per the pre-registered "
            f"rule this is interpreted through the accuracy/latency/VRAM tradeoff, not as "
            f"evidence that {first_name} and {second_name} are identical."
        )

    return ComparisonVerdict(
        difference_claimed=claimed,
        winner=winner,
        sign_consistent=sign_consistent,
        per_seed_delta=tuple(per_seed_delta),
        interval=interval,
        mcnemar=tuple(mcnemar_results),
        rationale=rationale,
    )

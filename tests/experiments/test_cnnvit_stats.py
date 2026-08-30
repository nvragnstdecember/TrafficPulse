"""The pre-committed §12 decision rule (P4-U5).

This module decides what the project is allowed to claim at viva, so its tests are
about the rule's *refusals* as much as its verdicts -- above all the case where the
bootstrap interval excludes zero but the seeds disagree, which must still be
reported as a tie.
"""

from __future__ import annotations

import math

import pytest
from helmet_cnn_vit.errors import EmptyEvaluationError, MismatchedPredictionsError
from helmet_cnn_vit.metrics import CLASS_ORDER
from helmet_cnn_vit.stats import (
    BootstrapInterval,
    bootstrap_delta_macro_f1,
    bootstrap_delta_macro_f1_pooled,
    decide,
    mcnemar,
)

HELMET, NO_HELMET = CLASS_ORDER


def interval(lower: float, upper: float, observed: float = 0.0) -> BootstrapInterval:
    return BootstrapInterval(
        observed=observed, lower=lower, upper=upper, confidence=0.95, resamples=10
    )


# --- McNemar -----------------------------------------------------------------------


def test_mcnemar_counts_the_four_paired_cells() -> None:
    first = [True, True, False, False, True]
    second = [True, False, True, False, False]
    result = mcnemar(first, second)
    assert result.both_correct == 1
    assert result.only_first_correct == 2
    assert result.only_second_correct == 1
    assert result.both_wrong == 1
    assert result.discordant == 3


def test_mcnemar_matches_the_exact_binomial_by_hand() -> None:
    """10 vs 2 discordant: p = 2 * sum(C(12,i), i<=2) / 2^12 = 2 * 79 / 4096."""

    first = [True] * 10 + [False] * 2
    second = [False] * 10 + [True] * 2
    result = mcnemar(first, second)
    assert result.discordant == 12
    assert result.p_value == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_mcnemar_is_symmetric_in_its_arguments() -> None:
    first = [True] * 7 + [False] * 3
    second = [False] * 7 + [True] * 3
    assert mcnemar(first, second).p_value == pytest.approx(mcnemar(second, first).p_value)


def test_identical_models_yield_no_evidence() -> None:
    outcomes = [True, False, True, True]
    result = mcnemar(outcomes, outcomes)
    assert result.discordant == 0
    assert result.p_value == 1.0


def test_a_large_one_sided_discordance_is_significant() -> None:
    result = mcnemar([True] * 20 + [False] * 0, [False] * 20)
    assert result.p_value < 0.001


def test_mcnemar_refuses_unpaired_or_empty_input() -> None:
    with pytest.raises(MismatchedPredictionsError):
        mcnemar([True], [True, False])
    with pytest.raises(EmptyEvaluationError):
        mcnemar([], [])


# --- bootstrap -------------------------------------------------------------------------


def test_the_bootstrap_is_seeded_and_reproducible() -> None:
    truth = [HELMET, NO_HELMET] * 25
    first = [HELMET] * 50
    second = [HELMET, NO_HELMET] * 25
    a = bootstrap_delta_macro_f1(truth, first, second, resamples=200, seed=7)
    b = bootstrap_delta_macro_f1(truth, first, second, resamples=200, seed=7)
    assert (a.lower, a.upper, a.observed) == (b.lower, b.upper, b.observed)


def test_two_identical_models_have_a_zero_delta_and_an_interval_containing_zero() -> None:
    truth = [HELMET, NO_HELMET] * 25
    predicted = [HELMET, HELMET, NO_HELMET, NO_HELMET] * 12 + [HELMET, NO_HELMET]
    result = bootstrap_delta_macro_f1(truth, predicted, predicted, resamples=200, seed=1)
    assert result.observed == pytest.approx(0.0)
    assert result.lower == pytest.approx(0.0)
    assert result.upper == pytest.approx(0.0)
    assert not result.excludes_zero


def test_a_clearly_better_model_produces_a_positive_interval_excluding_zero() -> None:
    truth = [HELMET] * 50 + [NO_HELMET] * 50
    perfect = list(truth)
    majority = [HELMET] * 100  # macro-F1 ~0.33 against a perfect 1.0
    result = bootstrap_delta_macro_f1(truth, perfect, majority, resamples=400, seed=3)
    assert result.observed > 0.5
    assert result.lower > 0.0
    assert result.excludes_zero


def test_the_delta_sign_follows_the_argument_order() -> None:
    truth = [HELMET] * 50 + [NO_HELMET] * 50
    perfect = list(truth)
    majority = [HELMET] * 100
    forward = bootstrap_delta_macro_f1(truth, perfect, majority, resamples=200, seed=3)
    reverse = bootstrap_delta_macro_f1(truth, majority, perfect, resamples=200, seed=3)
    assert forward.observed == pytest.approx(-reverse.observed)


def test_the_bootstrap_refuses_misaligned_or_unknown_input() -> None:
    with pytest.raises(MismatchedPredictionsError):
        bootstrap_delta_macro_f1([HELMET], [HELMET], [HELMET, HELMET])
    with pytest.raises(MismatchedPredictionsError, match="outside the class space"):
        bootstrap_delta_macro_f1([HELMET], ["turban"], [HELMET])
    with pytest.raises(EmptyEvaluationError):
        bootstrap_delta_macro_f1([], [], [])


# --- the claim rule ----------------------------------------------------------------------


def test_a_difference_is_claimed_when_both_conditions_hold() -> None:
    verdict = decide(
        first_name="resnet50",
        second_name="deit_small",
        per_seed_delta=[0.03, 0.02, 0.04],
        interval=interval(0.01, 0.05, observed=0.03),
    )
    assert verdict.difference_claimed
    assert verdict.winner == "resnet50"
    assert verdict.sign_consistent
    assert "excludes zero" in verdict.rationale


def test_the_loser_is_named_when_the_delta_is_negative() -> None:
    verdict = decide(
        first_name="resnet50",
        second_name="deit_small",
        per_seed_delta=[-0.03, -0.02, -0.04],
        interval=interval(-0.05, -0.01, observed=-0.03),
    )
    assert verdict.difference_claimed
    assert verdict.winner == "deit_small"


def test_an_interval_containing_zero_is_a_tie_however_consistent_the_seeds() -> None:
    verdict = decide(
        first_name="resnet50",
        second_name="deit_small",
        per_seed_delta=[0.01, 0.02, 0.005],
        interval=interval(-0.01, 0.04, observed=0.012),
    )
    assert not verdict.difference_claimed
    assert verdict.winner is None
    assert verdict.sign_consistent
    assert "includes zero" in verdict.rationale


def test_sign_inconsistent_seeds_are_a_tie_even_when_the_interval_excludes_zero() -> None:
    """The clause that stops a lucky pooled interval from becoming a claim.

    Without sign-consistency, three seeds straddling zero could still produce a
    pooled CI that misses zero and be written up as a real difference.
    """

    verdict = decide(
        first_name="resnet50",
        second_name="deit_small",
        per_seed_delta=[0.05, -0.01, 0.03],
        interval=interval(0.005, 0.06, observed=0.023),
    )
    assert not verdict.difference_claimed
    assert verdict.winner is None
    assert not verdict.sign_consistent
    assert "do not agree on a sign" in verdict.rationale


def test_a_tie_is_never_written_up_as_equivalence() -> None:
    """§12 requires ties to be interpreted, not read as 'the models are the same'."""

    verdict = decide(
        first_name="resnet50",
        second_name="deit_small",
        per_seed_delta=[0.001, -0.002, 0.0005],
        interval=interval(-0.02, 0.02),
    )
    assert "TIE" in verdict.rationale
    assert "not as evidence" in verdict.rationale
    assert "latency" in verdict.rationale


def test_an_exactly_zero_delta_is_not_sign_consistent() -> None:
    """A dead heat on one seed cannot support a directional claim."""

    verdict = decide(
        first_name="a",
        second_name="b",
        per_seed_delta=[0.02, 0.0, 0.03],
        interval=interval(0.01, 0.04),
    )
    assert not verdict.sign_consistent
    assert not verdict.difference_claimed


def test_the_verdict_is_refused_without_any_seeds() -> None:
    with pytest.raises(EmptyEvaluationError):
        decide(
            first_name="a", second_name="b", per_seed_delta=[], interval=interval(0.1, 0.2)
        )


def test_the_verdict_carries_its_evidence() -> None:
    """The artifact must record what the decision was made from."""

    results = [mcnemar([True, False], [False, True])]
    verdict = decide(
        first_name="a",
        second_name="b",
        per_seed_delta=[0.02, 0.03, 0.01],
        interval=interval(0.005, 0.04, observed=0.02),
        mcnemar_results=results,
    )
    assert verdict.per_seed_delta == (0.02, 0.03, 0.01)
    assert len(verdict.mcnemar) == 1
    assert math.isfinite(verdict.interval.observed)


# --- the pooled interval (the §12 form) --------------------------------------------


def test_the_pooled_bootstrap_reduces_to_the_single_seed_case() -> None:
    """Three identical seeds must reproduce the single-seed result exactly."""

    truth = [HELMET] * 50 + [NO_HELMET] * 50
    perfect = list(truth)
    majority = [HELMET] * 100

    pooled = bootstrap_delta_macro_f1_pooled(
        truth,
        [perfect, perfect, perfect],
        [majority, majority, majority],
        resamples=200,
        seed=0,
    )
    single = bootstrap_delta_macro_f1(truth, perfect, majority, resamples=200, seed=0)
    assert pooled.observed == pytest.approx(single.observed)


def test_the_pooled_interval_carries_seed_variation() -> None:
    """Why the pooled form exists.

    Bootstrapping one seed describes that run, not the family. A family whose seeds
    disagree must not look as strong as one whose seeds all agree.
    """

    truth = [HELMET] * 50 + [NO_HELMET] * 50
    good = list(truth)
    bad = [HELMET] * 100

    agreeing = bootstrap_delta_macro_f1_pooled(
        truth, [good, good, good], [bad, bad, bad], resamples=300, seed=1
    )
    disagreeing = bootstrap_delta_macro_f1_pooled(
        truth, [good, good, bad], [bad, bad, bad], resamples=300, seed=1
    )
    assert disagreeing.observed < agreeing.observed


def test_the_pooled_bootstrap_is_seeded_and_reproducible() -> None:
    truth = [HELMET, NO_HELMET] * 25
    first = [[HELMET] * 50, list(truth)]
    second = [list(truth), [HELMET] * 50]
    a = bootstrap_delta_macro_f1_pooled(truth, first, second, resamples=150, seed=5)
    b = bootstrap_delta_macro_f1_pooled(truth, first, second, resamples=150, seed=5)
    assert (a.lower, a.upper, a.observed) == (b.lower, b.upper, b.observed)


def test_the_pooled_bootstrap_requires_every_seed_to_cover_the_split() -> None:
    """A seed that scored a different crop set would break the pairing."""

    truth = [HELMET, NO_HELMET] * 10
    with pytest.raises(MismatchedPredictionsError, match="whole test split"):
        bootstrap_delta_macro_f1_pooled(truth, [[HELMET] * 5], [list(truth)])


def test_the_pooled_bootstrap_refuses_a_family_with_no_seeds() -> None:
    truth = [HELMET, NO_HELMET]
    with pytest.raises(MismatchedPredictionsError, match="at least one seed"):
        bootstrap_delta_macro_f1_pooled(truth, [], [list(truth)])

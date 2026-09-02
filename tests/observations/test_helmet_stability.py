"""Temporal stabilization of helmet labels: a plurality vote, and nothing more.

What these tests are checking is narrow on purpose. The stabilizer is a **display**
aid; it makes no accuracy claim, and no test here asserts that smoothing produces a
*better* label -- only that it produces a deterministic, explainable, disclosed one.
That distinction is the module's whole contract.
"""

from __future__ import annotations

import pytest

from trafficpulse.observations.helmet_stability import (
    HelmetSample,
    HelmetStabilizationConfig,
    stabilize,
    stabilized_index,
    summarise_tracks,
)


def _samples(track: str, labels: list[str], *, score: float | None = 0.9) -> list[HelmetSample]:
    return [
        HelmetSample(frame_index=index, track_id=track, label=label, confidence=score)
        for index, label in enumerate(labels)
    ]


# --- the vote ---------------------------------------------------------------------
def test_a_single_frame_flip_does_not_survive_the_window() -> None:
    """The behaviour the module exists for (P4-U10 §5.1(7)).

    One dissenting frame inside a window of agreeing ones must not change the drawn
    label: that alternation is what made real footage unreadable.
    """

    stabilized = stabilize(_samples("r1", ["helmet", "helmet", "no_helmet", "helmet"]))

    assert [entry.label for entry in stabilized] == ["helmet"] * 4
    # The raw reading is preserved beside it -- the smoothing stays inspectable.
    assert [entry.raw_label for entry in stabilized] == [
        "helmet",
        "helmet",
        "no_helmet",
        "helmet",
    ]


def test_a_sustained_change_is_adopted_rather_than_suppressed() -> None:
    """Smoothing must lag a real change, not refuse it."""

    stabilized = stabilize(
        _samples("r1", ["helmet", "helmet", "no_helmet", "no_helmet", "no_helmet"]),
        config=HelmetStabilizationConfig(window=3, min_samples=1),
    )

    assert stabilized[-1].label == "no_helmet"


def test_agreement_reports_how_well_supported_the_label_is() -> None:
    """A 2-of-3 call must not be presentable as a unanimous one."""

    stabilized = stabilize(
        _samples("r1", ["helmet", "helmet", "no_helmet"]),
        config=HelmetStabilizationConfig(window=3, min_samples=1),
    )

    assert stabilized[-1].label == "helmet"
    assert stabilized[-1].agreement == pytest.approx(2 / 3)


def test_an_early_reading_is_reported_as_unsettled_rather_than_firm() -> None:
    """One frame is not evidence of stability, and the output says so."""

    stabilized = stabilize(
        _samples("r1", ["helmet", "helmet", "helmet"]),
        config=HelmetStabilizationConfig(min_samples=3),
    )

    assert [entry.settled for entry in stabilized] == [False, False, True]
    # The label is still emitted: it is the best available reading, not a refusal.
    assert stabilized[0].label == "helmet"


def test_an_abstention_votes_like_any_other_observation() -> None:
    """A crop the classifier could not read is a fact about that frame.

    Dropping abstentions would let one confident frame speak for a rider the
    classifier mostly could not see -- which is the opposite of honest.
    """

    samples = [
        HelmetSample(frame_index=0, track_id="r1", label="uncertain", confidence=None),
        HelmetSample(frame_index=1, track_id="r1", label="uncertain", confidence=None),
        HelmetSample(frame_index=2, track_id="r1", label="no_helmet", confidence=0.99),
    ]

    assert stabilize(samples)[-1].label == "uncertain"


def test_confidence_is_never_fabricated_for_an_unscored_window() -> None:
    """``None`` means "not measured" and must survive the fold as ``None``."""

    samples = _samples("r1", ["uncertain", "uncertain"], score=None)

    assert stabilize(samples)[-1].confidence is None


def test_the_reported_confidence_averages_only_the_supporting_samples() -> None:
    """The number shown belongs to the label shown, not to the whole window."""

    samples = [
        HelmetSample(frame_index=0, track_id="r1", label="helmet", confidence=0.8),
        HelmetSample(frame_index=1, track_id="r1", label="helmet", confidence=1.0),
        HelmetSample(frame_index=2, track_id="r1", label="no_helmet", confidence=0.1),
    ]

    assert stabilize(samples)[-1].confidence == pytest.approx(0.9)


# --- isolation + determinism -------------------------------------------------------
def test_one_riders_readings_never_influence_anothers() -> None:
    samples = [
        *_samples("r1", ["no_helmet", "no_helmet", "no_helmet"]),
        *_samples("r2", ["helmet", "helmet", "helmet"]),
    ]

    by_track = {
        entry.track_id: entry.label for entry in stabilize(samples) if entry.frame_index == 2
    }

    assert by_track == {"r1": "no_helmet", "r2": "helmet"}


def test_output_does_not_depend_on_input_order() -> None:
    """Replay determinism: the fold is a function of the samples, not their arrival."""

    samples = [
        *_samples("r1", ["helmet", "no_helmet", "helmet"]),
        *_samples("r2", ["no_helmet", "no_helmet", "helmet"]),
    ]

    assert stabilize(samples) == stabilize(list(reversed(samples)))


def test_a_tie_breaks_deterministically_on_score_then_name() -> None:
    """Two labels tied on count are separated by summed confidence, then by string."""

    samples = [
        HelmetSample(frame_index=0, track_id="r1", label="helmet", confidence=0.6),
        HelmetSample(frame_index=1, track_id="r1", label="no_helmet", confidence=0.95),
    ]

    assert stabilize(samples, config=HelmetStabilizationConfig(window=2))[-1].label == (
        "no_helmet"
    )


def test_disabling_the_stabilizer_passes_raw_labels_through_untouched() -> None:
    """There must be no second code path downstream for "unsmoothed"."""

    labels = ["helmet", "no_helmet", "helmet"]
    stabilized = stabilize(
        _samples("r1", labels), config=HelmetStabilizationConfig(enabled=False)
    )

    assert [entry.label for entry in stabilized] == labels
    assert all(entry.agreement == 1.0 and entry.settled for entry in stabilized)


def test_the_index_is_keyed_for_per_frame_lookup() -> None:
    index = stabilized_index(_samples("r1", ["helmet", "helmet"]))

    assert index[(1, "r1")].label == "helmet"
    assert (2, "r1") not in index


# --- the per-track fold -------------------------------------------------------------
def test_the_summary_reports_raw_instability_beside_the_smoothed_one() -> None:
    """The P4-U10 measurement, taken on live output rather than quoted."""

    summary = summarise_tracks(
        _samples("r1", ["helmet", "no_helmet", "helmet", "no_helmet", "helmet"])
    )[0]

    assert summary.raw_flips == 4
    assert summary.stabilized_flips < summary.raw_flips
    assert summary.label_counts == (("helmet", 3), ("no_helmet", 2))


def test_the_summary_reports_the_reading_the_run_ended_on() -> None:
    """It must agree with the last frame a viewer sees, not with a whole-run vote."""

    summary = summarise_tracks(
        _samples("r1", ["helmet", "helmet", "helmet", "no_helmet", "no_helmet", "no_helmet"]),
        config=HelmetStabilizationConfig(window=3, min_samples=1),
    )[0]

    assert summary.label == "no_helmet"
    assert (summary.first_frame, summary.last_frame) == (0, 5)


def test_an_empty_stream_folds_to_nothing_rather_than_a_zero_row() -> None:
    assert summarise_tracks([]) == ()

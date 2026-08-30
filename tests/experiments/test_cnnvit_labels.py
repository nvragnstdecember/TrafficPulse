"""The HELMET positional-encoding label grammar (P4-U5).

The label is the experiment's entire supervision signal, so these tests care most
about what the parser *refuses*. A parser that quietly accepts ``DHelmetX`` as
``DHelmet`` would shift the class balance without anything failing.
"""

from __future__ import annotations

import pytest
from helmet_cnn_vit.errors import MalformedRiderLabelError
from helmet_cnn_vit.labels import (
    VERIFIED_LABEL_COUNTS,
    HelmetState,
    RiderPosition,
    parse_label,
)

# --- the real vocabulary -------------------------------------------------------


def test_the_verified_vocabulary_has_the_36_observed_labels() -> None:
    """The count the literature reports, and the count the archive actually holds."""

    assert len(VERIFIED_LABEL_COUNTS) == 36


def test_the_verified_counts_sum_to_the_archives_row_count() -> None:
    """283,377 rows across the 910 annotation CSVs, counted at acquisition."""

    assert sum(VERIFIED_LABEL_COUNTS.values()) == 283_377


@pytest.mark.parametrize("label", sorted(VERIFIED_LABEL_COUNTS))
def test_every_real_label_round_trips(label: str) -> None:
    """Parsing then re-serialising must reproduce the source string exactly."""

    assert parse_label(label).to_label() == label


@pytest.mark.parametrize("label", sorted(VERIFIED_LABEL_COUNTS))
def test_every_real_label_names_exactly_one_driver(label: str) -> None:
    positions = [position for position, _ in parse_label(label).riders]
    assert positions.count(RiderPosition.DRIVER) == 1
    assert positions[0] is RiderPosition.DRIVER


# --- derived properties ---------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "driver", "riders", "any_bare"),
    [
        ("DHelmet", HelmetState.HELMET, 1, False),
        ("DNoHelmet", HelmetState.NO_HELMET, 1, True),
        ("DHelmetP1Helmet", HelmetState.HELMET, 2, False),
        # The case that matters most: a helmeted driver carrying a bare-headed
        # pillion. The classification target is the DRIVER, so this is `helmet`
        # even though the motorcycle is in violation.
        ("DHelmetP1NoHelmet", HelmetState.HELMET, 2, True),
        ("DNoHelmetP1Helmet", HelmetState.NO_HELMET, 2, True),
        ("DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet", HelmetState.HELMET, 5, True),
        ("DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet", HelmetState.NO_HELMET, 5, True),
    ],
)
def test_derived_properties(
    label: str, driver: HelmetState, riders: int, any_bare: bool
) -> None:
    config = parse_label(label)
    assert config.driver_state is driver
    assert config.rider_count == riders
    assert config.any_no_helmet is any_bare


def test_driver_state_is_independent_of_passenger_states() -> None:
    """The target must not be contaminated by the pillion labels."""

    assert parse_label("DHelmetP1NoHelmetP2NoHelmet").driver_state is HelmetState.HELMET
    assert parse_label("DNoHelmetP1HelmetP2Helmet").driver_state is HelmetState.NO_HELMET


# --- what the parser must refuse -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("", "empty label"),
        ("Helmet", "no position token"),
        ("DHelmetX", "trailing junk"),
        ("XHelmet", "unknown position"),
        ("DSomething", "unknown state"),
        ("D", "position without a state"),
        ("P1Helmet", "driver missing"),
        ("P1HelmetDHelmet", "driver not first"),
        ("DHelmetDNoHelmet", "driver repeated"),
        ("DHelmetP1HelmetP1NoHelmet", "passenger repeated"),
        ("dhelmet", "wrong case"),
        ("DHelmet P1Helmet", "interior whitespace"),
    ],
)
def test_malformed_labels_are_rejected(label: str, reason: str) -> None:
    with pytest.raises(MalformedRiderLabelError):
        parse_label(label)


def test_helmet_is_not_mistaken_for_the_suffix_of_no_helmet() -> None:
    """`Helmet` is a suffix of `NoHelmet`; a greedy tokeniser would invert the label.

    This is the single most dangerous way this parser could fail, because the
    result would be a plausible configuration with every class flipped.
    """

    assert parse_label("DNoHelmet").driver_state is HelmetState.NO_HELMET
    assert parse_label("DHelmet").driver_state is HelmetState.HELMET
    assert parse_label("DNoHelmetP1NoHelmet").to_label() == "DNoHelmetP1NoHelmet"


def test_the_no_helmet_share_of_the_verified_vocabulary_is_recorded() -> None:
    """Row-weighted driver-state balance, reproducible from the repository alone."""

    bare = sum(
        count
        for label, count in VERIFIED_LABEL_COUNTS.items()
        if parse_label(label).driver_state is HelmetState.NO_HELMET
    )
    assert bare == 90_342
    assert 0.31 < bare / sum(VERIFIED_LABEL_COUNTS.values()) < 0.32

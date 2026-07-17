"""Rider <-> motorcycle association derivation (P4-U4).

Pure geometry over frozen contracts: no pixels, no classifier, no ML.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from trafficpulse.association import (
    RiderAssociationConfig,
    associate_riders,
    overlap_over_min_area,
)
from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import AssociationType, ObjectClass, TrackStatus

BASE = datetime(1970, 1, 1, tzinfo=UTC)


def state(
    track_id: str,
    object_class: ObjectClass,
    box: tuple[float, float, float, float],
    *,
    tainted: bool = False,
    frame_index: int = 0,
) -> TrackState:
    x1, y1, x2, y2 = box
    return TrackState(
        track_id=track_id,
        camera_id="cam-1",
        timestamp=BASE + timedelta(seconds=frame_index),
        frame_index=frame_index,
        object_class=object_class,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        status=TrackStatus.ACTIVE,
        tainted=tainted,
    )


def bike(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.MOTORCYCLE, box, **kw)  # type: ignore[arg-type]


def person(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.PERSON, box, **kw)  # type: ignore[arg-type]


# --- the overlap measure -----------------------------------------------------
def test_disjoint_boxes_do_not_overlap() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=20, y1=20, x2=30, y2=30)
    assert overlap_over_min_area(a, b) == 0.0


def test_touching_boxes_do_not_overlap() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=10, y1=0, x2=20, y2=10)
    assert overlap_over_min_area(a, b) == 0.0


def test_fully_contained_box_scores_one() -> None:
    """IoMin, not IoU: full containment is 1.0 regardless of the size difference."""

    big = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    small = BoundingBox(x1=10, y1=10, x2=20, y2=20)
    assert overlap_over_min_area(big, small) == pytest.approx(1.0)


def test_overlap_is_symmetric() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=5, y1=0, x2=25, y2=10)
    assert overlap_over_min_area(a, b) == pytest.approx(overlap_over_min_area(b, a))


def test_half_overlap_scores_half() -> None:
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)  # area 100, the smaller
    b = BoundingBox(x1=5, y1=0, x2=25, y2=10)  # intersection 50
    assert overlap_over_min_area(a, b) == pytest.approx(0.5)


# --- association -------------------------------------------------------------
def test_rider_on_a_motorcycle_associates() -> None:
    states = [bike("m1", (0, 50, 40, 100)), person("p1", (5, 20, 35, 90))]

    associations = associate_riders(states)

    assert len(associations) == 1
    link = associations[0]
    assert link.subject_track_id == "p1"  # the rider
    assert link.object_track_id == "m1"  # the motorcycle
    assert link.association_type is AssociationType.RIDER_OF_MOTORCYCLE
    assert link.camera_id == "cam-1"


def test_distant_pedestrian_does_not_associate() -> None:
    states = [bike("m1", (0, 50, 40, 100)), person("p1", (500, 20, 530, 90))]
    assert associate_riders(states) == ()


def test_association_is_per_frame_and_carries_no_interval() -> None:
    """A per-frame fact: an instant, not a sustained window (see module docs)."""

    states = [bike("m1", (0, 50, 40, 100)), person("p1", (5, 20, 35, 90))]
    link = associate_riders(states)[0]

    assert link.interval is None
    assert link.timestamp == BASE


def test_confidence_is_the_overlap_ratio() -> None:
    states = [bike("m1", (0, 0, 100, 100)), person("p1", (0, 0, 50, 100))]  # IoMin 1.0

    assert associate_riders(states)[0].confidence == pytest.approx(1.0)


def test_confidence_stays_within_the_contract_bound() -> None:
    states = [bike("m1", (0, 0, 100, 100)), person("p1", (10, 10, 20, 20))]
    assert 0.0 <= associate_riders(states)[0].confidence <= 1.0


def test_two_riders_on_one_motorcycle_both_associate() -> None:
    """Triple riding must remain expressible: no uniqueness on the bike side."""

    states = [
        bike("m1", (0, 50, 60, 100)),
        person("p1", (5, 20, 35, 90)),
        person("p2", (25, 20, 55, 90)),
    ]

    associations = associate_riders(states)

    assert {a.subject_track_id for a in associations} == {"p1", "p2"}
    assert {a.object_track_id for a in associations} == {"m1"}


def test_rider_is_assigned_to_at_most_one_motorcycle() -> None:
    states = [
        bike("m1", (0, 0, 100, 100)),
        bike("m2", (0, 0, 100, 100)),
        person("p1", (10, 10, 40, 90)),
    ]

    associations = associate_riders(states)

    assert len(associations) == 1


def test_best_overlap_wins_between_competing_motorcycles() -> None:
    states = [
        bike("m_far", (0, 0, 20, 100)),  # clips the rider's left edge only (IoMin ~0.32)
        bike("m_near", (10, 10, 45, 95)),  # contains the rider (IoMin 1.0)
        person("p1", (12, 12, 42, 92)),
    ]

    assert associate_riders(states)[0].object_track_id == "m_near"


def test_iomin_saturates_when_the_rider_is_inside_both_motorcycles() -> None:
    """A stated limitation of IoMin, pinned so it cannot regress silently.

    IoMin is "how much of the smaller box lies inside the larger", so a rider box
    fully contained by two overlapping motorcycle boxes scores 1.0 against BOTH and
    the measure cannot discriminate. The tie then resolves deterministically on the
    lowest track id -- a defensible arbitrary choice, not a claim about which bike
    the rider is actually on. Separating genuinely ambiguous overlaps needs
    persistence over time or depth, neither of which exists yet.
    """

    states = [
        bike("m_b", (0, 0, 100, 100)),
        bike("m_a", (5, 5, 95, 95)),
        person("p1", (12, 12, 42, 92)),  # fully inside both
    ]

    associations = associate_riders(states)

    assert len(associations) == 1
    assert associations[0].confidence == pytest.approx(1.0)
    assert associations[0].object_track_id == "m_a"  # tie -> lowest id


def test_exact_tie_breaks_on_lowest_track_id() -> None:
    """Determinism: identical geometry must not depend on input order."""

    box = (0.0, 0.0, 100.0, 100.0)
    states = [bike("m_b", box), bike("m_a", box), person("p1", (10, 10, 40, 90))]

    assert associate_riders(states)[0].object_track_id == "m_a"


# --- taint -------------------------------------------------------------------
def test_tainted_rider_abstains() -> None:
    states = [bike("m1", (0, 50, 40, 100)), person("p1", (5, 20, 35, 90), tainted=True)]
    assert associate_riders(states) == ()


def test_tainted_motorcycle_abstains() -> None:
    states = [bike("m1", (0, 50, 40, 100), tainted=True), person("p1", (5, 20, 35, 90))]
    assert associate_riders(states) == ()


# --- irrelevant classes ------------------------------------------------------
def test_cars_are_ignored() -> None:
    states = [
        state("c1", ObjectClass.CAR, (0, 0, 100, 100)),
        person("p1", (10, 10, 40, 90)),
    ]
    assert associate_riders(states) == ()


def test_no_riders_or_no_motorcycles_yields_nothing() -> None:
    assert associate_riders([bike("m1", (0, 0, 40, 40))]) == ()
    assert associate_riders([person("p1", (0, 0, 40, 40))]) == ()
    assert associate_riders([]) == ()


# --- determinism -------------------------------------------------------------
def test_output_is_independent_of_input_order() -> None:
    states = [
        bike("m1", (0, 50, 60, 100)),
        person("p1", (5, 20, 35, 90)),
        person("p2", (25, 20, 55, 90)),
    ]

    forward = associate_riders(states)
    backward = associate_riders(list(reversed(states)))

    assert forward == backward


def test_ids_are_deterministic_and_content_derived() -> None:
    states = [bike("m1", (0, 50, 40, 100)), person("p1", (5, 20, 35, 90))]

    assert associate_riders(states)[0].association_id == associate_riders(states)[0].association_id


def test_association_ids_differ_across_riders() -> None:
    states = [
        bike("m1", (0, 50, 60, 100)),
        person("p1", (5, 20, 35, 90)),
        person("p2", (25, 20, 55, 90)),
    ]

    ids = {a.association_id for a in associate_riders(states)}
    assert len(ids) == 2


def test_input_states_are_not_mutated() -> None:
    states = [bike("m1", (0, 50, 40, 100)), person("p1", (5, 20, 35, 90))]
    before = [s.model_dump_json() for s in states]

    associate_riders(states)

    assert [s.model_dump_json() for s in states] == before


# --- configuration -----------------------------------------------------------
def test_min_overlap_gates_the_association() -> None:
    states = [bike("m1", (0, 0, 100, 100)), person("p1", (90, 90, 110, 110))]  # small overlap

    assert associate_riders(states, config=RiderAssociationConfig(min_overlap=0.9)) == ()
    assert len(associate_riders(states, config=RiderAssociationConfig(min_overlap=0.01))) == 1


def test_config_is_frozen_and_strict() -> None:
    config = RiderAssociationConfig()
    with pytest.raises(ValidationError):
        config.min_overlap = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RiderAssociationConfig(unknown=1)  # type: ignore[call-arg]


def test_min_overlap_is_bounded() -> None:
    with pytest.raises(ValidationError):
        RiderAssociationConfig(min_overlap=1.5)
    with pytest.raises(ValidationError):
        RiderAssociationConfig(min_overlap=-0.1)

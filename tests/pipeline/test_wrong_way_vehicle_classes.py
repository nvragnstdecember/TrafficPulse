"""Wrong-way applies only to road vehicles, never to pedestrians (or plates).

``person`` is in the deployed label map because helmet and triple-riding reasoning
need riders, not because pedestrians commit movement violations. A person's track
opposes the traffic direction as a matter of course -- people cross roads, stand in
them and double back -- so scoring one against a vehicle's angular threshold is a
false positive waiting on a threshold.

That was not hypothetical. During real-footage validation of a contraflow site, a
hi-vis traffic controller standing in the road produced "against lane flow"
readings at 142 deg and 178 deg; nothing but temporal persistence stopped a
confirmation. These tests pin the class gate that removes that dependence.

The companion gate -- *where* a track has to be, rather than *what* it has to be --
is covered by ``test_wrong_way_containment.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from _pipeline_helpers import (
    DEFAULT_FRAME_COUNT,
    LANE_X,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_raw,
)

from trafficpulse.contracts.enums import ObjectClass
from trafficpulse.detector import DetectorConfig, StubDetector
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.pipeline.base import CompositionPipeline
from trafficpulse.pipeline.wrong_way import wrong_way_finalize_strategy
from trafficpulse.rules.red_light import VEHICLE_CLASSES as RED_LIGHT_CLASSES
from trafficpulse.rules.vehicles import VEHICLE_CLASSES
from trafficpulse.tracking import ScriptedAssignment, StubTracker

# One label per class, so a scripted detector can emit any of them into the lane.
_LABELS: dict[ObjectClass, str] = {
    ObjectClass.CAR: "car",
    ObjectClass.MOTORCYCLE: "motorcycle",
    ObjectClass.BUS: "bus",
    ObjectClass.TRUCK: "truck",
    ObjectClass.AUTO_RICKSHAW: "auto_rickshaw",
    ObjectClass.BICYCLE: "bicycle",
    ObjectClass.PERSON: "person",
    ObjectClass.LICENSE_PLATE: "license_plate",
}
_CONFIG = DetectorConfig(label_map={label: cls for cls, label in _LABELS.items()})

EXCLUDED = sorted(set(_LABELS) - VEHICLE_CLASSES, key=lambda c: c.value)
INCLUDED = sorted(VEHICLE_CLASSES, key=lambda c: c.value)


def _frames():  # type: ignore[no-untyped-def]
    return [make_frame_record(i) for i in range(DEFAULT_FRAME_COUNT)]


def _detector(object_class: ObjectClass, *, x: float = LANE_X) -> StubDetector:
    """One track of ``object_class`` driving the wrong way down the governed lane."""

    label = _LABELS[object_class]
    return StubDetector(
        per_frame={
            i: (replace(moving_raw(i, x=x), label=label),)
            for i in range(DEFAULT_FRAME_COUNT)
        }
    )


def _tracker(*track_ids: str) -> StubTracker:
    ids = track_ids or ("only",)
    return StubTracker(
        {
            i: tuple(ScriptedAssignment(track_id=t) for t in ids)
            for i in range(DEFAULT_FRAME_COUNT)
        }
    )


def _pipeline(detector: StubDetector) -> WrongWayPipeline:
    return WrongWayPipeline(
        detector=detector,
        tracker=_tracker(),
        scene=SCENE,
        detector_config=_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )


def _composed(detector: StubDetector, strategy) -> CompositionPipeline:  # type: ignore[no-untyped-def]
    """The composition ``engine/rules.py`` builds: an explicit strategy + the core.

    Used where a test needs the strategy object the run actually reasons with --
    to read its overlay capture, or to hand it a non-default class set.
    """

    return CompositionPipeline(
        detector=detector,
        tracker=_tracker(),
        scene=SCENE,
        detector_config=_CONFIG,
        finalize_strategy=strategy,
    )


# --- the requirement ----------------------------------------------------------
def test_a_pedestrian_walking_the_wrong_way_down_the_lane_is_not_a_violation() -> None:
    # Same lane, same motion, same thresholds as the confirming vehicle case below.
    # The only difference is what the object is.
    assert _pipeline(_detector(ObjectClass.PERSON)).process(_frames()) == ()


@pytest.mark.parametrize("object_class", EXCLUDED, ids=lambda c: c.value)
def test_no_excluded_class_can_produce_a_wrong_way_event(object_class: ObjectClass) -> None:
    assert _pipeline(_detector(object_class)).process(_frames()) == ()


# --- legitimate detection remains intact --------------------------------------
@pytest.mark.parametrize("object_class", INCLUDED, ids=lambda c: c.value)
def test_every_vehicle_class_still_confirms_a_wrong_way_event(
    object_class: ObjectClass,
) -> None:
    events = _pipeline(_detector(object_class)).process(_frames())
    assert len(events) == 1
    assert events[0].track_ids == ("only",)


def test_a_pedestrian_beside_a_vehicle_does_not_suppress_the_vehicle() -> None:
    # The realistic frame: a person in the road and a vehicle driving the wrong way
    # past them. Exactly one of the two is this rule's business.
    per_frame = {
        i: (
            replace(moving_raw(i, x=LANE_X), label="car"),
            replace(moving_raw(i, x=LANE_X + 40.0), label="person"),
        )
        for i in range(DEFAULT_FRAME_COUNT)
    }
    script = {
        i: (ScriptedAssignment(track_id="vehicle"), ScriptedAssignment(track_id="pedestrian"))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    pipeline = WrongWayPipeline(
        detector=StubDetector(per_frame=per_frame),
        tracker=StubTracker(script),
        scene=SCENE,
        detector_config=_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )
    events = pipeline.process(_frames())
    assert {tid for e in events for tid in e.track_ids} == {"vehicle"}


# --- the gate reaches the overlay too -----------------------------------------
def test_an_excluded_track_contributes_no_overlay_verdict() -> None:
    # The observed failure was visual as much as logical: the annotated video showed
    # "Against lane flow 178 deg" over a traffic controller. Returning before the
    # derivation keeps excluded tracks out of the capture entirely, so the overlay
    # cannot show a verdict the rule never intended to reach.
    strategy = wrong_way_finalize_strategy(SCENE, direction_id=NORTH_DIRECTION_ID)
    assert _composed(_detector(ObjectClass.PERSON), strategy).process(_frames()) == ()
    assert strategy.capture.frames == []


def test_a_vehicle_does_contribute_an_overlay_verdict() -> None:
    # The control for the test above: the capture IS populated for traffic the rule
    # may reason about, so an empty capture means the gate fired, not that captures
    # are broken.
    strategy = wrong_way_finalize_strategy(SCENE, direction_id=NORTH_DIRECTION_ID)
    assert len(_composed(_detector(ObjectClass.CAR), strategy).process(_frames())) == 1
    assert strategy.capture.frames


# --- the set itself -----------------------------------------------------------
def test_pedestrians_and_plates_are_excluded_by_the_shared_set() -> None:
    assert ObjectClass.PERSON not in VEHICLE_CLASSES
    assert ObjectClass.LICENSE_PLATE not in VEHICLE_CLASSES


def test_wrong_way_and_red_light_share_one_definition_so_they_cannot_drift() -> None:
    assert VEHICLE_CLASSES is RED_LIGHT_CLASSES


def test_the_strategy_defaults_to_the_shared_set() -> None:
    strategy = wrong_way_finalize_strategy(SCENE, direction_id=NORTH_DIRECTION_ID)
    assert strategy.vehicle_classes == VEHICLE_CLASSES


def test_a_deployment_can_widen_the_set_without_editing_the_rule() -> None:
    # Proves the exclusion is this filter doing the work rather than some unrelated
    # reason a pedestrian track fails to confirm: admit PERSON and it confirms.
    widened = VEHICLE_CLASSES | {ObjectClass.PERSON}
    strategy = wrong_way_finalize_strategy(
        SCENE, direction_id=NORTH_DIRECTION_ID, vehicle_classes=widened
    )
    events = _composed(_detector(ObjectClass.PERSON), strategy).process(_frames())
    assert len(events) == 1

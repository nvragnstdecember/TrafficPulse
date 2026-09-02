"""Dominant traffic-flow estimation from observed tracks (H12).

These tests pin the behaviour that matters for calibration: it measures the
traffic stream, it excludes what is not traffic, and it says "I cannot tell"
rather than inventing a direction.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import ObjectClass, TrackStatus
from trafficpulse.scenes import FLOW_CLASSES, estimate_dominant_flow

_T0 = datetime(1970, 1, 1, tzinfo=UTC)


_BOX = 20.0


def _state(
    track_id: str,
    *,
    seconds: float,
    cx: float,
    cy: float,
    object_class: ObjectClass = ObjectClass.CAR,
) -> TrackState:
    # The box's top-left is placed at (cx, cy); its centre is a constant offset
    # from that, so centre *displacement* is exactly the displacement asked for
    # while every coordinate stays in the non-negative range BoundingBox requires.
    return TrackState(
        track_id=track_id,
        camera_id="cam-1",
        timestamp=_T0 + timedelta(seconds=seconds),
        object_class=object_class,
        bbox=BoundingBox(x1=cx, y1=cy, x2=cx + _BOX, y2=cy + _BOX),
        status=TrackStatus.ACTIVE,
    )


def _mover(
    track_id: str,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    seconds: float = 2.0,
    object_class: ObjectClass = ObjectClass.CAR,
) -> list[TrackState]:
    return [
        _state(track_id, seconds=0.0, cx=start[0], cy=start[1], object_class=object_class),
        _state(track_id, seconds=seconds, cx=end[0], cy=end[1], object_class=object_class),
    ]


def test_the_flow_is_the_direction_the_traffic_actually_travels() -> None:
    states = [
        *_mover("t1", start=(100.0, 500.0), end=(100.0, 100.0)),
        *_mover("t2", start=(140.0, 520.0), end=(140.0, 140.0)),
    ]

    flow = estimate_dominant_flow(states)

    assert flow is not None
    assert flow.dx == 0.0 and flow.dy == -1.0  # straight "up" the frame
    assert flow.mover_count == 2
    assert math.isclose(flow.heading_degrees, 270.0)


def test_the_estimate_is_a_unit_vector() -> None:
    flow = estimate_dominant_flow(_mover("t1", start=(0.0, 0.0), end=(300.0, 400.0)))

    assert flow is not None
    assert math.isclose(math.hypot(flow.dx, flow.dy), 1.0, abs_tol=1e-4)


def test_states_may_arrive_unordered_and_interleaved() -> None:
    # A caller hands over a whole run's states; grouping and ordering are this
    # function's job, not the caller's.
    ordered = [*_mover("t1", start=(0.0, 400.0), end=(0.0, 0.0))]
    flow = estimate_dominant_flow(list(reversed(ordered)))

    assert flow is not None and flow.dy == -1.0


def test_a_stationary_track_does_not_vote() -> None:
    # Detector jitter on a parked vehicle would otherwise contribute a short,
    # near-random vector and pull the estimate off the traffic axis.
    states = [
        *_mover("moving", start=(0.0, 400.0), end=(0.0, 0.0)),
        *_mover("parked", start=(300.0, 300.0), end=(303.0, 302.0)),
    ]

    flow = estimate_dominant_flow(states)

    assert flow is not None
    assert flow.mover_count == 1
    assert flow.track_count == 2  # it was seen, it just did not qualify
    assert flow.dy == -1.0


def test_a_brief_track_does_not_vote() -> None:
    states = [
        *_mover("long", start=(0.0, 400.0), end=(0.0, 0.0), seconds=3.0),
        *_mover("blink", start=(50.0, 400.0), end=(400.0, 400.0), seconds=0.2),
    ]

    flow = estimate_dominant_flow(states)

    assert flow is not None and flow.mover_count == 1 and flow.dy == -1.0


def test_pedestrians_do_not_define_the_legal_direction_of_a_road() -> None:
    # The label map carries `person` because helmet reasoning needs riders. A
    # pedestrian crossing the carriageway travels *across* the traffic axis, so
    # counting them would rotate the calibrated direction.
    states = [
        *_mover("car", start=(0.0, 400.0), end=(0.0, 0.0)),
        *_mover(
            "walker",
            start=(0.0, 200.0),
            end=(400.0, 200.0),
            object_class=ObjectClass.PERSON,
        ),
    ]

    flow = estimate_dominant_flow(states)

    assert flow is not None
    assert flow.dx == 0.0 and flow.dy == -1.0
    assert flow.mover_count == 1


def test_motorcycles_count_as_traffic() -> None:
    flow = estimate_dominant_flow(
        _mover("m1", start=(0.0, 400.0), end=(0.0, 0.0), object_class=ObjectClass.MOTORCYCLE)
    )

    assert flow is not None and flow.mover_count == 1


def test_no_observed_motion_returns_none_rather_than_a_default_direction() -> None:
    # The honest answer for footage that cannot show a flow. Substituting a default
    # would calibrate a scene against a direction nothing in the clip supports, and
    # every vehicle would then be judged wrong-way or none would.
    assert estimate_dominant_flow([]) is None
    assert estimate_dominant_flow(_mover("parked", start=(0.0, 0.0), end=(2.0, 2.0))) is None
    assert estimate_dominant_flow([_state("single", seconds=0.0, cx=1.0, cy=1.0)]) is None


def test_two_way_traffic_that_cancels_returns_none() -> None:
    # Equal and opposite streams: a single legal direction cannot describe this
    # road, and picking one would make half the traffic a violation.
    states = [
        *_mover("north", start=(0.0, 400.0), end=(0.0, 0.0)),
        *_mover("south", start=(100.0, 0.0), end=(100.0, 400.0)),
    ]

    assert estimate_dominant_flow(states) is None


def test_thresholds_are_caller_overridable() -> None:
    brief = _mover("t1", start=(0.0, 100.0), end=(0.0, 40.0), seconds=0.3)

    assert estimate_dominant_flow(brief) is None
    assert estimate_dominant_flow(brief, min_lifetime_seconds=0.1) is not None


def test_flow_estimate_excludes_pedestrians() -> None:
    """Persons are detected (helmet reasoning needs riders) but never define flow.

    ``FLOW_CLASSES`` is the default class filter every auto-calibrated upload is
    measured through (``ProcessingService`` calls :func:`estimate_dominant_flow`
    without overriding it). Admitting ``PERSON`` here would let footpath movement
    define a road's legal direction, and the failure is silent: the scene still
    binds, and every vehicle is then judged against a pedestrian's heading.
    """

    assert ObjectClass.PERSON not in FLOW_CLASSES
    assert ObjectClass.CAR in FLOW_CLASSES
    assert ObjectClass.MOTORCYCLE in FLOW_CLASSES

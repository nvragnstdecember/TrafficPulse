"""Forward stop-line crossing instants reported by the crossing derivation (H13).

Additive to the P3-U4 derivation. The crossing instant and the junction-entry
instant are deliberately *different* facts: a stop line and the junction it guards
need not be contiguous, and red-light reasoning must read the signal at the former.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import ObjectClass, TrackStatus
from trafficpulse.contracts.scene import (
    DirectionVector,
    StopLine,
    StopLineEndpoints,
    Zone,
    ZoneType,
)
from trafficpulse.observations.crossing import derive_crossing_observations_with_taint

_T0 = datetime(1970, 1, 1, tzinfo=UTC)

# A stop line at y=100 crossed downward; the junction begins at y=150, so there is a
# real gap between crossing the line and occupying the polygon.
STOP_LINE = StopLine(
    stop_line_id="sl-1",
    enabled=True,
    endpoints=StopLineEndpoints(a=(0.0, 100.0), b=(300.0, 100.0)),
    crossing_direction=DirectionVector(dx=0.0, dy=1.0),
    signal_group_id="sg-1",
    zone_ids=("zone-junction",),
)
ZONE = Zone(
    zone_id="zone-junction",
    zone_type=ZoneType.INTERSECTION,
    enabled=True,
    polygon=((50.0, 150.0), (250.0, 150.0), (250.0, 260.0), (50.0, 260.0)),
)


_BOX_HEIGHT = 15.0


def _state(seconds: float, bottom_y: float, *, tainted: bool = False) -> TrackState:
    # The derivations use the bbox *bottom-centre* as the ground-contact reference,
    # so these fixtures are written in terms of that y and the box hangs above it.
    return TrackState(
        track_id="t-1",
        camera_id="cam-1",
        timestamp=_T0 + timedelta(seconds=seconds),
        object_class=ObjectClass.CAR,
        bbox=BoundingBox(x1=140.0, y1=bottom_y - _BOX_HEIGHT, x2=160.0, y2=bottom_y),
        status=TrackStatus.ACTIVE,
        tainted=tainted,
    )


def _track(*bottoms: float) -> list[TrackState]:
    return [_state(index * 0.5, bottom) for index, bottom in enumerate(bottoms)]


def test_the_crossing_instant_precedes_the_junction_entry() -> None:
    # Bottom-centre: 80 -> 110 (crosses y=100) -> 140 -> 170 (enters the polygon).
    derivation = derive_crossing_observations_with_taint(
        _track(80.0, 110.0, 140.0, 170.0), stop_line=STOP_LINE, zone=ZONE
    )

    crossed = derivation.forward_crossing_ids
    crossing = [o for o in derivation.observations if o.observation_id in crossed]
    entering = [o for o in derivation.observations if o.is_inside]

    assert len(crossing) == 1
    assert len(entering) == 1
    # The two are different steps: reading the signal at the second would exonerate
    # a vehicle that crossed on red and arrived after the light changed.
    assert crossing[0].timestamp < entering[0].timestamp


def test_a_backward_crossing_is_not_reported_as_a_forward_one() -> None:
    # Reversing over the line clears the validated entry; it is not a crossing in
    # the configured direction and must not latch anything.
    derivation = derive_crossing_observations_with_taint(
        _track(140.0, 80.0), stop_line=STOP_LINE, zone=ZONE
    )

    assert derivation.forward_crossing_ids == frozenset()
    assert not any(o.is_inside for o in derivation.observations)


def test_a_track_that_never_reaches_the_line_reports_no_crossing() -> None:
    derivation = derive_crossing_observations_with_taint(
        _track(20.0, 40.0, 60.0, 80.0), stop_line=STOP_LINE, zone=ZONE
    )

    assert derivation.forward_crossing_ids == frozenset()


def test_a_tainted_step_neither_crosses_nor_enters() -> None:
    # The crossing itself happens on a tainted step, so it is never validated. The
    # track then sits inside the polygon on clean steps -- and must still not count,
    # because an ID switch means those bytes may belong to a different vehicle.
    track = [
        _state(0.0, 80.0),
        _state(0.5, 110.0, tainted=True),  # the crossing step is unusable
        _state(1.0, 170.0),
        _state(1.5, 200.0),
        _state(2.0, 230.0),
    ]

    derivation = derive_crossing_observations_with_taint(
        track, stop_line=STOP_LINE, zone=ZONE
    )

    assert derivation.forward_crossing_ids == frozenset()
    # Without a validated crossing the polygon occupancy is not an entry.
    assert not any(o.is_inside for o in derivation.observations)
    # The first clean observation after the taint is flagged, so a reasoner breaks
    # its run there rather than bridging the discontinuity.
    assert derivation.taint_restart_ids


def test_the_new_field_defaults_empty_for_existing_constructions() -> None:
    # Additive: every pre-H13 construction of a CrossingDerivation still works.
    from trafficpulse.observations.crossing import CrossingDerivation

    derivation = CrossingDerivation((), frozenset())

    assert derivation.forward_crossing_ids == frozenset()

"""Tests for the stop-line / junction-entry crossing derivation (P3-U4).

Deterministic, model-free tests over the frozen ``InZoneObservation`` contract:
forward-crossing entry, reversing/non-crossing rejection, ``crossing_direction``
gating, the stop-line/zone gap bridged by the validated-entry flag,
``point_in_polygon`` boundary semantics, taint skip + restart, empty/short tracks,
zone-type validation, and byte-identical replay. Synthetic ``TrackState``
sequences only -- no detector, tracker, or video.
"""

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import (
    BoundingBox,
    InZoneObservation,
    ObjectClass,
    TrackState,
    TrackStatus,
    ZoneKind,
)
from trafficpulse.contracts.scene import (
    DirectionVector,
    StopLine,
    StopLineEndpoints,
    Zone,
    ZoneType,
)
from trafficpulse.observations.crossing import (
    DEFAULT_CROSSING_PRODUCER,
    derive_crossing_observations,
    derive_crossing_observations_with_taint,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
_HALF_W = 5.0
_HEIGHT = 10.0

# Stop line across y=100 (x in [0, 200]); legal crossing is upward (decreasing y).
STOP_LINE = StopLine(
    stop_line_id="sl-1",
    enabled=True,
    endpoints=StopLineEndpoints(a=(0.0, 100.0), b=(200.0, 100.0)),
    crossing_direction=DirectionVector(dx=0.0, dy=-1.0),
    signal_group_id="sg-1",
)
# Junction abutting the stop line: y in [0, 100].
JUNCTION_ABUTTING = Zone(
    zone_id="zone-junction",
    zone_type=ZoneType.INTERSECTION,
    enabled=True,
    polygon=((0.0, 100.0), (200.0, 100.0), (200.0, 0.0), (0.0, 0.0)),
)
# Junction separated from the stop line by a gap: y in [0, 60] (gap 60..100).
JUNCTION_GAPPED = Zone(
    zone_id="zone-junction-gap",
    zone_type=ZoneType.INTERSECTION,
    enabled=True,
    polygon=((0.0, 60.0), (200.0, 60.0), (200.0, 0.0), (0.0, 0.0)),
)


def _track(
    bottoms: list[tuple[float, float]],
    *,
    camera_id: str = "cam-1",
    track_id: str = "t1",
    tainted_indices: tuple[int, ...] = (),
) -> list[TrackState]:
    """Build a track whose bbox bottom-center equals each point in ``bottoms``."""

    states: list[TrackState] = []
    for i, (x, y) in enumerate(bottoms):
        states.append(
            TrackState(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=BASE + timedelta(seconds=i),
                frame_index=i,
                object_class=ObjectClass.CAR,
                bbox=BoundingBox(x1=x - _HALF_W, y1=y - _HEIGHT, x2=x + _HALF_W, y2=y),
                status=TrackStatus.ACTIVE,
                tainted=(i in tainted_indices),
            )
        )
    return states


def _inside_flags(track: list[TrackState], *, zone: Zone = JUNCTION_ABUTTING) -> list[bool]:
    obs = derive_crossing_observations(track, stop_line=STOP_LINE, zone=zone)
    return [o.is_inside for o in obs]


# --- forward crossing entry --------------------------------------------------
def test_forward_crossing_yields_entry() -> None:
    # Moving up: below the line, cross into the junction, sustained inside. Four
    # states -> three steps; the crossing is the 110->99 step.
    track = _track([(100.0, 120.0), (100.0, 110.0), (100.0, 99.0), (100.0, 90.0)])
    assert _inside_flags(track) == [False, True, True]


def test_entry_observation_carries_junction_zone() -> None:
    track = _track([(100.0, 110.0), (100.0, 99.0)])
    obs = derive_crossing_observations(track, stop_line=STOP_LINE, zone=JUNCTION_ABUTTING)
    (entry,) = [o for o in obs if o.is_inside]
    assert isinstance(entry, InZoneObservation)
    assert entry.zone_id == "zone-junction"
    assert entry.zone_kind is ZoneKind.JUNCTION_CONFLICT
    assert entry.track_id == "t1"
    assert entry.producer == DEFAULT_CROSSING_PRODUCER


# --- reversing / non-crossing rejection --------------------------------------
def test_reverse_crossing_is_not_an_entry() -> None:
    # Moving down across the line (out of the junction): never a validated entry.
    track = _track([(100.0, 90.0), (100.0, 110.0), (100.0, 120.0)])
    assert _inside_flags(track) == [False, False]


def test_inside_without_forward_crossing_is_not_an_entry() -> None:
    # A track observed only *inside* the polygon, never seen crossing the stop
    # line, never registers an entry (reversing/appeared-inside case).
    track = _track([(100.0, 90.0), (100.0, 80.0), (100.0, 70.0)])
    assert _inside_flags(track) == [False, False]


def test_movement_not_reaching_the_finite_segment_is_not_a_crossing() -> None:
    # A vertical move at x=250 is off the end of the finite stop-line segment
    # [x in 0..200]; the infinite line is crossed but the finite segment is not.
    off_segment = Zone(
        zone_id="z-wide",
        zone_type=ZoneType.INTERSECTION,
        enabled=True,
        polygon=((200.0, 100.0), (300.0, 100.0), (300.0, 0.0), (200.0, 0.0)),
    )
    track = _track([(250.0, 110.0), (250.0, 90.0), (250.0, 80.0)])
    assert _inside_flags(track, zone=off_segment) == [False, False]


# --- crossing_direction gating -----------------------------------------------
def test_crossing_direction_gates_forward() -> None:
    # Same upward movement, but the stop line's legal crossing is *downward* -> the
    # upward movement is a backward crossing -> no entry.
    downward_line = StopLine(
        stop_line_id="sl-down",
        enabled=True,
        endpoints=StopLineEndpoints(a=(0.0, 100.0), b=(200.0, 100.0)),
        crossing_direction=DirectionVector(dx=0.0, dy=1.0),  # legal crossing is downward
        signal_group_id="sg-1",
    )
    track = _track([(100.0, 110.0), (100.0, 90.0)])
    obs = derive_crossing_observations(track, stop_line=downward_line, zone=JUNCTION_ABUTTING)
    assert [o.is_inside for o in obs] == [False]


# --- gap bridged by the validated-entry flag ---------------------------------
def test_gap_between_stop_line_and_zone_is_bridged() -> None:
    # Cross the stop line (y=100), then enter the polygon (y<=60) a step later: the
    # entry registers at polygon entry, validated by the earlier forward crossing.
    track = _track([(100.0, 110.0), (100.0, 95.0), (100.0, 50.0), (100.0, 40.0)])
    assert _inside_flags(track, zone=JUNCTION_GAPPED) == [False, True, True]


def test_gap_zone_entered_without_forward_crossing_is_not_an_entry() -> None:
    # Reaching the gapped polygon from above without ever crossing the stop line.
    track = _track([(100.0, 90.0), (100.0, 55.0), (100.0, 45.0)])
    assert _inside_flags(track, zone=JUNCTION_GAPPED) == [False, False]


# --- point_in_polygon boundary semantics -------------------------------------
def test_boundary_membership_matches_point_in_polygon() -> None:
    # After a valid forward crossing, a current point exactly on the polygon edge
    # (y=60) counts as inside (boundary-inclusive), matching point_in_polygon.
    on_edge = _track([(100.0, 110.0), (100.0, 95.0), (100.0, 60.0)])
    assert _inside_flags(on_edge, zone=JUNCTION_GAPPED) == [False, True]
    # A point just past the edge (y=61) is outside.
    off_edge = _track([(100.0, 110.0), (100.0, 95.0), (100.0, 61.0)])
    assert _inside_flags(off_edge, zone=JUNCTION_GAPPED) == [False, False]


# --- taint skip + restart ----------------------------------------------------
def test_tainted_step_skipped_and_next_flagged_restart() -> None:
    # Forward-cross and enter (indices 0..2), taint index 3, then two clean steps.
    track = _track(
        [(100.0, 110.0), (100.0, 95.0), (100.0, 50.0), (100.0, 45.0), (100.0, 44.0),
         (100.0, 43.0)],
        tainted_indices=(3,),
    )
    result = derive_crossing_observations_with_taint(
        track, stop_line=STOP_LINE, zone=JUNCTION_GAPPED
    )
    # Steps: (0,1) cross, (1,2) enter -> True; (2,3) and (3,4) skipped (taint at 3);
    # (4,5) is the first clean step after taint.
    assert [o.timestamp.second for o in result.observations] == [1, 2, 5]
    restart_obs = next(
        o for o in result.observations if o.observation_id in result.taint_restart_ids
    )
    assert restart_obs.timestamp.second == 5


def test_taint_resets_validated_entry_no_bridge() -> None:
    # The clean step resuming after taint must NOT inherit the pre-taint forward
    # crossing: it is inside the polygon geometrically but reports is_inside=False.
    track = _track(
        [(100.0, 110.0), (100.0, 95.0), (100.0, 50.0), (100.0, 45.0), (100.0, 44.0),
         (100.0, 43.0)],
        tainted_indices=(3,),
    )
    result = derive_crossing_observations_with_taint(
        track, stop_line=STOP_LINE, zone=JUNCTION_GAPPED
    )
    by_second = {o.timestamp.second: o.is_inside for o in result.observations}
    assert by_second[2] is True  # entered before the taint
    assert by_second[5] is False  # resumed after taint: entry did not bridge


# --- empty / short track -----------------------------------------------------
def test_empty_track_yields_nothing() -> None:
    assert derive_crossing_observations([], stop_line=STOP_LINE, zone=JUNCTION_ABUTTING) == []


def test_single_state_yields_nothing() -> None:
    track = _track([(100.0, 99.0)])
    assert derive_crossing_observations(track, stop_line=STOP_LINE, zone=JUNCTION_ABUTTING) == []


# --- zone-type validation ----------------------------------------------------
def test_signal_controlled_region_is_accepted() -> None:
    zone = Zone(
        zone_id="z-ctrl",
        zone_type=ZoneType.SIGNAL_CONTROLLED_REGION,
        enabled=True,
        polygon=((0.0, 100.0), (200.0, 100.0), (200.0, 0.0), (0.0, 0.0)),
    )
    track = _track([(100.0, 110.0), (100.0, 99.0)])
    obs = derive_crossing_observations(track, stop_line=STOP_LINE, zone=zone)
    (entry,) = [o for o in obs if o.is_inside]
    assert entry.zone_kind is ZoneKind.JUNCTION_CONFLICT


def test_non_entry_zone_type_raises() -> None:
    lane = Zone(
        zone_id="z-lane",
        zone_type=ZoneType.LANE,
        enabled=True,
        polygon=((0.0, 100.0), (200.0, 100.0), (200.0, 0.0), (0.0, 0.0)),
    )
    track = _track([(100.0, 110.0), (100.0, 99.0)])
    with pytest.raises(ValueError, match="intersection or signal-controlled"):
        derive_crossing_observations(track, stop_line=STOP_LINE, zone=lane)


# --- determinism -------------------------------------------------------------
def test_replay_is_byte_identical() -> None:
    track = _track([(100.0, 110.0), (100.0, 95.0), (100.0, 50.0), (100.0, 40.0)])
    a = derive_crossing_observations(track, stop_line=STOP_LINE, zone=JUNCTION_GAPPED)
    b = derive_crossing_observations(track, stop_line=STOP_LINE, zone=JUNCTION_GAPPED)
    assert [o.observation_id for o in a] == [o.observation_id for o in b]
    assert [o.model_dump_json() for o in a] == [o.model_dump_json() for o in b]

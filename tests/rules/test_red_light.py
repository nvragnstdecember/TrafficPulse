"""Red-light jumping: the latching join and the reasoner (H13).

The decisive property under test is the latch. Red-light is an *act* committed at
the stop line, not a *condition* that persists — so a signal change after entry must
not be able to un-commit it. Several of these tests exist purely to make that
regression loud.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import InZoneObservation, Producer, SignalStateObservation
from trafficpulse.contracts.enums import ProducerKind, SignalState, ZoneKind
from trafficpulse.contracts.scene import ParameterStatus
from trafficpulse.observations.crossing import CrossingDerivation
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.red_light import (
    RedLightParameters,
    RedLightReasoner,
    join_entry_on_red,
    red_light_parameters,
)

_T0 = datetime(1970, 1, 1, tzinfo=UTC)
_CAMERA = "cam-1"
_ZONE = "zone-junction"
_PRODUCER = Producer(name="t", version="1", kind=ProducerKind.HEURISTIC)


def _at(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _crossing(
    seconds: float, *, inside: bool, track_id: str = "t-1", suffix: str = ""
) -> InZoneObservation:
    return InZoneObservation(
        observation_id=f"crs-{track_id}-{seconds}{suffix}",
        camera_id=_CAMERA,
        track_id=track_id,
        timestamp=_at(seconds),
        producer=_PRODUCER,
        zone_id=_ZONE,
        zone_kind=ZoneKind.JUNCTION_CONFLICT,
        is_inside=inside,
    )


def _signal(seconds: float, state: SignalState) -> SignalStateObservation:
    return SignalStateObservation(
        observation_id=f"sig-{seconds}",
        camera_id=_CAMERA,
        track_id=None,
        timestamp=_at(seconds),
        producer=_PRODUCER,
        signal_state=state,
    )


def _params(persistence: float = 0.4) -> RedLightParameters:
    return RedLightParameters(
        min_persistence_seconds=persistence, persistence_status=ParameterStatus.PROVISIONAL
    )


def _reason(
    crossing: CrossingDerivation,
    signal: list[SignalStateObservation],
    *,
    persistence: float = 0.4,
) -> tuple[object, ...]:
    reasoner = RedLightReasoner(RuleEngine(), _params(persistence))
    return reasoner.run_join(crossing, signal)


# --- the latch ---------------------------------------------------------------------
def test_a_light_turning_green_after_entry_does_not_un_commit_the_violation() -> None:
    # THE test for this milestone. The naive `in_junction AND signal_red` predicate
    # loses exactly this vehicle -- the one that entered latest in the red phase,
    # which is the one enforcement most cares about.
    observations = [
        _crossing(0.0, inside=False),  # approaching
        _crossing(1.0, inside=False),  # crosses the stop line here
        _crossing(1.4, inside=True),  # reaches the junction
        _crossing(1.8, inside=True),
        _crossing(2.2, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations),
        frozenset(),
        frozenset({observations[1].observation_id}),  # forward crossing at t=1.0
    )
    signal = [
        _signal(0.0, SignalState.RED),
        _signal(1.0, SignalState.RED),  # red at the stop line
        _signal(1.4, SignalState.GREEN),  # ...and green a moment later
        _signal(1.8, SignalState.GREEN),
        _signal(2.2, SignalState.GREEN),
    ]

    events = _reason(crossing, signal)

    assert len(events) == 1


def test_the_signal_is_read_at_the_stop_line_not_at_the_junction() -> None:
    # A stop line and the junction it guards are not contiguous. Reading the signal
    # at polygon entry would exonerate a vehicle that crossed on red and arrived
    # after the change -- the same class of vehicle the naive predicate loses.
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),  # crossing instant: RED
        _crossing(2.0, inside=True),  # polygon entry: GREEN by now
        _crossing(2.5, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )
    signal = [
        _signal(0.0, SignalState.RED),
        _signal(1.0, SignalState.RED),
        _signal(2.0, SignalState.GREEN),
        _signal(2.5, SignalState.GREEN),
    ]

    steps, _ = join_entry_on_red(crossing, signal)
    entered = [s for s in steps if s.entered_on_red]

    assert entered, "the crossing was on red; the later green must not exonerate it"
    assert entered[0].entry_state is SignalState.RED
    assert len(_reason(crossing, signal)) == 1


def test_leaving_the_junction_clears_the_latch() -> None:
    # The episode ends when the vehicle is out. A second entry must be judged on its
    # own signal state, not on the first entry's.
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False, suffix="-x1"),  # crossing 1: RED
        _crossing(1.4, inside=True),
        _crossing(2.0, inside=False),  # left the junction
        _crossing(3.0, inside=False, suffix="-x2"),  # crossing 2: GREEN
        _crossing(3.4, inside=True),
        _crossing(3.8, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations),
        frozenset(),
        frozenset({observations[1].observation_id, observations[4].observation_id}),
    )
    signal = [
        _signal(0.0, SignalState.RED),
        _signal(1.0, SignalState.RED),
        _signal(1.4, SignalState.RED),
        _signal(2.0, SignalState.GREEN),
        _signal(3.0, SignalState.GREEN),
        _signal(3.4, SignalState.GREEN),
        _signal(3.8, SignalState.GREEN),
    ]

    steps, _ = join_entry_on_red(crossing, signal)
    active_at = [s.observation.timestamp for s in steps if s.entered_on_red]

    assert _at(1.4) in active_at  # first entry, on red
    assert _at(3.4) not in active_at  # second entry, on green
    assert _at(3.8) not in active_at


# --- what must NOT confirm ----------------------------------------------------------
@pytest.mark.parametrize(
    "state", [SignalState.GREEN, SignalState.AMBER, SignalState.UNKNOWN, SignalState.OFF]
)
def test_only_red_confirms(state: SignalState) -> None:
    # Amber is deliberately excluded: "stop if safe" is a judgement about stopping
    # distance that no geometry here can make. UNKNOWN/OFF are absence of evidence.
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.4, inside=True),
        _crossing(2.0, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )
    signal = [_signal(o.timestamp.timestamp(), state) for o in observations]

    assert _reason(crossing, signal) == ()


def test_a_vehicle_already_inside_the_junction_never_confirms() -> None:
    # No validated forward crossing, so `is_inside` is structurally False for the
    # whole track (the crossing derivation's guarantee). Nothing to latch.
    observations = [_crossing(t, inside=False) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    crossing = CrossingDerivation(tuple(observations), frozenset(), frozenset())
    signal = [_signal(t, SignalState.RED) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]

    assert _reason(crossing, signal) == ()


def test_a_missing_signal_fact_resolves_to_unknown_not_to_red() -> None:
    # The conservative direction. A gap in the schedule must never be read as a
    # violation; it is the absence of evidence.
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.4, inside=True),
        _crossing(2.0, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )

    steps, _ = join_entry_on_red(crossing, [])  # no signal stream at all

    assert all(not s.entered_on_red for s in steps)
    assert all(s.entry_state is SignalState.UNKNOWN for s in steps)


def test_a_single_frame_inside_does_not_confirm() -> None:
    # The debounce. Confirmation structurally requires the run to outlast
    # min_persistence, which is what stops boundary jitter minting an event.
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.1, inside=True),  # one frame inside
        _crossing(1.2, inside=False),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )
    signal = [_signal(t, SignalState.RED) for t in (0.0, 1.0, 1.1, 1.2)]

    assert _reason(crossing, signal, persistence=0.5) == ()


# --- taint ---------------------------------------------------------------------------
def test_a_taint_restart_clears_the_latch_and_breaks_the_run() -> None:
    # An ID switch means the entry may belong to a different vehicle
    # (architecture-review §13: tainted tracks may abstain but never confirm).
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.4, inside=True),
        _crossing(1.8, inside=True),  # resumes after taint
        _crossing(2.2, inside=True),
        _crossing(2.6, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations),
        frozenset({observations[3].observation_id}),
        frozenset({observations[1].observation_id}),
    )
    signal = [_signal(t, SignalState.RED) for t in (0.0, 1.0, 1.4, 1.8, 2.2, 2.6)]

    steps, restarts = join_entry_on_red(crossing, signal)

    assert observations[3].observation_id in restarts
    after_taint = [s for s in steps if s.observation.timestamp >= _at(1.8)]
    assert all(not s.entered_on_red for s in after_taint), (
        "support must not survive an ID-switch discontinuity"
    )
    assert _reason(crossing, signal) == ()


# --- multi-track -----------------------------------------------------------------------
def test_tracks_are_latched_independently() -> None:
    red_track = [
        _crossing(0.0, inside=False, track_id="t-red"),
        _crossing(1.0, inside=False, track_id="t-red"),
        _crossing(1.4, inside=True, track_id="t-red"),
        _crossing(2.0, inside=True, track_id="t-red"),
    ]
    green_track = [
        _crossing(3.0, inside=False, track_id="t-green"),
        _crossing(3.5, inside=False, track_id="t-green"),
        _crossing(3.9, inside=True, track_id="t-green"),
        _crossing(4.5, inside=True, track_id="t-green"),
    ]
    crossing = CrossingDerivation(
        tuple(red_track + green_track),
        frozenset(),
        frozenset({red_track[1].observation_id, green_track[1].observation_id}),
    )
    signal = [
        *(_signal(t, SignalState.RED) for t in (0.0, 1.0, 1.4, 2.0)),
        *(_signal(t, SignalState.GREEN) for t in (3.0, 3.5, 3.9, 4.5)),
    ]

    events = _reason(crossing, signal)

    assert len(events) == 1
    assert events[0].track_ids == ("t-red",)  # type: ignore[attr-defined]


# --- the confirmed event ------------------------------------------------------------
def test_the_event_records_the_latched_state_and_the_threshold() -> None:
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.4, inside=True),
        _crossing(2.0, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )
    signal = [_signal(t, SignalState.RED) for t in (0.0, 1.0, 1.4, 2.0)]

    event = _reason(crossing, signal)[0]

    names = {m.name: m.value for m in event.measurements}  # type: ignore[attr-defined]
    assert "persistence_seconds" in names
    # The latched state travels on the event so a reviewer can audit what the system
    # was told the signal was, rather than inferring it.
    assert names["signal_state_at_entry"] == 4.0  # SignalState.RED ordinal
    thresholds = {t.name: t.value for t in event.thresholds}  # type: ignore[attr-defined]
    assert thresholds["min_persistence"] == 0.4
    assert event.violation_type.value == "red_light_jumping"  # type: ignore[attr-defined]


def test_reasoning_is_independent_of_input_order() -> None:
    observations = [
        _crossing(0.0, inside=False),
        _crossing(1.0, inside=False),
        _crossing(1.4, inside=True),
        _crossing(2.0, inside=True),
    ]
    crossing = CrossingDerivation(
        tuple(observations), frozenset(), frozenset({observations[1].observation_id})
    )
    shuffled = CrossingDerivation(
        tuple(reversed(observations)), frozenset(), crossing.forward_crossing_ids
    )
    signal = [_signal(t, SignalState.RED) for t in (0.0, 1.0, 1.4, 2.0)]

    forward = _reason(crossing, signal)
    backward = _reason(shuffled, signal)

    assert [e.event_id for e in forward] == [e.event_id for e in backward]  # type: ignore[attr-defined]


# --- scene parameters -----------------------------------------------------------------
def test_parameters_fail_fast_on_a_scene_without_a_red_light_block() -> None:
    # A scene with no junction geometry declares no red-light block, and reasoning
    # must refuse it rather than default a threshold and appear to work.
    from trafficpulse.contracts.scene import ZoneType
    from trafficpulse.scenes import SceneDraft, ZoneDraft, build_scene, full_frame_polygon

    scene = build_scene(
        SceneDraft(
            scene_name="No junction",
            camera_id="cam-1",
            frame_width=640,
            frame_height=480,
            zones=(
                ZoneDraft(
                    zone_id="zone-lane",
                    zone_type=ZoneType.LANE,
                    polygon=full_frame_polygon(640, 480),
                ),
            ),
        ),
        scene_id="scene-1",
    )

    with pytest.raises(ValueError, match="red_light_jumping"):
        red_light_parameters(scene)


def test_parameters_are_read_from_a_scene_that_declares_the_block() -> None:
    from trafficpulse.contracts.scene import ZoneType
    from trafficpulse.scenes import (
        SceneDraft,
        SignalGroupDraft,
        StopLineDraft,
        ZoneDraft,
        build_scene,
        full_frame_polygon,
    )

    scene = build_scene(
        SceneDraft(
            scene_name="Junction",
            camera_id="cam-1",
            frame_width=640,
            frame_height=480,
            zones=(
                ZoneDraft(
                    zone_id="zone-lane",
                    zone_type=ZoneType.LANE,
                    polygon=full_frame_polygon(640, 480),
                ),
                ZoneDraft(
                    zone_id="zone-junction",
                    zone_type=ZoneType.INTERSECTION,
                    polygon=((100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)),
                ),
            ),
            stop_lines=(
                StopLineDraft(
                    stop_line_id="sl-1",
                    a=(100.0, 90.0),
                    b=(400.0, 90.0),
                    crossing_dx=0.0,
                    crossing_dy=1.0,
                    signal_group_id="sg-1",
                    zone_ids=("zone-junction",),
                ),
            ),
            signal_groups=(
                SignalGroupDraft(
                    signal_group_id="sg-1",
                    roi_polygon=((10.0, 10.0), (60.0, 10.0), (60.0, 90.0)),
                    zone_ids=("zone-junction",),
                ),
            ),
        ),
        scene_id="scene-1",
    )

    params = red_light_parameters(scene)

    assert params.min_persistence_seconds > 0.0
    assert params.persistence_status is ParameterStatus.PROVISIONAL

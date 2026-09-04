"""The controlled demonstration scenario: geometry, determinism, and non-interference.

The scenario is only useful if its four situations are genuinely independent -- if
the illegally stopped car could also read as wrong-way, "four reasoners agreed" is
not a demonstration of anything. These tests check that separation **geometrically**,
against the same polygons the scene declares, so a future edit that nudges an actor
into the wrong zone fails here rather than silently weakening the demo.

Fast by construction: pure geometry over the specification. Nothing decodes a clip
or runs an engine; the end-to-end proof lives in ``tests/app/test_app_controlled_demo``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trafficpulse.contracts.enums import ObjectClass, SignalState, ViolationType
from trafficpulse.contracts.scene import ZoneType, scene_config_hash
from trafficpulse.geometry import point_in_polygon
from trafficpulse.scenes import demo_scenario as scenario


def _ground_contact(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """The bbox bottom-center -- the reference every zone derivation uses."""

    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _actor(actor_id: str) -> scenario.DemoActor:
    return next(a for a in scenario.demo_actors() if a.actor_id == actor_id)


def _inside(polygon: tuple[tuple[float, float], ...], point: tuple[float, float]) -> bool:
    return point_in_polygon(point, polygon)


# --- the specification ------------------------------------------------------------
def test_the_scenario_declares_the_four_families_it_was_built_for() -> None:
    assert set(scenario.DEMO_EXPECTED_VIOLATIONS) == {
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.RED_LIGHT_JUMPING,
        ViolationType.TRIPLE_RIDING,
    }


def test_the_draft_expands_into_a_valid_scene_supporting_all_four() -> None:
    from trafficpulse.app.capabilities import supported_violations

    scene = scenario.demo_scene()
    supported = supported_violations(scene, no_helmet_available=False)
    assert set(scenario.DEMO_EXPECTED_VIOLATIONS) <= set(supported)


def test_the_scene_is_deterministic_so_its_hash_is_stable() -> None:
    """Content-addressed storage and content-derived event ids both depend on this."""

    assert scene_config_hash(scenario.demo_scene()) == scene_config_hash(
        scenario.demo_scene()
    )


def test_the_operator_dwell_threshold_is_carried_not_defaulted() -> None:
    """The declared 3.0 s reaches the scene rather than the builder's own default."""

    from trafficpulse.scenes.builder import DEFAULT_STATIONARY_DURATION

    blocks = {b.violation_type: b for b in scenario.demo_scene().rule_parameters}
    dwell = next(
        p
        for p in blocks[ViolationType.ILLEGAL_STOPPING].parameters
        if p.id == "stationary_duration"
    )
    assert dwell.value == pytest.approx(scenario.DEMO_STATIONARY_DURATION_S)
    assert dwell.value != DEFAULT_STATIONARY_DURATION


def test_the_scene_declares_itself_a_controlled_demonstration() -> None:
    """The honesty record travels with the scene, not only with the UI that drew it."""

    scene = scenario.demo_scene()
    assert "Controlled demonstration" in scene.scene.description
    assert scene.scene.status.value == "draft"
    # No world calibration was solved, so none is claimed.
    assert scene.calibration.type.value == "none"


def test_the_signal_turns_green_after_the_red_light_actor_has_crossed() -> None:
    """Otherwise the run would demonstrate a convenient signal, not the H13 latch."""

    schedule = dict(scenario.DEMO_SIGNAL_SCHEDULE)
    assert schedule[0.0] is SignalState.RED
    green_at = next(at for at, state in scenario.DEMO_SIGNAL_SCHEDULE if state is SignalState.GREEN)

    runner = _actor("rl-runner")
    crossing_frame = next(
        index
        for index, box in enumerate(runner.boxes)
        if _ground_contact(box)[1] >= scenario.STOP_LINE_A[1]
    )
    assert crossing_frame / scenario.DEMO_FPS < green_at


# --- non-interference --------------------------------------------------------------
def test_the_wrong_way_actor_never_registers_a_forward_stop_line_crossing() -> None:
    """Climbing the lane, its crossing is backward -- it cannot become a red-light event."""

    driver = _actor("ww-driver")
    ys = [_ground_contact(box)[1] for box in driver.boxes]
    assert ys == sorted(ys, reverse=True), "the wrong-way actor must travel upward only"


def test_the_illegal_stopping_actor_stays_out_of_the_governed_lane() -> None:
    """No heading is derived outside the lane, so wrong-way cannot see it."""

    stopper = _actor("is-stopper")
    assert not any(
        _inside(scenario.LANE_POLYGON, _ground_contact(box)) for box in stopper.boxes
    )


def test_the_illegal_stopping_actor_reaches_and_holds_inside_the_no_stopping_zone() -> None:
    stopper = _actor("is-stopper")
    inside = [
        index
        for index, box in enumerate(stopper.boxes)
        if _inside(scenario.NO_STOPPING_POLYGON, _ground_contact(box))
    ]
    assert inside, "the stopper never enters the zone it is supposed to stop in"

    held = {_ground_contact(stopper.boxes[i]) for i in inside[-10:]}
    assert len(held) == 1, "the stopper must be genuinely stationary at the end"


def test_the_illegal_stopping_actor_is_nowhere_near_the_junction() -> None:
    """Stopping at a signal must never be confusable with stopping where prohibited."""

    stopper = _actor("is-stopper")
    assert not any(
        _inside(scenario.JUNCTION_POLYGON, _ground_contact(box)) for box in stopper.boxes
    )


def test_the_motorcycle_stays_outside_every_governed_polygon() -> None:
    """Triple riding is geometry-free; the bike must be invisible to the other three."""

    bike = _actor("tr-motorcycle")
    for box in bike.boxes:
        point = _ground_contact(box)
        assert not _inside(scenario.LANE_POLYGON, point)
        assert not _inside(scenario.NO_STOPPING_POLYGON, point)
        assert not _inside(scenario.JUNCTION_POLYGON, point)


def test_the_red_light_actor_travels_with_the_declared_legal_direction() -> None:
    """It jumps a signal; it is not also driving the wrong way."""

    runner = _actor("rl-runner")
    ys = [_ground_contact(box)[1] for box in runner.boxes]
    assert ys == sorted(ys), "the red-light actor must travel with the legal direction"
    assert scenario.LEGAL_DY > 0


def test_the_motorcycle_carries_exactly_three_riders_overlapping_it() -> None:
    """Each rider must clear the association overlap, or the count is not three."""

    from trafficpulse.association.riders import DEFAULT_MIN_OVERLAP

    riders = [a for a in scenario.demo_actors() if a.object_class is ObjectClass.PERSON]
    assert len(riders) == scenario.DEMO_RIDER_COUNT

    bike_box = _actor("tr-motorcycle").boxes[0]
    for rider in riders:
        assert _io_min(bike_box, rider.boxes[0]) >= DEFAULT_MIN_OVERLAP


def _io_min(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the smaller area -- the association policy's own measure."""

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    areas = ((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / min(areas)


# --- the actors and the clip -------------------------------------------------------
def test_every_box_stays_inside_the_declared_frame() -> None:
    """An out-of-frame box is rejected by the detector adapter, not silently clipped."""

    for actor in scenario.demo_actors():
        for x1, y1, x2, y2 in actor.boxes:
            assert 0 <= x1 < x2 <= scenario.DEMO_WIDTH
            assert 0 <= y1 < y2 <= scenario.DEMO_HEIGHT


def test_actors_are_deterministic() -> None:
    assert scenario.demo_actors() == scenario.demo_actors()


def test_an_actor_that_has_left_the_frame_reports_no_box() -> None:
    """A departed vehicle stops being detected; it does not freeze in place."""

    runner = _actor("rl-runner")
    assert runner.box_at(len(runner.boxes)) is None
    assert runner.box_at(-1) is None


def test_the_zone_types_are_what_each_rule_looks_for() -> None:
    zones = {z.zone_id: z.zone_type for z in scenario.demo_scene().zones}
    assert zones[scenario.LANE_ZONE_ID] is ZoneType.LANE
    assert zones[scenario.NO_STOPPING_ZONE_ID] is ZoneType.NO_STOPPING
    assert zones[scenario.JUNCTION_ZONE_ID] is ZoneType.INTERSECTION


def test_the_clip_renders_to_a_decodable_video(tmp_path: Path) -> None:
    """The one I/O function, checked against the project's own ingestion layer."""

    from trafficpulse.ingestion import open_video

    path = scenario.render_demo_clip(tmp_path / "demo.mp4", frames=12)
    assert path.is_file() and path.stat().st_size > 0

    with open_video(path) as reader:
        frames = list(reader)
    assert len(frames) == 12
    assert frames[0].image.shape[:2] == (scenario.DEMO_HEIGHT, scenario.DEMO_WIDTH)


# --- two renderings, one geometry ---------------------------------------------------
def test_scaling_multiplies_every_coordinate_and_nothing_else() -> None:
    """The composited rendering needs a bigger canvas; it must not need a second scenario.

    A detector cannot resolve a 40x30 vehicle, so the real-pixel clip is rendered at
    4x. If scaling drifted from a pure multiplication, the browser demo and the test
    suite would be exercising different geometry while claiming to be one scenario.
    """

    plain = scenario.demo_actors()
    scaled = scenario.demo_actors(scale=4)

    assert len(plain) == len(scaled)
    for one, four in zip(plain, scaled, strict=True):
        assert one.actor_id == four.actor_id
        assert len(one.boxes) == len(four.boxes)
        for box, big in zip(one.boxes, four.boxes, strict=True):
            assert big == tuple(v * 4 for v in box)


def test_a_scaled_scene_is_still_valid_and_supports_the_same_families() -> None:
    from trafficpulse.app.capabilities import supported_violations

    scene = scenario.demo_scene(scale=4)
    assert (scene.frame.reference_width, scene.frame.reference_height) == (
        scenario.DEMO_WIDTH * 4,
        scenario.DEMO_HEIGHT * 4,
    )
    assert set(scenario.DEMO_EXPECTED_VIOLATIONS) <= set(
        supported_violations(scene, no_helmet_available=False)
    )


def test_a_scaled_actor_stays_in_the_same_zone_as_its_unscaled_self() -> None:
    """Scaling geometry and actors together must preserve every containment fact."""

    scaled_zones = {
        zone.zone_id: zone.polygon for zone in scenario.demo_scene(scale=4).zones
    }
    stopper = next(a for a in scenario.demo_actors(scale=4) if a.actor_id == "is-stopper")

    assert any(
        point_in_polygon(_ground_contact(box), scaled_zones[scenario.NO_STOPPING_ZONE_ID])
        for box in stopper.boxes
    )
    assert not any(
        point_in_polygon(_ground_contact(box), scaled_zones[scenario.LANE_ZONE_ID])
        for box in stopper.boxes
    )


def test_a_scaled_clip_renders_at_the_scaled_size(tmp_path: Path) -> None:
    from trafficpulse.ingestion import open_video

    path = scenario.render_demo_clip(tmp_path / "big.mp4", frames=6, scale=2)
    with open_video(path) as reader:
        frames = list(reader)
    assert frames[0].image.shape[:2] == (scenario.DEMO_HEIGHT * 2, scenario.DEMO_WIDTH * 2)

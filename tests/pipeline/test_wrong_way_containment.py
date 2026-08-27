"""Wrong-way reasoning is confined to its governing lane (architecture-review 5a).

The rule compares a track's heading to *a particular lane's* legal direction, so it
may only reason about traffic in that lane. Before containment the direction was
applied to every track in frame, and lawful traffic on an opposing carriageway --
which moves at ~180 deg to the declared direction by construction -- was confirmed
as a violation. On a real divided-road camera that is a false-positive generator,
which is how it was found: a validation run over elevated highway CCTV confirmed
two "violations", both vehicles on the far carriageway, while the compliant
near-carriageway tracks were correctly cleared.

These are end-to-end pipeline tests rather than derivation unit tests (those live in
``tests/observations/test_heading.py``): they assert the wiring actually reaches the
gate, because the defect was not in the geometry -- it was that nobody passed the
polygon in. They also cover the scene-resolution guards that stop a misconfigured
scene from silently reasoning over the whole frame instead.
"""

from __future__ import annotations

import pytest
from _pipeline_helpers import (
    CAMERA,
    DEFAULT_FRAME_COUNT,
    DETECTOR_CONFIG,
    LANE_X,
    LANE_Y0_UP,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_down_detector,
    moving_raw,
)
from pydantic import ValidationError

from trafficpulse.contracts import SceneConfig
from trafficpulse.detector import StubDetector
from trafficpulse.pipeline import SceneConfigurationError, WrongWayPipeline
from trafficpulse.tracking import ScriptedAssignment, StubTracker

# Far outside SCENE's governing lane (x 100..220): a second carriageway the camera
# happens to see. Same motion as the wrong-way track, different road.
OPPOSING_CARRIAGEWAY_X = 600.0


def _frames(count: int = DEFAULT_FRAME_COUNT):  # type: ignore[no-untyped-def]
    return [make_frame_record(i) for i in range(count)]


def _script(track_id: str = "only"):  # type: ignore[no-untyped-def]
    return {i: (ScriptedAssignment(track_id=track_id),) for i in range(DEFAULT_FRAME_COUNT)}


def _pipeline(detector, scene: SceneConfig = SCENE):  # type: ignore[no-untyped-def]
    return WrongWayPipeline(
        detector=detector,
        tracker=StubTracker(_script()),
        scene=scene,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )


def _scene_with_lane(*, enabled: bool = True, margin: float | None = None) -> SceneConfig:
    """SCENE with ``zone-lane-north`` toggled and/or an abstain margin declared."""

    raw = SCENE.model_dump(mode="json")
    for zone in raw["zones"]:
        if zone["zone_id"] == "zone-lane-north":
            zone["enabled"] = enabled
    if margin is not None:
        for block in raw["rule_parameters"]:
            if block["violation_type"] == "wrong_way":
                block["parameters"].append(
                    {
                        "id": "boundary_abstain_margin",
                        "value": margin,
                        # ParameterUnit has no pixel member; a pixel count is a count.
                        "unit": "count",
                        "status": "provisional",
                        "note": "test fixture",
                    }
                )
    return SceneConfig.model_validate(raw)


# --- inside the lane: unchanged behaviour ------------------------------------
def test_a_wrong_way_track_inside_the_governing_lane_still_confirms() -> None:
    events = _pipeline(moving_down_detector()).process(_frames())
    assert len(events) == 1
    assert events[0].track_ids == ("only",)


def test_a_legal_track_inside_the_governing_lane_confirms_nothing() -> None:
    events = _pipeline(moving_down_detector(direction=-1, y0=LANE_Y0_UP)).process(_frames())
    assert events == ()


# --- outside the lane: the defect --------------------------------------------
def test_opposing_carriageway_traffic_is_not_a_violation_of_this_lane() -> None:
    # Identical motion to the confirming case, moved to the other carriageway.
    detector = moving_down_detector(x=OPPOSING_CARRIAGEWAY_X)
    assert _pipeline(detector).process(_frames()) == ()


def test_the_same_track_confirms_in_lane_and_is_ignored_out_of_lane() -> None:
    # The two runs differ in one thing only: where the vehicle was driving. That
    # isolates containment from every other reason a run might not confirm.
    in_lane = _pipeline(moving_down_detector(x=LANE_X)).process(_frames())
    out_of_lane = _pipeline(moving_down_detector(x=OPPOSING_CARRIAGEWAY_X)).process(_frames())
    assert len(in_lane) == 1
    assert out_of_lane == ()


def test_only_the_in_lane_vehicle_of_a_mixed_frame_is_confirmed() -> None:
    # Both carriageways in every frame, tracked separately -- the realistic shape of
    # the footage this was found on. Exactly one of them is this lane's traffic.
    per_frame = {
        i: (moving_raw(i, x=LANE_X), moving_raw(i, x=OPPOSING_CARRIAGEWAY_X))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    script = {
        i: (ScriptedAssignment(track_id="in-lane"), ScriptedAssignment(track_id="other-road"))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    pipeline = WrongWayPipeline(
        detector=StubDetector(per_frame=per_frame),
        tracker=StubTracker(script),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )
    events = pipeline.process(_frames())
    assert {tid for e in events for tid in e.track_ids} == {"in-lane"}


# --- the lane boundary --------------------------------------------------------
def test_a_track_hugging_the_lane_boundary_abstains() -> None:
    # The lane is x 100..220; a 20-wide box at x=100 has its center 20 px from the
    # edge. Under a 40 px abstain band it is inside, but not confidently so.
    widened = _scene_with_lane(margin=40.0)
    assert _pipeline(moving_down_detector(x=100.0), widened).process(_frames()) == ()


def test_the_boundary_band_does_not_silence_the_lane_interior() -> None:
    # The same widened band leaves a mid-lane track confirming: the band abstains
    # near the edge, it does not shrink the usable lane.
    widened = _scene_with_lane(margin=40.0)
    assert len(_pipeline(moving_down_detector(x=LANE_X), widened).process(_frames())) == 1


def test_the_scene_margin_is_optional_and_defaults_rather_than_failing() -> None:
    # SCENE declares no boundary_abstain_margin, so scenes written before lane
    # containment existed must keep loading -- and keep confirming.
    assert not any(
        parameter.id == "boundary_abstain_margin"
        for block in SCENE.rule_parameters
        if block.violation_type.value == "wrong_way"
        for parameter in block.parameters
    )
    assert len(_pipeline(moving_down_detector()).process(_frames())) == 1


# --- scene-resolution guards --------------------------------------------------
def test_the_contract_itself_refuses_a_direction_governing_an_undeclared_zone() -> None:
    # Containment needs the governing zone's polygon, so a direction naming a zone
    # that does not exist must never reach the pipeline. It cannot: SceneConfig
    # resolves the cross-reference first. The pipeline keeps a defensive guard for
    # callers that build a strategy from a hand-assembled scene, but this is the
    # boundary that actually holds the invariant, so this is where it is asserted.
    raw = SCENE.model_dump(mode="json")
    raw["zones"] = [z for z in raw["zones"] if z["zone_id"] != "zone-lane-north"]
    with pytest.raises(ValidationError, match="unresolved references"):
        SceneConfig.model_validate(raw)


def test_a_direction_governing_a_disabled_zone_fails_fast() -> None:
    with pytest.raises(SceneConfigurationError, match="disabled"):
        _pipeline(moving_down_detector(), _scene_with_lane(enabled=False))


def test_the_resolved_lane_id_still_names_the_governing_zone() -> None:
    assert _pipeline(moving_down_detector()).lane_id == "zone-lane-north"


def test_events_are_still_attributed_to_the_governing_camera() -> None:
    events = _pipeline(moving_down_detector()).process(_frames())
    assert events[0].camera_id == CAMERA

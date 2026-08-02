"""Authoring a validated ``SceneConfig`` from an analyst's drawing (H12).

The builder's contract: it fills in everything that is bookkeeping, claims only
what was actually drawn, and is deterministic -- the same drawing must always
produce the same scene, because the scene's hash is its storage address and is
stamped into every event reasoned under it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trafficpulse.contracts.enums import ViolationType
from trafficpulse.contracts.scene import (
    CalibrationStatus,
    CalibrationType,
    ParameterStatus,
    SceneStatus,
    ZoneType,
    scene_config_hash,
)
from trafficpulse.scenes import (
    DirectionDraft,
    RuleTuning,
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)

WIDTH, HEIGHT = 1280, 720


def _lane(zone_id: str = "zone-lane") -> ZoneDraft:
    return ZoneDraft(
        zone_id=zone_id, zone_type=ZoneType.LANE, polygon=full_frame_polygon(WIDTH, HEIGHT)
    )


def _draft(**overrides: object) -> SceneDraft:
    base: dict[str, object] = {
        "scene_name": "Junction North",
        "camera_id": "cam-1",
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "zones": (_lane(),),
    }
    base.update(overrides)
    return SceneDraft.model_validate(base)


# --- determinism -------------------------------------------------------------------
def test_the_same_drawing_always_produces_the_same_scene() -> None:
    # The property the content-addressed store rests on: if this were not true,
    # re-saving an unchanged calibration would mint a new revision every time and
    # an event's scene_config_hash would name a scene nobody could reproduce.
    draft = _draft(direction=DirectionDraft(dx=0.0, dy=-1.0, zone_id="zone-lane"))

    first = build_scene(draft, scene_id="scene-x")
    second = build_scene(draft, scene_id="scene-x")

    assert first == second
    assert scene_config_hash(first) == scene_config_hash(second)


def test_the_scene_carries_no_wall_clock_instant() -> None:
    # Baking "now" into scene content would make two identical drawings hash
    # differently, which is exactly what determinism forbids.
    scene = build_scene(_draft(), scene_id="scene-x")

    assert scene.scene.created_at.year == 1970
    assert scene.scene.updated_at == scene.scene.created_at
    assert scene.calibration.created_at == scene.scene.created_at


def test_a_changed_drawing_produces_a_different_revision() -> None:
    lane = build_scene(_draft(), scene_id="scene-x")
    moved = build_scene(
        _draft(
            zones=(
                ZoneDraft(
                    zone_id="zone-lane",
                    zone_type=ZoneType.LANE,
                    polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0)),
                ),
            )
        ),
        scene_id="scene-x",
    )

    assert scene_config_hash(lane) != scene_config_hash(moved)


# --- what is claimed ---------------------------------------------------------------
def test_an_authored_scene_claims_no_verification_and_no_metric_calibration() -> None:
    # An analyst drew polygons on a video. That is image-space geometry, not a
    # surveyed deployment calibration, and the scene must not imply otherwise.
    scene = build_scene(_draft(), scene_id="scene-x")

    assert scene.scene.status is SceneStatus.DRAFT
    assert scene.calibration.type is CalibrationType.NONE
    assert scene.calibration.status is CalibrationStatus.ABSENT
    assert scene.calibration.homography_matrix is None
    assert scene.scene.provenance.origin == "analyst_calibration"
    assert scene.scene.provenance.synthetic is False


def test_every_rule_parameter_is_marked_provisional() -> None:
    scene = build_scene(_draft(), scene_id="scene-x")

    statuses = {
        parameter.status
        for block in scene.rule_parameters
        for parameter in block.parameters
    }
    assert statuses == {ParameterStatus.PROVISIONAL}


# --- geometry -> capability --------------------------------------------------------
def test_a_bare_lane_scene_declares_only_the_geometry_free_rules() -> None:
    # No direction and no no-stopping zone: advertising a wrong_way block here
    # would promise a rule that fails fast the moment it is configured.
    scene = build_scene(_draft(), scene_id="scene-x")

    declared = {block.violation_type for block in scene.rule_parameters}
    assert declared == {ViolationType.NO_HELMET, ViolationType.TRIPLE_RIDING}


def test_drawing_a_direction_declares_the_wrong_way_block_and_wires_the_lane() -> None:
    scene = build_scene(
        _draft(direction=DirectionDraft(dx=0.0, dy=-1.0, zone_id="zone-lane")),
        scene_id="scene-x",
    )

    assert scene.legal_directions[0].vector.dy == -1.0
    assert scene.legal_directions[0].zone_ids == ("zone-lane",)
    # The lane points back at the direction, which is the link wrong-way resolves.
    assert scene.zones[0].legal_direction_id == scene.legal_directions[0].direction_id
    assert ViolationType.WRONG_WAY in scene.zones[0].applicable_violations
    assert "heading_vs_lane" in scene.zones[0].observation_consumers
    assert any(b.violation_type is ViolationType.WRONG_WAY for b in scene.rule_parameters)


def test_drawing_a_no_stopping_zone_declares_the_illegal_stopping_block() -> None:
    scene = build_scene(
        _draft(
            zones=(
                _lane(),
                ZoneDraft(
                    zone_id="zone-nostop",
                    zone_type=ZoneType.NO_STOPPING,
                    polygon=((10.0, 10.0), (200.0, 10.0), (200.0, 200.0), (10.0, 200.0)),
                ),
            )
        ),
        scene_id="scene-x",
    )

    stopping = next(
        b for b in scene.rule_parameters if b.violation_type is ViolationType.ILLEGAL_STOPPING
    )
    assert {p.id for p in stopping.parameters} == {"stationary_duration"}
    nostop = next(z for z in scene.zones if z.zone_id == "zone-nostop")
    assert nostop.enabled
    assert set(nostop.observation_consumers) == {"in_zone", "stationary"}


def test_tuning_overrides_reach_the_parameter_block() -> None:
    scene = build_scene(
        _draft(
            direction=DirectionDraft(dx=1.0, dy=0.0, zone_id="zone-lane"),
            tuning=RuleTuning(heading_deviation_max_degrees=150.0),
        ),
        scene_id="scene-x",
    )

    block = next(b for b in scene.rule_parameters if b.violation_type is ViolationType.WRONG_WAY)
    deviation = next(p for p in block.parameters if p.id == "heading_deviation_max")
    assert deviation.value == 150.0
    # An omitted knob keeps the provisional default rather than becoming null.
    persistence = next(p for p in block.parameters if p.id == "min_persistence")
    assert persistence.value == 1.0


# --- H13 readiness -----------------------------------------------------------------
def test_stop_lines_and_signal_groups_are_authorable_and_cross_referenced() -> None:
    # Red-light jumping needs exactly this geometry. It has to be expressible now,
    # or H13 becomes a scene redesign rather than a reasoner.
    scene = build_scene(
        _draft(
            zones=(
                _lane(),
                ZoneDraft(
                    zone_id="zone-junction",
                    zone_type=ZoneType.INTERSECTION,
                    polygon=((300.0, 300.0), (600.0, 300.0), (600.0, 600.0), (300.0, 600.0)),
                ),
            ),
            stop_lines=(
                StopLineDraft(
                    stop_line_id="sl-1",
                    a=(300.0, 280.0),
                    b=(600.0, 280.0),
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
        scene_id="scene-x",
    )

    assert scene.stop_lines[0].signal_group_id == "sg-1"
    # The group's back-reference is derived, not asked for -- so the two halves of
    # the linkage cannot be authored inconsistently.
    assert scene.signal_groups[0].stop_line_ids == ("sl-1",)
    junction = next(z for z in scene.zones if z.zone_id == "zone-junction")
    assert junction.signal_group_id == "sg-1"


# --- validation is the contract's, not a second rule set ---------------------------
def test_geometry_outside_the_frame_is_refused_by_the_contract() -> None:
    with pytest.raises(ValidationError):
        build_scene(
            _draft(
                zones=(
                    ZoneDraft(
                        zone_id="zone-lane",
                        zone_type=ZoneType.LANE,
                        polygon=((0.0, 0.0), (WIDTH + 500.0, 0.0), (10.0, 10.0)),
                    ),
                )
            ),
            scene_id="scene-x",
        )


def test_a_draft_is_refused_before_it_can_build_an_invalid_scene() -> None:
    with pytest.raises(ValidationError):
        _draft(zones=())  # a scene needs a zone
    with pytest.raises(ValidationError):
        _draft(direction=DirectionDraft(dx=0.0, dy=-1.0, zone_id="zone-missing"))
    with pytest.raises(ValidationError):
        ZoneDraft(zone_id="z", zone_type=ZoneType.LANE, polygon=((0.0, 0.0), (1.0, 1.0)))
    with pytest.raises(ValidationError):
        DirectionDraft(dx=0.0, dy=0.0, zone_id="z")


def test_duplicate_zone_ids_are_refused() -> None:
    with pytest.raises(ValidationError):
        _draft(zones=(_lane(), _lane()))


def test_full_frame_polygon_is_a_closed_ring_in_bounds() -> None:
    ring = full_frame_polygon(WIDTH, HEIGHT)

    assert len(ring) == 4
    assert all(0 <= x <= WIDTH and 0 <= y <= HEIGHT for x, y in ring)

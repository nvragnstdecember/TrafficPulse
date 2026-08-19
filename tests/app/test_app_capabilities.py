"""Scene capability probing and default-rule derivation (R1 + R5).

Two claims, tested together because they share one probe:

* **R1** -- a scene that declares several approaches supports the governed rules
  (wrong-way, red-light). The probe asks the real factories about each declared
  direction / junction instead of making one ambiguous call, so a multi-approach
  site is no longer reported as unsupported. Validation is not weakened: a scene
  genuinely missing the geometry or the parameter block still reports unsupported.
* **R5** -- the derived rule set is exactly "shipped AND scene-supported AND
  deployment-configured", carries the resolved selector, and **builds** through the
  engine's own rule registry (which is the only proof that "supported" means
  "runnable").
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from _app_helpers import EXAMPLE_SCENE_PATH

from trafficpulse.app.capabilities import probe_scene, rules_for, supported_violations
from trafficpulse.contracts import SceneConfig
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.contracts.scene import ZoneType
from trafficpulse.engine.rules import build_rules
from trafficpulse.scenes import (
    DirectionDraft,
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)

WIDTH, HEIGHT = 320, 240


def _example_scene() -> SceneConfig:
    """The committed example scene: two legal directions, two junction zones."""

    return SceneConfig.model_validate(
        yaml.safe_load(EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    )


def _authored_scene(**overrides: Any) -> SceneConfig:
    """A single-approach scene as the calibration surface would author it."""

    fields: dict[str, Any] = {
        "scene_name": "Capability junction",
        "camera_id": "cam-capability",
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "zones": (
            ZoneDraft(
                zone_id="zone-lane",
                zone_type=ZoneType.LANE,
                polygon=full_frame_polygon(WIDTH, HEIGHT),
            ),
            ZoneDraft(
                zone_id="zone-junction",
                zone_type=ZoneType.INTERSECTION,
                polygon=((100.0, 150.0), (220.0, 150.0), (220.0, 235.0), (100.0, 235.0)),
            ),
        ),
        "direction": DirectionDraft(direction_id="dir-1", dx=0.0, dy=-1.0, zone_id="zone-lane"),
        "stop_lines": (
            StopLineDraft(
                stop_line_id="sl-1",
                a=(100.0, 120.0),
                b=(220.0, 120.0),
                crossing_dx=0.0,
                crossing_dy=1.0,
                signal_group_id="sg-1",
                zone_ids=("zone-junction",),
            ),
        ),
        "signal_groups": (
            SignalGroupDraft(
                signal_group_id="sg-1",
                roi_polygon=((5.0, 5.0), (45.0, 5.0), (45.0, 60.0)),
                zone_ids=("zone-junction",),
            ),
        ),
    }
    fields.update(overrides)
    return build_scene(SceneDraft(**fields), scene_id="scene-capability")


# --- R1: multi-approach scenes ------------------------------------------------------
def test_a_multi_direction_scene_supports_wrong_way() -> None:
    """The R1 regression: two legal directions no longer read as 'unsupported'.

    The old probe called the factory with no ``direction_id``, which the factory
    rightly refuses as ambiguous -- so every multi-lane site reported wrong-way
    unsupported even though the rule runs fine with an id.
    """

    scene = _example_scene()
    assert len(scene.legal_directions) > 1, "fixture must actually be ambiguous"

    capabilities = probe_scene(scene)
    assert capabilities.wrong_way is True
    # The *first declared* direction, so the answer is deterministic.
    assert capabilities.wrong_way_direction_id == scene.legal_directions[0].direction_id
    assert ViolationType.WRONG_WAY in supported_violations(scene, classifier_available=False)


def test_a_multi_junction_scene_supports_red_light() -> None:
    """The same regression for red-light, which selects a stop line *and* a zone."""

    scene = _example_scene()
    junctions = [
        zone
        for zone in scene.zones
        if zone.enabled
        and zone.zone_type in (ZoneType.INTERSECTION, ZoneType.SIGNAL_CONTROLLED_REGION)
    ]
    assert len(junctions) > 1, "fixture must actually be ambiguous"

    capabilities = probe_scene(scene)
    assert capabilities.red_light is True
    assert capabilities.red_light_selectors is not None
    stop_line_id, zone_id = capabilities.red_light_selectors
    assert stop_line_id in {line.stop_line_id for line in scene.stop_lines}
    assert zone_id in {zone.zone_id for zone in junctions}
    assert ViolationType.RED_LIGHT_JUMPING in supported_violations(
        scene, classifier_available=False
    )


def test_an_unambiguous_scene_still_resolves_its_single_approach() -> None:
    """The single-direction / single-junction case is unchanged by R1."""

    capabilities = probe_scene(_authored_scene())
    assert capabilities.wrong_way_direction_id == "dir-1"
    assert capabilities.red_light_selectors == ("sl-1", "zone-junction")


@pytest.mark.parametrize(
    ("removed", "unsupported"),
    [
        ("directions", ViolationType.WRONG_WAY),
        ("junction", ViolationType.RED_LIGHT_JUMPING),
    ],
)
def test_a_scene_without_the_geometry_is_still_unsupported(
    removed: str, unsupported: ViolationType
) -> None:
    """R1 must not weaken validation: absent geometry is still 'cannot run'."""

    if removed == "directions":
        scene = _authored_scene(direction=None)
    else:
        scene = _authored_scene(stop_lines=(), signal_groups=())

    assert unsupported not in supported_violations(scene, classifier_available=True)
    assert unsupported not in {
        _violation_of(rule) for rule in rules_for(scene, classifier_available=True)
    }


def test_a_scene_with_geometry_but_no_parameter_block_is_unsupported() -> None:
    """Geometry alone is not support: the rule's parameter block is required too."""

    scene = _example_scene().model_copy(update={"rule_parameters": ()})
    capabilities = probe_scene(scene)

    assert capabilities.wrong_way is False
    assert capabilities.red_light is False
    assert capabilities.illegal_stopping is False
    assert capabilities.no_helmet is False
    assert capabilities.triple_riding is False
    assert supported_violations(scene, classifier_available=True) == ()
    assert rules_for(scene, classifier_available=True) == ()


# --- R4: the shipped example scene --------------------------------------------------
def test_the_shipped_example_scene_can_construct_red_light() -> None:
    """R4: the example scene declares ``min_persistence``, so red-light builds.

    Without it the shipped scene declared a ``red_light_jumping`` block that could
    never be used -- the parameter loader refused it and the rule was unreachable
    for the default deployment.
    """

    from trafficpulse.pipeline.red_light import resolve_red_light_geometry
    from trafficpulse.rules.red_light import red_light_parameters

    scene = _example_scene()
    parameters = red_light_parameters(scene)
    assert parameters.min_persistence_seconds > 0.0

    selectors = probe_scene(scene).red_light_selectors
    assert selectors is not None
    stop_line, zone = resolve_red_light_geometry(
        scene, stop_line_id=selectors[0], zone_id=selectors[1]
    )
    assert stop_line.stop_line_id == selectors[0]
    assert zone.zone_id == selectors[1]


# --- R5: derived rule sets ----------------------------------------------------------
def _violation_of(rule: Any) -> ViolationType:
    return ViolationType(rule.kind)


def test_derived_rules_are_shipped_scene_supported_and_deterministic() -> None:
    scene = _example_scene()
    derived = rules_for(scene, classifier_available=True)

    kinds = [rule.kind for rule in derived]
    assert kinds == ["wrong_way", "illegal_stopping", "no_helmet", "triple_riding"]
    # Deterministic: the same scene derives the same set, in the same order.
    assert rules_for(scene, classifier_available=True) == derived
    # `speeding` has no shipped reasoner and can never be derived.
    assert "speeding" not in kinds


def test_derived_rules_omit_no_helmet_without_a_classifier() -> None:
    """Deployment fact, not a scene fact -- so it gates the derived set too."""

    scene = _example_scene()
    assert "no_helmet" not in [
        rule.kind for rule in rules_for(scene, classifier_available=False)
    ]
    assert "no_helmet" in [rule.kind for rule in rules_for(scene, classifier_available=True)]


def test_derived_rules_omit_red_light_because_its_schedule_is_per_run() -> None:
    """Supported by the scene, but not derivable: no default can invent the timing."""

    scene = _example_scene()
    assert ViolationType.RED_LIGHT_JUMPING in supported_violations(
        scene, classifier_available=True
    )
    assert "red_light_jumping" not in [
        rule.kind for rule in rules_for(scene, classifier_available=True)
    ]


def test_derived_rules_actually_build_through_the_engine_registry() -> None:
    """The claim that makes R1 + R5 meaningful: supported implies runnable.

    ``build_rules`` is the engine's own registry -- the same call the processing
    service makes at submit. A derived rule that carried no resolved selector would
    raise here on this deliberately ambiguous scene, which is exactly the failure
    mode the audit found.
    """

    scene = _example_scene()
    derived = rules_for(scene, classifier_available=False)
    built = build_rules(derived, scene=scene)

    assert [rule.violation for rule in built] == [
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.TRIPLE_RIDING,
    ]

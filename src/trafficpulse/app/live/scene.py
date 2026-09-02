"""The scene a live camera is reasoned about, and what that scene can support.

A live camera has the same problem an uncalibrated upload has -- reasoning needs
geometry in *this* camera's pixel space -- and it is solved the same way, through
the same authorities:

* an operator who has calibrated this camera passes the scene revision's hash, and
  live mode reasons through the analyst's own geometry;
* otherwise the session gets a **provisional** scene built from the first frame's
  measured size and nothing else -- one full-frame lane, no legal direction, no
  no-stopping zone, no stop line, no signal timing.

Nothing is derived from a live stream. Automatic calibration
(:meth:`~trafficpulse.app.services.SceneService.derive_from_motion`) estimates a
clip's dominant flow from a bounded prefix of a *recorded* file, which a live
session does not have: at the moment monitoring starts there is no traffic history
to measure, and a direction estimated from the first few seconds of an arbitrary
camera view would be a guess presented as a measurement. So live mode declines,
and the violations that need geometry stay explicitly unavailable until an analyst
supplies it.

The consequence is stated, not hidden. :func:`live_rule_plan` reports both what
will run and what will not *with the reason*, and the session sends that to the
client before the first frame -- so a viewer is told "wrong-way is not being
evaluated on this camera" rather than being left to infer it from an empty event
list.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...contracts import SceneConfig, ZoneType
from ...contracts.enums import ViolationType
from ...engine import RuleConfig
from ...scenes import (
    CALIBRATION_SOURCE_AUTO,
    SceneDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)
from ..capabilities import probe_scene, rules_for

#: The full-frame lane a provisional live scene declares. Named, so an operator
#: reading a live scene can tell it apart from a drawn one at a glance.
LIVE_LANE_ID = "zone-live-view"


@dataclass(frozen=True)
class UnavailableViolation:
    """One violation live mode will not evaluate, and the reason it will not."""

    violation_type: ViolationType
    reason: str


@dataclass(frozen=True)
class LiveRulePlan:
    """What a live session will run, and what it explicitly will not."""

    rules: tuple[RuleConfig, ...]
    supported: tuple[ViolationType, ...]
    unavailable: tuple[UnavailableViolation, ...]


def provisional_live_scene(
    *, width: int, height: int, camera_id: str, scene_id: str
) -> SceneConfig:
    """A frame-correct scene for a live camera that claims nothing else.

    One zone spanning the frame, for the same reason the upload path uses one: the
    roadway's true extent is not observable, and a smaller invented polygon would
    silently exclude traffic from every rule that is scoped to a zone.
    """

    draft = SceneDraft(
        scene_name="live-provisional",
        camera_id=camera_id,
        site_id="site-live",
        description=(
            "Provisional scene for a live camera. Only the frame size is measured; "
            "no legal direction, no-stopping zone, stop line or signal timing is "
            "claimed, because none of them is observable from a camera view. "
            "Calibrate the camera to enable the violations that need geometry."
        ),
        frame_width=width,
        frame_height=height,
        zones=(
            ZoneDraft(
                zone_id=LIVE_LANE_ID,
                zone_type=ZoneType.LANE,
                polygon=full_frame_polygon(width, height),
                description="Full frame; the roadway's true extent is not observable.",
            ),
        ),
        direction=None,
    )
    return build_scene(draft, scene_id=scene_id, calibration_source=CALIBRATION_SOURCE_AUTO)


#: Why each violation is unavailable, given the probe said its scene support is
#: absent. One sentence per violation, naming the *missing evidence* rather than
#: restating that it is off -- a client shows these verbatim.
_SCENE_REASONS: dict[ViolationType, str] = {
    ViolationType.WRONG_WAY: (
        "This camera's scene declares no legal travel direction, so there is nothing "
        "to judge a vehicle's heading against. Calibrate the camera to enable it."
    ),
    ViolationType.ILLEGAL_STOPPING: (
        "This camera's scene declares no no-stopping zone, so there is no region a "
        "dwelling vehicle could be stopped illegally in. Calibrate the camera to enable it."
    ),
    ViolationType.RED_LIGHT_JUMPING: (
        "Red-light reasoning needs a stop line, the junction it guards, and the "
        "signal timing for the period being watched. None of the three is observable "
        "from a camera stream, and a live session is given no schedule."
    ),
    ViolationType.NO_HELMET: (
        "The no-helmet rule cannot be built for this deployment or scene. Helmet state "
        "is still classified and displayed where a classifier is configured, as analysis."
    ),
    ViolationType.TRIPLE_RIDING: (
        "This camera's scene cannot satisfy the triple-riding rule's parameters."
    ),
}


def live_rule_plan(scene: SceneConfig, *, no_helmet_available: bool) -> LiveRulePlan:
    """The rules a live session runs over ``scene``, and the ones it cannot.

    Delegates entirely to the existing capability probe -- the same one the upload
    path uses to decide a job's default rules -- so live mode can never run a rule
    file mode would refuse, or refuse one file mode would run. Nothing about a rule
    is re-decided here; this only *explains* the probe's answer.

    Red-light is absent from the running set even on a fully calibrated camera, and
    that is the shipped derivation's own behaviour rather than a live restriction:
    its rule config carries the run's signal schedule, which no default can supply.
    """

    capabilities = probe_scene(scene)
    rules = rules_for(scene, no_helmet_available=no_helmet_available)
    running = {
        ViolationType.WRONG_WAY: capabilities.wrong_way,
        ViolationType.ILLEGAL_STOPPING: capabilities.illegal_stopping,
        ViolationType.NO_HELMET: no_helmet_available and capabilities.no_helmet,
        ViolationType.TRIPLE_RIDING: capabilities.triple_riding,
        # Never derived, on any scene -- see the docstring.
        ViolationType.RED_LIGHT_JUMPING: False,
    }
    supported = tuple(kind for kind in _SCENE_REASONS if running[kind])
    unavailable = tuple(
        UnavailableViolation(violation_type=kind, reason=_SCENE_REASONS[kind])
        for kind in _SCENE_REASONS
        if not running[kind]
    )
    return LiveRulePlan(rules=rules, supported=supported, unavailable=unavailable)

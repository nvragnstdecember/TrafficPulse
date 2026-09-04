"""Scene authoring and calibration (H12).

The layer between an analyst's calibration surface and the frozen
:class:`~trafficpulse.contracts.SceneConfig` contract. Two concerns, deliberately
separate because one is measured and the other is declared:

* :mod:`~trafficpulse.scenes.calibration` derives the facts a clip can *show* --
  currently the dominant traffic-flow direction, estimated from observed vehicle
  tracks.
* :mod:`~trafficpulse.scenes.builder` expands a :class:`SceneDraft` -- the minimal
  analyst-authorable vocabulary -- into a complete validated ``SceneConfig``,
  deterministically, so the same drawing always yields the same scene hash.

:mod:`~trafficpulse.scenes.demo_scenario` sits beside them as the one *declared*
scene the project authors itself: the controlled demonstration, whose geometry,
timing and expected outcomes are hand-written so a single clip can exercise four
independent reasoners. It is input, never result -- it mints no event and no
reasoner is shown its expectations.

This package computes no geometry of its own (no point-in-polygon, no crossing,
no heading comparison -- those live in ``geometry``/``observations``), runs no
detector or tracker, and imports no ML framework. It authors declarative data and
estimates one vector. The single exception to "no I/O" is
:func:`~trafficpulse.scenes.demo_scenario.render_demo_clip`, which encodes the
controlled scenario to a video file; it is opt-in, imports PyAV lazily, and
nothing else in the package calls it.
"""

from __future__ import annotations

from .builder import (
    CALIBRATION_SOURCE_ANALYST,
    CALIBRATION_SOURCE_AUTO,
    DirectionDraft,
    RuleTuning,
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)
from .calibration import (
    FLOW_CLASSES,
    MIN_NET_DISPLACEMENT_PX,
    MIN_TRACK_LIFETIME_SECONDS,
    FlowEstimate,
    estimate_dominant_flow,
)
from .demo_scenario import (
    DEMO_EXPECTED_VIOLATIONS,
    DEMO_FPS,
    DEMO_FRAME_COUNT,
    DEMO_HEIGHT,
    DEMO_LABEL_MAP,
    DEMO_SCENE_NAME,
    DEMO_SCENE_NOTES,
    DEMO_SIGNAL_SCHEDULE,
    DEMO_STATIONARY_DURATION_S,
    DEMO_WIDTH,
    DemoActor,
    demo_actors,
    demo_scene,
    demo_scene_draft,
    render_demo_clip,
)

__all__ = [
    # authoring
    "SceneDraft",
    "ZoneDraft",
    "DirectionDraft",
    "StopLineDraft",
    "SignalGroupDraft",
    "RuleTuning",
    "build_scene",
    "CALIBRATION_SOURCE_ANALYST",
    "CALIBRATION_SOURCE_AUTO",
    "full_frame_polygon",
    # calibration
    "FlowEstimate",
    "estimate_dominant_flow",
    "FLOW_CLASSES",
    "MIN_TRACK_LIFETIME_SECONDS",
    "MIN_NET_DISPLACEMENT_PX",
    # controlled demonstration scenario
    "DemoActor",
    "demo_actors",
    "demo_scene",
    "demo_scene_draft",
    "render_demo_clip",
    "DEMO_EXPECTED_VIOLATIONS",
    "DEMO_FPS",
    "DEMO_FRAME_COUNT",
    "DEMO_HEIGHT",
    "DEMO_LABEL_MAP",
    "DEMO_SCENE_NAME",
    "DEMO_SCENE_NOTES",
    "DEMO_SIGNAL_SCHEDULE",
    "DEMO_STATIONARY_DURATION_S",
    "DEMO_WIDTH",
]

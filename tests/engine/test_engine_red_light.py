"""Red-light jumping composed onto the shared engine front half (H13).

The differential test every rule in this repository carries: a one-rule engine must
be event-identical to the corresponding standalone pipeline. If the engine's
composition ever adds or loses behaviour, this is what says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pipeline_helpers import DETECTOR_CONFIG
from _slice_fixtures import HEIGHT, WIDTH, scripted_down_detector, write_wrong_way_clip

from trafficpulse.contracts import SceneConfig
from trafficpulse.contracts.enums import SignalState, ViolationType
from trafficpulse.contracts.scene import ZoneType
from trafficpulse.engine import (
    EngineConfig,
    EngineConfigurationError,
    FileFrameSource,
    InferenceEngine,
    RedLightRuleConfig,
    SignalPhaseSpec,
    build_rules,
)
from trafficpulse.pipeline.red_light import RedLightPipeline, phases_from_offsets
from trafficpulse.scenes import (
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)
from trafficpulse.tracking import IouTracker

STOP_LINE_Y = 120.0
JUNCTION_TOP = 150.0
CAMERA = "cam-synthetic"


def junction_scene() -> SceneConfig:
    return build_scene(
        SceneDraft(
            scene_name="Synthetic junction",
            camera_id=CAMERA,
            frame_width=WIDTH,
            frame_height=HEIGHT,
            zones=(
                ZoneDraft(
                    zone_id="zone-approach",
                    zone_type=ZoneType.LANE,
                    polygon=full_frame_polygon(WIDTH, HEIGHT),
                ),
                ZoneDraft(
                    zone_id="zone-junction",
                    zone_type=ZoneType.INTERSECTION,
                    polygon=(
                        (100.0, JUNCTION_TOP),
                        (220.0, JUNCTION_TOP),
                        (220.0, 235.0),
                        (100.0, 235.0),
                    ),
                ),
            ),
            stop_lines=(
                StopLineDraft(
                    stop_line_id="sl-1",
                    a=(100.0, STOP_LINE_Y),
                    b=(220.0, STOP_LINE_Y),
                    crossing_dx=0.0,
                    crossing_dy=1.0,
                    signal_group_id="sg-1",
                    zone_ids=("zone-junction",),
                ),
            ),
            signal_groups=(
                SignalGroupDraft(
                    signal_group_id="sg-1",
                    roi_polygon=((5.0, 5.0), (45.0, 5.0), (45.0, 60.0)),
                    zone_ids=("zone-junction",),
                ),
            ),
        ),
        scene_id="scene-red-light",
    )


_RED_SCHEDULE = (SignalPhaseSpec(at_seconds=0.0, state=SignalState.RED),)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wrong_way_clip(tmp_path_factory.mktemp("engine-red-light") / "clip.mp4")


def _engine(scene: SceneConfig) -> InferenceEngine:
    return InferenceEngine(
        scene=scene,
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=DETECTOR_CONFIG,
        config=EngineConfig(rules=(RedLightRuleConfig(schedule=_RED_SCHEDULE),)),
    )


def test_engine_events_equal_the_standalone_pipeline(clip: Path) -> None:
    """The composition adds no behaviour: one-rule engine == RedLightPipeline."""

    scene = junction_scene()
    standalone_pipeline = RedLightPipeline(
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        scene=scene,
        detector_config=DETECTOR_CONFIG,
        schedule=phases_from_offsets([(0.0, SignalState.RED)]),
    )
    from trafficpulse.ingestion.video import open_video

    with open_video(clip, camera_id=CAMERA) as reader:
        for record in reader:
            standalone_pipeline.process_frame(record)
    standalone = standalone_pipeline.finalize()
    assert standalone, "the scenario must genuinely confirm for this to prove anything"

    result = _engine(scene).run(FileFrameSource(clip, camera_id=CAMERA))

    assert result.events == standalone


def test_the_engine_produces_evidence_for_a_red_light_event(clip: Path) -> None:
    result = _engine(junction_scene()).run(FileFrameSource(clip, camera_id=CAMERA))

    assert [e.violation_type for e in result.events] == [ViolationType.RED_LIGHT_JUMPING]
    trigger = result.manifests[0].trigger_frame
    assert trigger is not None
    assert trigger.locator.startswith(f"frames/{CAMERA}/vfrm-")


def test_the_engine_exposes_the_overlay_capture_without_a_pixel_observer(clip: Path) -> None:
    engine = _engine(junction_scene())
    engine.run(FileFrameSource(clip, camera_id=CAMERA))

    # Red-light reasons over geometry, so it contributes no frame observer...
    assert engine.frame_observers() == ()
    # ...but still publishes overlay metadata through the pixel-free channel.
    captures = engine.overlay_captures()
    assert len(captures) == 1
    assert captures[0].frames  # type: ignore[attr-defined]


def test_a_schedule_less_rule_is_refused_at_build_time() -> None:
    # Silently confirming nothing is the failure mode this refusal prevents.
    with pytest.raises(EngineConfigurationError, match="schedule"):
        build_rules((RedLightRuleConfig(),), scene=junction_scene())


def test_red_light_runs_alongside_another_rule_over_one_front_half(clip: Path) -> None:
    from trafficpulse.engine import TripleRidingRuleConfig

    scene = junction_scene()
    engine = InferenceEngine(
        scene=scene,
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        detector_config=DETECTOR_CONFIG,
        config=EngineConfig(
            rules=(
                RedLightRuleConfig(schedule=_RED_SCHEDULE),
                TripleRidingRuleConfig(),
            )
        ),
    )

    result = engine.run(FileFrameSource(clip, camera_id=CAMERA))

    # Triple riding needs motorcycles + riders, which this clip has none of, so the
    # union is exactly the red-light event -- and both rules shared one detect+track.
    assert [e.violation_type for e in result.events] == [ViolationType.RED_LIGHT_JUMPING]
    assert engine.metrics.frames_processed == 30

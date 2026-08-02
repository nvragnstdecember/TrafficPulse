"""Red-light jumping end to end over a real decoded clip (H13).

Uses the shared synthetic clip: one rectangle moving **down** the frame at 10 fps.
Placing a horizontal stop line at y=120 and the junction polygon below it gives a
genuine ~0.5 s gap between the stop-line crossing and the junction entry -- which is
exactly the window in which a real signal can change, and therefore the window this
milestone's latch exists to handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pipeline_helpers import DETECTOR_CONFIG
from _slice_fixtures import FPS, HEIGHT, WIDTH, scripted_down_detector, write_wrong_way_clip

from trafficpulse.contracts import SceneConfig
from trafficpulse.contracts.enums import SignalState, ViolationType
from trafficpulse.contracts.scene import ZoneType
from trafficpulse.ingestion.video import open_video
from trafficpulse.pipeline.errors import SceneConfigurationError
from trafficpulse.pipeline.red_light import (
    RedLightPipeline,
    phases_from_offsets,
    resolve_red_light_geometry,
)
from trafficpulse.scenes import (
    SceneDraft,
    SignalGroupDraft,
    StopLineDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)
from trafficpulse.tracking import IouTracker

# The rectangle's bottom-centre is (160, 50 + 6*frame). It crosses y=120 between
# frames 11 and 12 (t=1.2 s) and reaches the junction polygon at y>=150 on frame 17
# (t=1.7 s) -- a 0.5 s gap in which the signal can legitimately change.
STOP_LINE_Y = 120.0
JUNCTION_TOP = 150.0
CROSSING_SECONDS = 1.2
ENTRY_SECONDS = 1.7


def _scene(*, with_junction: bool = True) -> SceneConfig:
    zones = [
        ZoneDraft(
            zone_id="zone-approach",
            zone_type=ZoneType.LANE,
            polygon=full_frame_polygon(WIDTH, HEIGHT),
        )
    ]
    stop_lines: tuple[StopLineDraft, ...] = ()
    groups: tuple[SignalGroupDraft, ...] = ()
    if with_junction:
        zones.append(
            ZoneDraft(
                zone_id="zone-junction",
                zone_type=ZoneType.INTERSECTION,
                polygon=(
                    (100.0, JUNCTION_TOP),
                    (220.0, JUNCTION_TOP),
                    (220.0, 235.0),
                    (100.0, 235.0),
                ),
            )
        )
        stop_lines = (
            StopLineDraft(
                stop_line_id="sl-1",
                a=(100.0, STOP_LINE_Y),
                b=(220.0, STOP_LINE_Y),
                crossing_dx=0.0,
                crossing_dy=1.0,  # entering means travelling DOWN the frame
                signal_group_id="sg-1",
                zone_ids=("zone-junction",),
            ),
        )
        groups = (
            SignalGroupDraft(
                signal_group_id="sg-1",
                roi_polygon=((5.0, 5.0), (45.0, 5.0), (45.0, 60.0)),
                zone_ids=("zone-junction",),
            ),
        )
    return build_scene(
        SceneDraft(
            scene_name="Synthetic junction",
            camera_id="cam-synthetic",
            frame_width=WIDTH,
            frame_height=HEIGHT,
            zones=tuple(zones),
            stop_lines=stop_lines,
            signal_groups=groups,
        ),
        scene_id="scene-red-light",
    )


def _run(clip: Path, scene: SceneConfig, offsets: list[tuple[float, SignalState]]):
    pipeline = RedLightPipeline(
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        scene=scene,
        detector_config=DETECTOR_CONFIG,
        schedule=phases_from_offsets(offsets),
    )
    with open_video(clip, camera_id=scene.scene.camera_id) as reader:
        for record in reader:
            pipeline.process_frame(record)
    return pipeline.finalize(), pipeline.overlay_capture


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wrong_way_clip(tmp_path_factory.mktemp("red-light") / "clip.mp4")


# --- the milestone's success path -------------------------------------------------
def test_a_vehicle_entering_on_red_confirms(clip: Path) -> None:
    events, _ = _run(clip, _scene(), [(0.0, SignalState.RED)])

    assert len(events) == 1
    assert events[0].violation_type is ViolationType.RED_LIGHT_JUMPING
    assert events[0].camera_id == "cam-synthetic"


def test_a_vehicle_entering_on_green_confirms_nothing(clip: Path) -> None:
    events, _ = _run(clip, _scene(), [(0.0, SignalState.GREEN)])

    assert events == ()


def test_a_light_turning_green_between_the_line_and_the_junction_still_confirms(
    clip: Path,
) -> None:
    # The whole point of H13, on real decoded frames. The vehicle crosses the stop
    # line at t=1.2 s on RED; the light goes green at t=1.5 s; it reaches the
    # junction at t=1.7 s. The violation was committed at the line and must stand.
    events, _ = _run(
        clip,
        _scene(),
        [(0.0, SignalState.RED), (1.5, SignalState.GREEN)],
    )

    assert len(events) == 1, "the violation is committed at the stop line, not the polygon"


def test_a_light_turning_red_after_the_crossing_confirms_nothing(clip: Path) -> None:
    # The mirror image, and the reason the latch must read the *crossing* instant:
    # a vehicle that crossed lawfully on green does not become a violator because
    # the light changed while it was still in the junction.
    events, _ = _run(
        clip,
        _scene(),
        [(0.0, SignalState.GREEN), (1.5, SignalState.RED)],
    )

    assert events == ()


@pytest.mark.parametrize("state", [SignalState.AMBER, SignalState.OFF])
def test_neither_amber_nor_off_confirms(clip: Path, state: SignalState) -> None:
    events, _ = _run(clip, _scene(), [(0.0, state)])

    assert events == ()


def test_an_unknown_signal_before_the_schedule_starts_confirms_nothing(clip: Path) -> None:
    # The schedule begins after the vehicle has already crossed, so the crossing
    # instant resolves to UNKNOWN -- the absence of evidence, never a violation.
    events, _ = _run(clip, _scene(), [(2.5, SignalState.RED)])

    assert events == ()


# --- scene resolution --------------------------------------------------------------
def test_a_scene_without_junction_geometry_fails_fast(clip: Path) -> None:
    with pytest.raises((SceneConfigurationError, ValueError)):
        RedLightPipeline(
            detector=scripted_down_detector(),
            tracker=IouTracker(),
            scene=_scene(with_junction=False),
            detector_config=DETECTOR_CONFIG,
            schedule=phases_from_offsets([(0.0, SignalState.RED)]),
        )


def test_geometry_resolves_the_single_declared_approach() -> None:
    stop_line, zone = resolve_red_light_geometry(_scene())

    assert stop_line.stop_line_id == "sl-1"
    assert zone.zone_id == "zone-junction"


def test_an_unknown_stop_line_id_is_refused() -> None:
    with pytest.raises(SceneConfigurationError, match="sl-nope"):
        resolve_red_light_geometry(_scene(), stop_line_id="sl-nope")


# --- determinism + replay ------------------------------------------------------------
def test_replaying_the_run_reproduces_the_same_events(clip: Path) -> None:
    first, _ = _run(clip, _scene(), [(0.0, SignalState.RED)])
    second, _ = _run(clip, _scene(), [(0.0, SignalState.RED)])

    assert [e.event_id for e in first] == [e.event_id for e in second]


def test_reset_returns_the_pipeline_to_a_replayable_state(clip: Path) -> None:
    pipeline = RedLightPipeline(
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        scene=_scene(),
        detector_config=DETECTOR_CONFIG,
        schedule=phases_from_offsets([(0.0, SignalState.RED)]),
    )
    with open_video(clip, camera_id="cam-synthetic") as reader:
        records = list(reader)
    for record in records:
        pipeline.process_frame(record)
    first = pipeline.finalize()

    pipeline.reset()
    pipeline._core._detector = scripted_down_detector()  # noqa: SLF001 - fresh scripted replay
    for record in records:
        pipeline.process_frame(record)
    second = pipeline.finalize()

    assert [e.event_id for e in first] == [e.event_id for e in second]


# --- overlay capture -------------------------------------------------------------------
def test_the_run_captures_overlay_metadata_without_a_pixel_observer(clip: Path) -> None:
    # Red-light reasons over geometry alone, so its overlay metadata is produced by
    # the reasoning pass rather than by a FrameObserver.
    events, capture = _run(clip, _scene(), [(0.0, SignalState.RED)])

    assert events
    assert capture.stop_line == ((100.0, STOP_LINE_Y), (220.0, STOP_LINE_Y))
    assert len(capture.zone_polygon) == 4
    assert capture.frames, "the reasoning pass must record per-frame geometry"
    entered = [f for f in capture.frames if f.entered_on_red]
    assert entered
    assert all(f.entry_state is SignalState.RED for f in entered)
    # The capture describes the same frames the clip has, at real media times.
    assert max(f.media_seconds for f in capture.frames) <= 30 / FPS


def test_the_capture_is_cleared_between_runs(clip: Path) -> None:
    pipeline = RedLightPipeline(
        detector=scripted_down_detector(),
        tracker=IouTracker(),
        scene=_scene(),
        detector_config=DETECTOR_CONFIG,
        schedule=phases_from_offsets([(0.0, SignalState.RED)]),
    )
    with open_video(clip, camera_id="cam-synthetic") as reader:
        for record in reader:
            pipeline.process_frame(record)

    pipeline.finalize()
    first = len(pipeline.overlay_capture.frames)
    pipeline.finalize()  # a second finalize over the same history
    second = len(pipeline.overlay_capture.frames)

    assert first == second, "a replay must describe one run, not an accumulation"

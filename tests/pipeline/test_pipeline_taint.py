"""Taint-restart propagation through orchestration (P1-U10).

Taint markers originate *only* in the existing ``derive_heading_observations_with_taint``
(from tracker-declared ``TrackState.tainted``) and reset the reasoner's run at the
restart. The orchestration must route those markers through faithfully: tainted
support cannot bridge into a confirmation, while a fresh clean post-taint episode
may confirm on its own. Taint is injected via ``StubTracker`` scripted assignments
(the ``IouTracker`` never taints).
"""

from _pipeline_helpers import (
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_down_detector,
)

from trafficpulse.contracts import ViolationType
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.tracking import ScriptedAssignment, StubTracker


def _pipeline(detector, tracker) -> WrongWayPipeline:
    return WrongWayPipeline(
        detector=detector,
        tracker=tracker,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )


def _tracker(frame_count: int, tainted_frames: frozenset[int] = frozenset()) -> StubTracker:
    script = {
        i: (ScriptedAssignment(track_id="t1", tainted=i in tainted_frames),)
        for i in range(frame_count)
    }
    return StubTracker(script)


def _frames(frame_count: int) -> list:
    return [make_frame_record(i) for i in range(frame_count)]


# --- control: the full clean run confirms ------------------------------------
def test_clean_run_confirms_without_taint() -> None:
    # 60 frames span ~2.0 s > 1.0 s min_persistence.
    pipeline = _pipeline(moving_down_detector(60), _tracker(60))
    assert len(pipeline.process(_frames(60))) == 1


# --- tainted support cannot bridge into confirmation -------------------------
def test_midstream_taint_splits_the_run_and_prevents_confirmation() -> None:
    # The same 60 frames (which DO confirm when clean) are split by a taint at
    # frame 30 into two sub-episodes each < 1.0 s, so neither confirms.
    pipeline = _pipeline(moving_down_detector(60), _tracker(60, frozenset({30})))
    assert pipeline.process(_frames(60)) == ()


# --- a fresh clean post-taint episode may confirm ----------------------------
def test_long_clean_episode_after_taint_confirms() -> None:
    # Taint early (frame 5); the long clean segment that resumes after it spans
    # well over min_persistence and confirms on its own.
    pipeline = _pipeline(moving_down_detector(80), _tracker(80, frozenset({5})))
    events = pipeline.process(_frames(80))
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.WRONG_WAY
    assert events[0].track_ids == ("t1",)

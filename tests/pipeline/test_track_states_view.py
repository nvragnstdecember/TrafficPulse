"""The accumulated-history view scene auto-calibration reads.

``finalize`` has always been a pure function of the history the pipeline
accumulates. This exposes that same history to a caller that wants the *observed
motion* without any rule's conclusion about it -- which is what deriving a clip's
dominant traffic flow needs, and the reason the view exists at all.

Two properties are load-bearing. It must be **ordered deterministically**, because
the flow estimate feeds a scene whose content hash addresses one revision and whose
events carry that hash in their ids. And it must be a **copy**, because a caller
inspecting a run mid-stream must not be able to alter what that run will reason
over.
"""

from __future__ import annotations

from _pipeline_helpers import (
    DEFAULT_FRAME_COUNT,
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_raw,
)

from trafficpulse.detector import StubDetector
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.pipeline.base import CompositionPipeline
from trafficpulse.pipeline.wrong_way import wrong_way_finalize_strategy
from trafficpulse.tracking import ScriptedAssignment, StubTracker


def _frames(count: int = DEFAULT_FRAME_COUNT):  # type: ignore[no-untyped-def]
    return [make_frame_record(i) for i in range(count)]


def _pipeline(track_ids: tuple[str, ...] = ("only",)) -> CompositionPipeline:
    per_frame = {
        i: tuple(moving_raw(i, x=140.0 + n * 20.0) for n in range(len(track_ids)))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    script = {
        i: tuple(ScriptedAssignment(track_id=t) for t in track_ids)
        for i in range(DEFAULT_FRAME_COUNT)
    }
    return CompositionPipeline(
        detector=StubDetector(per_frame=per_frame),
        tracker=StubTracker(script),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=wrong_way_finalize_strategy(
            SCENE, direction_id=NORTH_DIRECTION_ID
        ),
    )


def test_states_accumulate_as_frames_are_processed() -> None:
    pipeline = _pipeline()
    assert pipeline.track_states == ()
    for record in _frames(5):
        pipeline.process_frame(record)
    assert len(pipeline.track_states) == 5


def test_the_view_is_ordered_deterministically() -> None:
    pipeline = _pipeline(("b-track", "a-track"))
    for record in _frames():
        pipeline.process_frame(record)
    states = pipeline.track_states
    keys = [(s.camera_id, s.track_id, s.timestamp, s.frame_index or 0) for s in states]
    assert keys == sorted(keys)


def test_repeated_reads_are_equal() -> None:
    pipeline = _pipeline()
    for record in _frames(6):
        pipeline.process_frame(record)
    assert pipeline.track_states == pipeline.track_states


def test_the_view_is_a_copy_a_reader_cannot_mutate_the_run() -> None:
    pipeline = _pipeline()
    for record in _frames(4):
        pipeline.process_frame(record)
    snapshot = pipeline.track_states
    assert isinstance(snapshot, tuple)
    pipeline.process_frame(make_frame_record(4))
    # The earlier read did not grow, and the run kept accumulating independently.
    assert len(snapshot) == 4
    assert len(pipeline.track_states) == 5


def test_reset_clears_the_view() -> None:
    pipeline = _pipeline()
    for record in _frames(4):
        pipeline.process_frame(record)
    pipeline.reset()
    assert pipeline.track_states == ()


def test_reading_the_view_does_not_disturb_reasoning() -> None:
    # The view must be an observation of the run, never a participant in it.
    without = _pipeline().process(_frames())

    watched = _pipeline()
    for record in _frames():
        watched.process_frame(record)
        _ = watched.track_states  # read on every frame
    with_reads = watched.finalize()

    assert [e.event_id for e in with_reads] == [e.event_id for e in without]


def test_the_engine_exposes_the_same_view() -> None:
    # The application reads it through the engine, not the pipeline.
    engine = WrongWayPipeline(
        detector=StubDetector(
            per_frame={i: (moving_raw(i),) for i in range(DEFAULT_FRAME_COUNT)}
        ),
        tracker=StubTracker(
            {i: (ScriptedAssignment(track_id="only"),) for i in range(DEFAULT_FRAME_COUNT)}
        ),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )
    for record in _frames(3):
        engine.process_frame(record)
    # WrongWayPipeline delegates to the same core the engine uses.
    assert len(engine._core.track_states) == 3  # noqa: SLF001 - asserting the seam

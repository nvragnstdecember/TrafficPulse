"""Stub-driven end-to-end orchestration (P1-U10).

Drives the ``WrongWayPipeline`` with a scripted ``StubDetector`` + ``StubTracker``
over synthetic ``FrameRecord``s and asserts the wiring: it produces a real
wrong-way ``ConfirmedEvent``, adds no behaviour over calling the existing
derivation + reasoner directly, is deterministic and order-independent, and does
legal / short episodes / empty input correctly.
"""

from _pipeline_helpers import (
    CAMERA,
    DEFAULT_FRAME_COUNT,
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_detection,
    moving_down_detector,
    moving_raw,
)

from trafficpulse.contracts import ConfirmedEvent, ViolationType
from trafficpulse.contracts import scene_config_hash as _scene_hash
from trafficpulse.detector import Detector, StubDetector
from trafficpulse.observations.heading import derive_heading_observations_with_taint
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.wrong_way import WrongWayReasoner, wrong_way_parameters
from trafficpulse.tracking import IouTracker, ScriptedAssignment, StubTracker, Tracker


def _single_track_script(frame_count: int = DEFAULT_FRAME_COUNT, track_id: str = "t1") -> dict:
    return {i: (ScriptedAssignment(track_id=track_id),) for i in range(frame_count)}


def _pipeline(
    detector: Detector, tracker: Tracker, *, direction_id: str = NORTH_DIRECTION_ID
) -> WrongWayPipeline:
    return WrongWayPipeline(
        detector=detector,
        tracker=tracker,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=direction_id,
    )


def _frames(frame_count: int = DEFAULT_FRAME_COUNT) -> list:
    return [make_frame_record(i) for i in range(frame_count)]


# --- a known wrong-way event -------------------------------------------------
def test_pipeline_produces_a_wrong_way_confirmed_event() -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    events = pipeline.process(_frames())
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ConfirmedEvent)
    assert event.violation_type is ViolationType.WRONG_WAY
    assert event.camera_id == CAMERA
    assert event.track_ids == ("t1",)
    assert event.scene_config_hash == _scene_hash(SCENE)


# --- equivalence: wiring adds no behaviour -----------------------------------
def _direct_events(track_id: str = "t1", frame_count: int = DEFAULT_FRAME_COUNT) -> tuple:
    """The reference: feed equivalent Detections to the SAME tracker, then call the
    existing P1-U4 derivation + reasoner directly (no pipeline)."""

    tracker = StubTracker(_single_track_script(frame_count, track_id))
    history: dict = {}
    for i in range(frame_count):
        for state in tracker.update([moving_detection(i)]):
            history.setdefault((state.camera_id, state.track_id), []).append(state)
    params = wrong_way_parameters(SCENE)
    reasoner = WrongWayReasoner(RuleEngine(), params, scene_config_hash=_scene_hash(SCENE))
    north = next(d for d in SCENE.legal_directions if d.direction_id == NORTH_DIRECTION_ID)
    events = []
    for key in sorted(history):
        derivation = derive_heading_observations_with_taint(
            history[key],
            legal_direction=north.vector,
            lane_id=north.zone_ids[0],
            deviation_max_degrees=params.deviation_max_degrees,
        )
        events.extend(reasoner.run_derivation(derivation))
    return tuple(e.event_id for e in events)


def test_pipeline_matches_direct_derivation_and_reasoning() -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    pipeline_ids = tuple(e.event_id for e in pipeline.process(_frames()))
    assert pipeline_ids == _direct_events()  # identical event set: wiring adds nothing


# --- legal motion produces no event ------------------------------------------
def test_legal_direction_motion_produces_no_event() -> None:
    # Moving up (direction=-1, starting high so the box stays in frame) agrees
    # with legal north -> no contradiction.
    pipeline = _pipeline(
        moving_down_detector(direction=-1, y0=300.0), StubTracker(_single_track_script())
    )
    assert pipeline.process(_frames()) == ()


# --- insufficient persistence produces no event ------------------------------
def test_insufficient_persistence_produces_no_event() -> None:
    # 10 frames at 30 fps span ~0.3 s < the scene's 1.0 s min_persistence.
    short = 10
    pipeline = _pipeline(
        moving_down_detector(short), StubTracker(_single_track_script(short))
    )
    assert pipeline.process(_frames(short)) == ()


# --- empty input --------------------------------------------------------------
def test_empty_frame_stream_produces_no_event() -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    assert pipeline.process([]) == ()


# --- multiple tracks ----------------------------------------------------------
def test_two_tracks_each_confirm_independently() -> None:
    per_frame = {
        i: (moving_raw(i, x=50.0), moving_raw(i, x=300.0)) for i in range(DEFAULT_FRAME_COUNT)
    }
    script = {
        i: (ScriptedAssignment(track_id="left"), ScriptedAssignment(track_id="right"))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    pipeline = _pipeline(StubDetector(per_frame=per_frame), StubTracker(script))
    events = pipeline.process(_frames())
    assert {tid for e in events for tid in e.track_ids} == {"left", "right"}
    assert len(events) == 2


# --- deterministic event ordering --------------------------------------------
def test_event_ordering_is_deterministic_by_trigger_then_id() -> None:
    per_frame = {
        i: (moving_raw(i, x=50.0), moving_raw(i, x=300.0)) for i in range(DEFAULT_FRAME_COUNT)
    }
    script = {
        i: (ScriptedAssignment(track_id="b"), ScriptedAssignment(track_id="a"))
        for i in range(DEFAULT_FRAME_COUNT)
    }
    pipeline = _pipeline(StubDetector(per_frame=per_frame), StubTracker(script))
    events = pipeline.process(_frames())
    keys = [(e.trigger_at, e.event_id) for e in events]
    assert keys == sorted(keys)


# --- determinism: fresh instances --------------------------------------------
def test_fresh_instances_are_deterministic() -> None:
    def run() -> tuple:
        pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
        return tuple(e.event_id for e in pipeline.process(_frames()))

    assert run() == run()


# --- determinism: reset + replay on one instance -----------------------------
def test_reset_and_replay_on_one_instance_is_deterministic() -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    first = tuple(e.event_id for e in pipeline.process(_frames()))
    second = tuple(e.event_id for e in pipeline.process(_frames()))  # process() resets internally
    assert first == second


# --- real IouTracker compatibility (in-memory integration) -------------------
def test_real_iou_tracker_integration() -> None:
    pipeline = _pipeline(moving_down_detector(), IouTracker())
    events = pipeline.process(_frames())
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.WRONG_WAY
    assert events[0].track_ids == ("iou-1",)  # backend-assigned stream-local id


# --- no persistence side effects ---------------------------------------------
def test_process_returns_events_and_writes_nothing(tmp_path) -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    before = set(tmp_path.iterdir())
    pipeline.process(_frames())
    assert set(tmp_path.iterdir()) == before  # orchestration performs no I/O


# --- finalize idempotency -----------------------------------------------------
def test_finalize_is_idempotent_over_history() -> None:
    pipeline = _pipeline(moving_down_detector(), StubTracker(_single_track_script()))
    pipeline.reset()
    for frame in _frames():
        pipeline.process_frame(frame)
    first = tuple(e.event_id for e in pipeline.finalize())
    second = tuple(e.event_id for e in pipeline.finalize())
    assert first == second == (pipeline.finalize()[0].event_id,)

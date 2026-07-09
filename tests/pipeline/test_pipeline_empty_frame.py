"""Zero-detection-frame semantics through orchestration (P1-U10).

The tracker seam treats an empty ``Sequence[Detection]`` as inert (no aging, no
frame-progress advance -- documented P1-U8/U9 behaviour). The orchestration must
honour that faithfully: fabricate no detection to advance time, emit no state for
the frame, and let identity bridge across the gap when detections resume.
"""

from _pipeline_helpers import (
    DEFAULT_FRAME_COUNT,
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_raw,
)

from trafficpulse.contracts import ViolationType
from trafficpulse.detector import StubDetector
from trafficpulse.pipeline import WrongWayPipeline
from trafficpulse.tracking import IouTracker, ScriptedAssignment, StubTracker


def _pipeline(detector, tracker) -> WrongWayPipeline:
    return WrongWayPipeline(
        detector=detector,
        tracker=tracker,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=NORTH_DIRECTION_ID,
    )


def _detector_with_gap(gap: int, frame_count: int = DEFAULT_FRAME_COUNT) -> StubDetector:
    """A moving-down detector that emits nothing on the ``gap`` frame."""

    per_frame = {
        i: (moving_raw(i),) for i in range(frame_count) if i != gap
    }
    return StubDetector(per_frame=per_frame)  # unscripted 'gap' frame -> default () -> empty


def test_zero_detection_frame_emits_no_state() -> None:
    pipeline = _pipeline(StubDetector(), StubTracker())  # default detector: always empty
    assert pipeline.process_frame(make_frame_record(0)) == ()


def test_zero_detection_frame_does_not_error_between_populated_frames() -> None:
    # StubTracker: script only the populated frames; the empty frame is inert and
    # never consults the script.
    gap = 22
    script = {
        i: (ScriptedAssignment(track_id="t1"),)
        for i in range(DEFAULT_FRAME_COUNT)
        if i != gap
    }
    pipeline = _pipeline(_detector_with_gap(gap), StubTracker(script))
    events = pipeline.process([make_frame_record(i) for i in range(DEFAULT_FRAME_COUNT)])
    assert len(events) == 1  # the ordinary gap is bridged by timestamp in derivation


def test_identity_bridges_across_a_zero_detection_frame_with_iou_tracker() -> None:
    # A one-frame gap: the surviving track re-matches the resumed detection (IoU
    # ~0.33 >= 0.3), so identity bridges and a single event still confirms.
    gap = 22
    pipeline = _pipeline(_detector_with_gap(gap), IouTracker())
    events = pipeline.process([make_frame_record(i) for i in range(DEFAULT_FRAME_COUNT)])
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.WRONG_WAY
    assert events[0].track_ids == ("iou-1",)  # one continuous identity across the gap


def test_all_zero_detection_frames_produce_no_events() -> None:
    pipeline = _pipeline(StubDetector(), IouTracker())  # never emits a detection
    assert pipeline.process([make_frame_record(i) for i in range(DEFAULT_FRAME_COUNT)]) == ()

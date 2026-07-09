"""Backend independence, dependency injection, and the error model (P1-U10).

Proves the orchestration depends only on the ``Detector`` / ``Tracker``
abstractions and the frozen contracts: arbitrary in-test implementations (neither
the stubs nor the real backends) plug in unchanged, the real backends satisfy the
seams without being run, no backend-native type leaks, and lower-level errors
propagate while the one orchestration-owned error (lane resolution) is raised.
"""


import pytest
from _pipeline_helpers import (
    CAMERA,
    DETECTOR_CONFIG,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_down_detector,
)
from pydantic import ValidationError

from trafficpulse.contracts import ConfirmedEvent, TrackState
from trafficpulse.contracts.enums import TrackStatus
from trafficpulse.detector import (
    Detector,
    MalformedDetectorOutputError,
    RawDetection,
    RTDetrDetector,
    StubDetector,
)
from trafficpulse.detector.frame import Frame
from trafficpulse.pipeline import SceneConfigurationError, WrongWayPipeline
from trafficpulse.tracking import (
    FrameProgress,
    IouTracker,
    NonMonotonicFrameError,
    ScriptedAssignment,
    StubTracker,
    TrackAdapter,
    TrackAssignment,
    Tracker,
    single_frame_key,
)


def _pipeline(
    detector, tracker, *, direction_id: str | None = NORTH_DIRECTION_ID
) -> WrongWayPipeline:
    return WrongWayPipeline(
        detector=detector,
        tracker=tracker,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        direction_id=direction_id,
    )


# --- arbitrary in-test backends satisfy the seams ----------------------------
class _DownCarDetector(Detector):
    """A minimal ``Detector`` (neither StubDetector nor RTDetrDetector)."""

    def detect(self, frame: Frame) -> tuple[RawDetection, ...]:
        top = 50.0 + frame.frame_index * 5.0
        return (RawDetection(label="car", score=0.9, box=(50.0, top, 70.0, top + 20.0)),)


class _SingleTrackTracker(Tracker):
    """A minimal ``Tracker`` (neither StubTracker nor IouTracker) via the public seam."""

    def __init__(self) -> None:
        self._adapter = TrackAdapter()
        self._progress = FrameProgress()

    def update(self, detections):  # type: ignore[no-untyped-def]
        key = single_frame_key(detections)
        if key is None:
            return ()
        self._progress.advance(key)
        assignments = [
            TrackAssignment(track_id="only", detection=d, status=TrackStatus.ACTIVE)
            for d in detections
        ]
        return self._adapter.adapt(assignments)

    def reset(self) -> None:
        self._progress.reset()


def test_arbitrary_detector_and_tracker_implementations_inject() -> None:
    pipeline = _pipeline(_DownCarDetector(), _SingleTrackTracker())
    events = pipeline.process([make_frame_record(i) for i in range(45)])
    assert len(events) == 1
    assert events[0].track_ids == ("only",)


# --- real backends satisfy the seams (without being run) ---------------------
def test_real_backends_satisfy_the_injected_seams() -> None:
    # Compatibility only: no torch/model loading, no inference. Proves the real
    # backends are valid injections for the abstractions the pipeline depends on.
    assert issubclass(RTDetrDetector, Detector)
    assert issubclass(IouTracker, Tracker)


# --- no backend-native type leaks --------------------------------------------
def test_only_frozen_contracts_cross_the_boundary() -> None:
    pipeline = _pipeline(moving_down_detector(45), StubTracker(
        {i: (ScriptedAssignment(track_id="t1"),) for i in range(45)}
    ))
    states = pipeline.process_frame(make_frame_record(0))
    assert all(isinstance(s, TrackState) for s in states)
    events = pipeline.finalize()
    assert all(isinstance(e, ConfirmedEvent) for e in events)


def test_pipeline_module_imports_no_backend() -> None:
    # Bindings, not docstring prose: no backend-concrete or ML-framework symbol is
    # imported into the orchestration module's namespace (the abstractions are).
    import trafficpulse.pipeline.wrong_way as core

    forbidden = (
        "RTDetrDetector",
        "IouTracker",
        "StubTracker",
        "StubDetector",
        "torch",
        "transformers",
    )
    for name in forbidden:
        assert not hasattr(core, name), f"pipeline core imports backend {name!r}"


# --- error model --------------------------------------------------------------
def test_ambiguous_scene_without_direction_id_raises_scene_error() -> None:
    # The example scene declares two legal directions; the single-lane slice needs
    # an explicit direction_id.
    with pytest.raises(SceneConfigurationError):
        _pipeline(StubDetector(), StubTracker(), direction_id=None)


def test_unknown_direction_id_raises_scene_error() -> None:
    with pytest.raises(SceneConfigurationError):
        _pipeline(StubDetector(), StubTracker(), direction_id="dir-does-not-exist")


def test_detector_malformed_output_propagates() -> None:
    bad = StubDetector(default=(RawDetection(label="car", score=2.0, box=(1.0, 1.0, 2.0, 2.0)),))
    pipeline = _pipeline(bad, StubTracker())
    with pytest.raises(MalformedDetectorOutputError):
        pipeline.process_frame(make_frame_record(0))


def test_tracker_non_monotonic_frame_propagates() -> None:
    pipeline = _pipeline(moving_down_detector(10), IouTracker())
    pipeline.process_frame(make_frame_record(5))
    with pytest.raises(NonMonotonicFrameError):
        pipeline.process_frame(make_frame_record(3))  # frame goes backwards


# --- frozen ConfirmedEvent ----------------------------------------------------
def test_confirmed_event_is_frozen() -> None:
    pipeline = _pipeline(moving_down_detector(45), StubTracker(
        {i: (ScriptedAssignment(track_id="t1"),) for i in range(45)}
    ))
    (event,) = pipeline.process([make_frame_record(i) for i in range(45)])
    assert isinstance(event, ConfirmedEvent)
    with pytest.raises(ValidationError):  # pydantic frozen model rejects mutation
        event.camera_id = "other"  # type: ignore[misc]
    assert event.camera_id == CAMERA

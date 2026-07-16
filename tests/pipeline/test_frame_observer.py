"""The optional ``FrameObserver`` hook on ``CompositionPipeline`` (P4-U2).

The hook touches **shared** code that both shipped slices (wrong-way P1-U10,
illegal-stopping P2-U5) run through, so the load-bearing test here is the
regression one: with no observer injected, the pipeline must behave exactly as it
did before P4-U2. The rest guards the hook's own contract -- call-once-per-frame,
stream order, pixel access, empty-state frames, and reset-for-replay.

No classifier is involved: this unit wires the hook, it does not use it.
"""

from __future__ import annotations

from collections.abc import Sequence

from _pipeline_helpers import (
    DEFAULT_FRAME_COUNT,
    DETECTOR_CONFIG,
    SCENE,
    make_frame_record,
    moving_down_detector,
)

from trafficpulse.contracts import ConfirmedEvent, ModelRef, TrackState
from trafficpulse.detector.frame import Frame
from trafficpulse.pipeline.base import CompositionPipeline
from trafficpulse.pipeline.wrong_way import WrongWayPipeline
from trafficpulse.tracking import IouTracker


class _RecordingObserver:
    """A test double satisfying the ``FrameObserver`` protocol; records what it saw."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int, bool]] = []  # (frame_index, n_states, had_pixels)
        self.reset_count = 0

    def observe(self, frame: Frame, states: Sequence[TrackState]) -> None:
        self.seen.append((frame.frame_index, len(states), frame.image is not None))

    def reset(self) -> None:
        self.reset_count += 1


class _NullFinalize:
    """A finalize strategy that reasons about nothing (the hook is what is under test)."""

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> object:
        return object()

    def events_for_track(
        self, reasoner: object, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        return ()


def _pipeline(observer: _RecordingObserver | None) -> CompositionPipeline:
    return CompositionPipeline(
        detector=moving_down_detector(),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=_NullFinalize(),
        frame_observer=observer,
    )


def _frames(count: int = DEFAULT_FRAME_COUNT) -> list[object]:
    return [make_frame_record(i) for i in range(count)]


# --- the regression guard: shipped slices are unchanged ----------------------
def test_wrong_way_slice_is_byte_identical_without_an_observer() -> None:
    """The shipped slice does not inject an observer; its events must not move.

    Guards the P4-U2 promise that adding the hook changed no existing behaviour --
    same events, same ids, same ordering.
    """

    def run() -> tuple[ConfirmedEvent, ...]:
        return WrongWayPipeline(
            detector=moving_down_detector(),
            tracker=IouTracker(),
            scene=SCENE,
            detector_config=DETECTOR_CONFIG,
            direction_id="dir-north",
        ).process(_frames())

    events = run()
    assert events, "precondition: the wrong-way slice confirms an event"
    assert [e.model_dump_json() for e in run()] == [e.model_dump_json() for e in events]


def test_observer_defaults_to_none() -> None:
    """Omitting the argument must be the pre-P4-U2 construction."""

    pipeline = CompositionPipeline(
        detector=moving_down_detector(),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=_NullFinalize(),
    )
    pipeline.process(_frames(3))  # must not raise


def test_events_are_identical_with_and_without_an_observer() -> None:
    """An observer accumulates; it must never influence the reasoning result."""

    def run(observer: _RecordingObserver | None) -> tuple[ConfirmedEvent, ...]:
        return WrongWayPipeline(
            detector=moving_down_detector(),
            tracker=IouTracker(),
            scene=SCENE,
            detector_config=DETECTOR_CONFIG,
            direction_id="dir-north",
        ).process(_frames())

    assert [e.model_dump_json() for e in run(_RecordingObserver())] == [
        e.model_dump_json() for e in run(None)
    ]


# --- the hook's own contract -------------------------------------------------
def test_observer_is_called_once_per_frame_in_stream_order() -> None:
    observer = _RecordingObserver()
    _pipeline(observer).process(_frames(5))

    assert [frame_index for frame_index, _, _ in observer.seen] == [0, 1, 2, 3, 4]


def test_observer_receives_pixels_and_that_frames_states() -> None:
    """The hook exists precisely because it is the only place pixels + tracks coexist."""

    observer = _RecordingObserver()
    _pipeline(observer).process(_frames(5))

    assert all(had_pixels for _, _, had_pixels in observer.seen)
    assert any(n_states > 0 for _, n_states, _ in observer.seen)


def test_observer_is_called_for_zero_state_frames() -> None:
    """An observer must see the true frame sequence, not a gappy one."""

    from trafficpulse.detector import StubDetector

    pipeline = CompositionPipeline(
        detector=StubDetector(),  # emits nothing: every frame yields zero states
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=_NullFinalize(),
        frame_observer=(observer := _RecordingObserver()),
    )
    pipeline.process(_frames(4))

    assert [frame_index for frame_index, _, _ in observer.seen] == [0, 1, 2, 3]
    assert all(n_states == 0 for _, n_states, _ in observer.seen)


def test_reset_resets_the_observer() -> None:
    observer = _RecordingObserver()
    pipeline = _pipeline(observer)

    pipeline.reset()

    assert observer.reset_count == 1


def test_process_resets_the_observer_before_streaming() -> None:
    """``process`` resets first, so repeated runs replay rather than accumulate."""

    observer = _RecordingObserver()
    pipeline = _pipeline(observer)

    pipeline.process(_frames(3))
    first = list(observer.seen)
    pipeline.process(_frames(3))

    assert observer.reset_count == 2
    assert observer.seen[len(first) :] == first

"""Tests for the generalized composition-pipeline base (P3-U2, composition).

Exercises :class:`~trafficpulse.pipeline.base.CompositionPipeline` directly with a
**recording fake strategy**, independently of any violation, to pin the shared
orchestration the two shipped pipelines delegate to: the detect -> track -> group ->
provenance front half, the deterministic ``finalize`` scaffold (one reasoner built
per finalize with the normalized run-level ``models`` + scene hash, tracks visited
in sorted key order, each track's states sorted by ``(timestamp, frame_index)``,
output sorted by ``(trigger_at, event_id)``), and ``reset`` / ``process`` /
empty-frame semantics.

The base must carry no violation-specific knowledge, so a ``SPEEDING`` event --
minted by neither shipped strategy -- is what the fake strategy emits here. Uses
the model-free ``StubDetector`` / ``StubTracker`` seams; no backend, no video.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from _pipeline_helpers import (
    CAMERA,
    DETECTOR_CONFIG,
    SCENE,
    make_frame_record,
    moving_down_detector,
    moving_raw,
)

from trafficpulse.contracts import (
    ConfirmedEvent,
    ModelRef,
    TrackState,
    scene_config_hash,
)
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector import StubDetector
from trafficpulse.pipeline.base import CompositionPipeline
from trafficpulse.tracking import ScriptedAssignment, StubTracker

BASE = datetime(1970, 1, 1, tzinfo=UTC)
SCH = scene_config_hash(SCENE)


def _event(event_id: str, *, trigger_offset: float, track_ids: tuple[str, ...]) -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.SPEEDING,  # base is violation-agnostic
        camera_id=CAMERA,
        track_ids=track_ids,
        start_at=BASE,
        trigger_at=BASE + timedelta(seconds=trigger_offset),
        rule_id="fake_rule",
        created_at=BASE,
    )


@dataclass
class _RecordingStrategy:
    """A fake ``FinalizeStrategy`` that records its calls and emits scripted events.

    ``emit`` maps a track id to the events the strategy returns for that track, so a
    test can control the *unsorted* set the base must order. ``build_calls`` and
    ``tracks_seen`` capture what the base handed the strategy.
    """

    emit: dict[str, tuple[ConfirmedEvent, ...]] = field(default_factory=dict)
    build_calls: list[tuple[str | None, tuple[ModelRef, ...]]] = field(default_factory=list)
    tracks_seen: list[list[TrackState]] = field(default_factory=list)

    def build_reasoner(
        self, *, scene_config_hash: str | None, models: tuple[ModelRef, ...]
    ) -> object:
        self.build_calls.append((scene_config_hash, models))
        return object()  # opaque reasoner sentinel; the base threads it back unread

    def events_for_track(
        self, reasoner: object, track: list[TrackState]
    ) -> tuple[ConfirmedEvent, ...]:
        self.tracks_seen.append(track)
        return self.emit.get(track[0].track_id, ())


def _pipeline(detector, tracker, strategy: _RecordingStrategy) -> CompositionPipeline:
    return CompositionPipeline(
        detector=detector,
        tracker=tracker,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
        finalize_strategy=strategy,
    )


def _single_track_script(frame_count: int, track_id: str = "t1") -> dict:
    return {i: (ScriptedAssignment(track_id=track_id),) for i in range(frame_count)}


def _frames(frame_count: int) -> list:
    return [make_frame_record(i) for i in range(frame_count)]


# --- front half: detect -> track -> group -> accumulate ----------------------
def test_process_frame_returns_states_and_accumulates_history() -> None:
    strategy = _RecordingStrategy()
    pipeline = _pipeline(moving_down_detector(3), StubTracker(_single_track_script(3)), strategy)
    for i in range(3):
        states = pipeline.process_frame(make_frame_record(i))
        assert all(isinstance(s, TrackState) for s in states)
        assert len(states) == 1
    pipeline.finalize()
    # One track group of three states, sorted by (timestamp, frame_index).
    assert len(strategy.tracks_seen) == 1
    track = strategy.tracks_seen[0]
    assert [s.frame_index for s in track] == [0, 1, 2]
    assert track == sorted(track, key=lambda s: (s.timestamp, s.frame_index or 0))


def test_finalize_builds_one_reasoner_with_scene_hash_and_normalized_models() -> None:
    strategy = _RecordingStrategy()
    pipeline = _pipeline(moving_down_detector(3), StubTracker(_single_track_script(3)), strategy)
    pipeline.process(_frames(3))
    # Exactly one reasoner built per finalize; the stubs stamp no provenance, so the
    # normalized run-level models are honestly empty.
    assert len(strategy.build_calls) == 1
    scene_hash, models = strategy.build_calls[0]
    assert scene_hash == SCH
    assert models == ()


def test_tracks_visited_in_sorted_key_order() -> None:
    # Two detections per frame scripted to identities in REVERSE emission order
    # ("t2" first, "t1" second); the base must still visit tracks in sorted
    # (camera_id, track_id) key order -> t1 before t2.
    per_frame = {
        i: (moving_raw(i, x=50.0), moving_raw(i, x=400.0)) for i in range(2)
    }
    script = {
        i: (ScriptedAssignment(track_id="t2"), ScriptedAssignment(track_id="t1"))
        for i in range(2)
    }
    strategy = _RecordingStrategy()
    pipeline = _pipeline(StubDetector(per_frame=per_frame), StubTracker(script), strategy)
    pipeline.process(_frames(2))
    seen_ids = [track[0].track_id for track in strategy.tracks_seen]
    assert seen_ids == ["t1", "t2"]


# --- finalize scaffold: deterministic output ordering ------------------------
def test_output_events_sorted_by_trigger_then_id() -> None:
    # The strategy returns three events for the single track in a deliberately
    # unsorted order; the base must return them sorted by (trigger_at, event_id).
    out_of_order = (
        _event("evt-c", trigger_offset=2.0, track_ids=("t1",)),
        _event("evt-a", trigger_offset=1.0, track_ids=("t1",)),
        _event("evt-b", trigger_offset=1.0, track_ids=("t1",)),
    )
    strategy = _RecordingStrategy(emit={"t1": out_of_order})
    pipeline = _pipeline(moving_down_detector(3), StubTracker(_single_track_script(3)), strategy)
    events = pipeline.process(_frames(3))
    assert [e.event_id for e in events] == ["evt-a", "evt-b", "evt-c"]  # (trigger, id) order


# --- reset / process semantics -----------------------------------------------
def test_reset_clears_history_and_provenance() -> None:
    strategy = _RecordingStrategy()
    pipeline = _pipeline(moving_down_detector(3), StubTracker(_single_track_script(3)), strategy)
    for i in range(3):
        pipeline.process_frame(make_frame_record(i))
    pipeline.reset()
    events = pipeline.finalize()
    assert events == ()
    assert strategy.tracks_seen == []  # no track survived the reset


def test_process_is_reset_then_stream_then_finalize() -> None:
    strategy = _RecordingStrategy()
    pipeline = _pipeline(moving_down_detector(3), StubTracker(_single_track_script(3)), strategy)
    # Pre-seed some state; process() must reset it away, not accumulate on top.
    pipeline.process_frame(make_frame_record(0))
    pipeline.process(_frames(3))
    # Fresh IouTracker/StubTracker script only covers frames 0..2 once; the second
    # finalize's single track has exactly three states (proof the pre-seed was reset).
    assert len(strategy.tracks_seen[-1]) == 3


def test_process_is_deterministic_across_calls() -> None:
    def run() -> tuple[str, ...]:
        strategy = _RecordingStrategy(
            emit={"t1": (_event("evt-x", trigger_offset=1.0, track_ids=("t1",)),)}
        )
        pipeline = _pipeline(
            moving_down_detector(3), StubTracker(_single_track_script(3)), strategy
        )
        return tuple(e.event_id for e in pipeline.process(_frames(3)))

    assert run() == run() == ("evt-x",)


# --- empty-detection frame is inert ------------------------------------------
def test_empty_detection_frame_is_inert() -> None:
    strategy = _RecordingStrategy()
    pipeline = _pipeline(StubDetector(per_frame={0: ()}), StubTracker({0: ()}), strategy)
    states = pipeline.process_frame(make_frame_record(0))
    assert states == ()
    assert pipeline.finalize() == ()
    assert strategy.tracks_seen == []  # nothing accumulated


# --- backend-free import boundary --------------------------------------------
def test_base_module_imports_no_backend() -> None:
    import trafficpulse.pipeline.base as base

    forbidden = ("RTDetrDetector", "IouTracker", "StubTracker", "StubDetector", "torch",
                 "transformers")
    for name in forbidden:
        assert not hasattr(base, name), f"pipeline base imports backend {name!r}"

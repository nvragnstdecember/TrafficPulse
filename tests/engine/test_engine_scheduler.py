"""Frame scheduler: decimation, back-pressure, determinism (H6)."""

from __future__ import annotations

import pytest
from _engine_helpers import frame_records

from trafficpulse.engine import FrameScheduler, ScheduleDecision, SchedulerConfig


def _decisions(scheduler: FrameScheduler, count: int, *, interval: float) -> list[str]:
    return [
        scheduler.submit(record).value
        for record in frame_records(count, interval_seconds=interval)
    ]


# --- pass-through ----------------------------------------------------------------
def test_default_config_admits_everything() -> None:
    scheduler = FrameScheduler(SchedulerConfig())
    decisions = _decisions(scheduler, 5, interval=0.1)
    assert decisions == ["admitted"] * 5
    assert scheduler.queue_depth == 5


# --- stride ----------------------------------------------------------------------
def test_stride_keeps_every_nth_read_frame() -> None:
    scheduler = FrameScheduler(SchedulerConfig(frame_stride=3))
    decisions = _decisions(scheduler, 7, interval=0.1)
    assert decisions == [
        "admitted",
        "skipped_stride",
        "skipped_stride",
        "admitted",
        "skipped_stride",
        "skipped_stride",
        "admitted",
    ]


# --- target FPS --------------------------------------------------------------------
def test_target_fps_decimates_by_media_time() -> None:
    # 10 fps stream decimated to 5 fps: every other frame admitted, exactly.
    scheduler = FrameScheduler(SchedulerConfig(target_fps=5.0))
    decisions = _decisions(scheduler, 6, interval=0.1)
    assert decisions == [
        "admitted",
        "skipped_fps",
        "admitted",
        "skipped_fps",
        "admitted",
        "skipped_fps",
    ]


def test_target_fps_above_stream_rate_admits_everything() -> None:
    scheduler = FrameScheduler(SchedulerConfig(target_fps=60.0))
    assert _decisions(scheduler, 4, interval=0.1) == ["admitted"] * 4


def test_fps_gate_measures_from_last_admitted_frame() -> None:
    # Irregular PTS: the gate compares against the last ADMITTED timestamp, so a
    # long gap admits immediately even after several skips.
    scheduler = FrameScheduler(SchedulerConfig(target_fps=2.0))  # 0.5 s interval
    records = frame_records(5, interval_seconds=0.2)  # 0.0 0.2 0.4 0.6 0.8
    decisions = [scheduler.submit(r).value for r in records]
    assert decisions == ["admitted", "skipped_fps", "skipped_fps", "admitted", "skipped_fps"]


# --- stride + fps compose -----------------------------------------------------------
def test_stride_applies_before_the_fps_gate() -> None:
    # Stride 2 halves a 10 fps stream to 5 fps; a 5 fps target then admits all
    # surviving frames -- the gates compose without double-decimation.
    scheduler = FrameScheduler(SchedulerConfig(frame_stride=2, target_fps=5.0))
    decisions = _decisions(scheduler, 6, interval=0.1)
    assert decisions == [
        "admitted",
        "skipped_stride",
        "admitted",
        "skipped_stride",
        "admitted",
        "skipped_stride",
    ]


# --- back-pressure -------------------------------------------------------------------
def test_full_queue_drops_the_incoming_frame() -> None:
    scheduler = FrameScheduler(SchedulerConfig(queue_capacity=2))
    decisions = _decisions(scheduler, 4, interval=0.1)
    assert decisions == ["admitted", "admitted", "dropped_queue_full", "dropped_queue_full"]
    # The queued (older) frames survive; the newest were rejected.
    assert [r.frame_index for r in scheduler.take(10)] == [0, 1]


def test_draining_reopens_admission() -> None:
    scheduler = FrameScheduler(SchedulerConfig(queue_capacity=1))
    records = frame_records(3, interval_seconds=0.1)
    assert scheduler.submit(records[0]) is ScheduleDecision.ADMITTED
    assert scheduler.submit(records[1]) is ScheduleDecision.DROPPED_QUEUE_FULL
    assert scheduler.take(1) == (records[0],)
    assert scheduler.submit(records[2]) is ScheduleDecision.ADMITTED


def test_a_dropped_frame_does_not_advance_the_fps_gate() -> None:
    scheduler = FrameScheduler(SchedulerConfig(target_fps=5.0, queue_capacity=1))
    records = frame_records(3, interval_seconds=0.2)  # all pass the fps gate
    scheduler.submit(records[0])
    assert scheduler.submit(records[1]) is ScheduleDecision.DROPPED_QUEUE_FULL
    scheduler.take(1)
    # records[2] is 0.2 s after records[1] but 0.4 s after the last ADMITTED
    # frame; it must be admitted (the dropped frame never became the reference).
    assert scheduler.submit(records[2]) is ScheduleDecision.ADMITTED


# --- take / reset ---------------------------------------------------------------------
def test_take_returns_oldest_first_up_to_limit() -> None:
    scheduler = FrameScheduler(SchedulerConfig())
    records = frame_records(5, interval_seconds=0.1)
    for record in records:
        scheduler.submit(record)
    assert [r.frame_index for r in scheduler.take(2)] == [0, 1]
    assert [r.frame_index for r in scheduler.take(10)] == [2, 3, 4]
    assert scheduler.take(1) == ()


def test_take_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        FrameScheduler(SchedulerConfig()).take(0)


def test_reset_restores_the_initial_state() -> None:
    scheduler = FrameScheduler(SchedulerConfig(frame_stride=2, target_fps=5.0))
    first = _decisions(scheduler, 6, interval=0.1)
    scheduler.reset()
    assert scheduler.queue_depth == 0
    second = _decisions(scheduler, 6, interval=0.1)
    assert first == second  # identical stream, identical decisions: deterministic

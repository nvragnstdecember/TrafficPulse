"""In-memory registries + job records (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import FakeEngine, make_metrics

from trafficpulse.app.registry import (
    JobRecord,
    JobStatus,
    JobStore,
    VideoRecord,
    VideoStore,
)
from trafficpulse.engine import EngineRunResult


def _video(video_id: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        filename=f"{video_id}.mp4",
        path=Path(f"{video_id}.mp4"),
        size_bytes=1,
        width=320,
        height=240,
        fps=10.0,
        frame_count=30,
        duration_seconds=3.0,
        codec="mpeg4",
    )


# --- video store ---------------------------------------------------------------
def test_video_store_add_get_contains() -> None:
    store = VideoStore()
    assert store.get("vid-a") is None
    assert not store.contains("vid-a")
    store.add(_video("vid-a"))
    assert store.contains("vid-a")
    assert store.get("vid-a") is not None


# --- job store -----------------------------------------------------------------
def test_job_lifecycle_transitions() -> None:
    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v"))
    assert store.get("j") is not None and store.get("j").status is JobStatus.PENDING

    store.mark_running("j", frames_total=10)
    assert store.get("j").status is JobStatus.RUNNING
    assert store.get("j").frames_total == 10

    result = EngineRunResult(
        source_id="v", events=(), manifests=(), metrics=make_metrics()
    )
    store.mark_succeeded("j", result)
    assert store.get("j").status is JobStatus.SUCCEEDED


def test_job_failure_records_the_message() -> None:
    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v"))
    store.mark_failed("j", "it broke")
    record = store.get("j")
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.error == "it broke"


def test_succeeded_for_video_filters() -> None:
    store = JobStore()
    for job_id, video in (("j1", "v1"), ("j2", "v2"), ("j3", "v1")):
        store.add(JobRecord(job_id=job_id, video_id=video))
        store.mark_running(job_id, frames_total=None)
    # only j1 succeeds
    store.mark_succeeded(
        "j1", EngineRunResult(source_id="v1", events=(), manifests=(), metrics=make_metrics())
    )
    assert [r.job_id for r in store.succeeded_for_video("v1")] == ["j1"]
    assert store.succeeded_for_video("v2") == ()
    assert [r.job_id for r in store.succeeded_for_video(None)] == ["j1"]


# --- job record metrics snapshot -----------------------------------------------
def test_metrics_snapshot_prefers_result_then_engine_then_none() -> None:
    none_record = JobRecord(job_id="j", video_id="v")
    assert none_record.metrics() is None

    engine_record = JobRecord(
        job_id="j",
        video_id="v",
        engine=FakeEngine(make_metrics(frames_processed=4)),  # type: ignore[arg-type]
    )
    snapshot = engine_record.metrics()
    assert snapshot is not None and snapshot.frames_processed == 4

    result = EngineRunResult(
        source_id="v", events=(), manifests=(), metrics=make_metrics(frames_processed=9)
    )
    done_record = JobRecord(
        job_id="j",
        video_id="v",
        engine=FakeEngine(make_metrics(frames_processed=4)),  # type: ignore[arg-type]
        result=result,
    )
    final = done_record.metrics()
    assert final is not None and final.frames_processed == 9  # result wins over engine

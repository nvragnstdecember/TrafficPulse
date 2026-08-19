"""In-memory registries + job records (H7A)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _app_helpers import FakeEngine, make_metrics

from trafficpulse.app.registry import (
    JobRecord,
    JobStatus,
    JobStore,
    VideoRecord,
    VideoStore,
)
from trafficpulse.contracts import ConfirmedEvent
from trafficpulse.contracts.enums import ViolationType
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


def test_video_store_enumerates_in_insertion_order() -> None:
    # The accessor the historical library needs (H11): a store you can only query by
    # id cannot be browsed.
    store = VideoStore()
    for video_id in ("vid-a", "vid-b", "vid-c"):
        store.add(_video(video_id))

    assert [record.video_id for record in store.videos()] == ["vid-a", "vid-b", "vid-c"]


def test_video_store_starts_empty() -> None:
    assert VideoStore().videos() == ()


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


def test_for_video_returns_every_run_whatever_its_status() -> None:
    # The library has to describe a video whose processing failed, which
    # succeeded_for_video would report as having no runs at all (H11).
    store = JobStore()
    for job_id, video in (("j1", "v1"), ("j2", "v2"), ("j3", "v1")):
        store.add(JobRecord(job_id=job_id, video_id=video))
    store.mark_failed("j1", "it broke")

    assert [r.job_id for r in store.for_video("v1")] == ["j1", "j3"]
    assert store.for_video("v-unknown") == ()


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


# --- duplicate-event folding (R2) -----------------------------------------------------
def _event(event_id: str, violation: ViolationType, *, second: int) -> ConfirmedEvent:
    """A minimal ConfirmedEvent; only identity, type, and ordering matter here."""

    at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=second)
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=violation,
        camera_id="cam-1",
        track_ids=("trk-1",),
        start_at=at,
        trigger_at=at,
        rule_id=violation.value,
        created_at=at,
    )


def test_mark_succeeded_folds_events_that_share_an_event_id() -> None:
    """R2: a duplicated rule declaration re-confirms one violation, counted once.

    ``event_id`` is content-derived, so two entries sharing one are the same
    violation reasoned twice. The write-once store already treated the repeat as a
    byte-identical no-op; before this fold the job-level count and histogram did
    not, and analytics reported twice the violations that exist.
    """

    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v"))
    stop = _event("evt-stop", ViolationType.ILLEGAL_STOPPING, second=4)
    wrong = _event("evt-wrong", ViolationType.WRONG_WAY, second=1)
    store.mark_succeeded(
        "j",
        EngineRunResult(
            source_id="v",
            # The shape MultiRuleFinalize produces for [wrong_way, wrong_way,
            # illegal_stopping, illegal_stopping]: sorted by (trigger_at, event_id).
            events=(wrong, wrong, stop, stop),
            manifests=(),
            metrics=make_metrics(),
        ),
    )

    record = store.get("j")
    assert record is not None
    assert record.event_ids == ("evt-wrong", "evt-stop")  # deduped, order preserved
    assert record.violation_counts == {"wrong_way": 1, "illegal_stopping": 1}
    assert store.job_for_event("evt-wrong") == "j"
    assert store.job_for_event("evt-stop") == "j"


def test_mark_succeeded_keeps_distinct_events_distinct() -> None:
    """Folding is by ``event_id`` alone -- two real violations stay two."""

    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v"))
    store.mark_succeeded(
        "j",
        EngineRunResult(
            source_id="v",
            events=(
                _event("evt-a", ViolationType.WRONG_WAY, second=1),
                _event("evt-b", ViolationType.WRONG_WAY, second=2),
            ),
            manifests=(),
            metrics=make_metrics(),
        ),
    )

    record = store.get("j")
    assert record is not None
    assert record.event_ids == ("evt-a", "evt-b")
    assert record.violation_counts == {"wrong_way": 2}

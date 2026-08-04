"""AnalyticsService aggregation over the registries (H15) -- unit level.

Drives the service directly over hand-built registry records so every branch is
reachable without running inference: an empty repository, a partial one, a
repository recovered from pre-H15 snapshots, and reprocessed videos.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trafficpulse.app.analytics import AnalyticsService
from trafficpulse.app.registry import JobRecord, JobStatus, JobStore, VideoRecord, VideoStore
from trafficpulse.evidence import ArtifactStore

_T0 = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class _Provider:
    """The EngineProvider seam, reduced to the one method analytics reads."""

    def __init__(self, readiness: str = "ready") -> None:
        self._readiness = readiness

    def describe(self) -> str:
        return self._readiness


def _video(video_id: str, tmp_path: Path, **overrides: object) -> VideoRecord:
    path = tmp_path / f"{video_id}.mp4"
    path.write_bytes(b"x" * 10)
    fields: dict[str, object] = {
        "video_id": video_id,
        "filename": f"{video_id}.mp4",
        "path": path,
        "size_bytes": 10,
        "width": 640,
        "height": 480,
        "fps": 25.0,
        "frame_count": 250,
        "duration_seconds": 10.0,
        "codec": "h264",
        "uploaded_at": _T0,
    }
    fields.update(overrides)
    return VideoRecord(**fields)  # type: ignore[arg-type]


def _job(job_id: str, video_id: str, **overrides: object) -> JobRecord:
    fields: dict[str, object] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": JobStatus.SUCCEEDED,
        "event_ids": (),
        "submitted_at": _T0,
        "started_at": _T0,
        "finished_at": _T0 + timedelta(seconds=4),
    }
    fields.update(overrides)
    record = JobRecord(job_id=str(fields.pop("job_id")), video_id=str(fields.pop("video_id")))
    for key, value in fields.items():
        setattr(record, key, value)
    return record


def _service(
    videos: list[VideoRecord],
    jobs: list[JobRecord],
    *,
    readiness: str = "ready",
    artifacts_dir: Path | None = None,
) -> AnalyticsService:
    video_store = VideoStore()
    for video in videos:
        video_store.restore(video)
    job_store = JobStore()
    for job in jobs:
        job_store.restore(job, job.event_ids)
    return AnalyticsService(
        videos=video_store,
        jobs=job_store,
        provider=_Provider(readiness),
        # H16: analytics reads the store's own maintained usage figure rather than
        # walking a directory on every request.
        artifacts=ArtifactStore(artifacts_dir) if artifacts_dir is not None else None,
    )


# --- empty repository ----------------------------------------------------------------
def test_an_empty_repository_reports_zeroes_not_nulls() -> None:
    summary = _service([], []).summary()

    assert summary.repository.videos_total == 0
    assert summary.violations.events_total == 0
    assert summary.violations.by_type == ()
    assert summary.review.events_total == 0
    assert summary.recent_activity == ()
    assert summary.latest_run is None
    # A count of nothing is genuinely zero; an unmeasured *mean* is not.
    assert summary.processing.average_duration_seconds is None
    assert summary.repository.footage_seconds is None


# --- repository overview -------------------------------------------------------------
def test_counts_processed_calibrated_and_unprocessed_videos(tmp_path: Path) -> None:
    videos = [
        _video("vid-a", tmp_path, scene_hash="scene-1"),
        _video("vid-b", tmp_path),
        _video("vid-c", tmp_path),
    ]
    jobs = [_job("job-1", "vid-a"), _job("job-2", "vid-b", status=JobStatus.FAILED)]

    overview = _service(videos, jobs).summary().repository
    assert overview.videos_total == 3
    assert overview.videos_processed == 1  # only vid-a has a succeeded run
    assert overview.videos_unprocessed == 2
    assert overview.videos_calibrated == 1
    assert overview.footage_seconds == 30.0
    assert overview.storage_bytes == 30


def test_a_video_with_no_declared_duration_is_excluded_not_zeroed(tmp_path: Path) -> None:
    videos = [_video("vid-a", tmp_path), _video("vid-b", tmp_path, duration_seconds=None)]
    assert _service(videos, []).summary().repository.footage_seconds == 10.0


def test_footage_is_null_when_no_video_declares_a_duration(tmp_path: Path) -> None:
    videos = [_video("vid-a", tmp_path, duration_seconds=None)]
    assert _service(videos, []).summary().repository.footage_seconds is None


# --- processing ----------------------------------------------------------------------
def test_averages_only_the_runs_that_recorded_timing(tmp_path: Path) -> None:
    jobs = [
        _job("job-1", "vid-a", finished_at=_T0 + timedelta(seconds=2)),
        _job("job-2", "vid-b", finished_at=_T0 + timedelta(seconds=6)),
        # Recovered from a pre-H15 snapshot: no timing at all.
        _job("job-3", "vid-c", started_at=None, finished_at=None, submitted_at=None),
    ]
    processing = _service([_video("vid-a", tmp_path)], jobs).summary().processing

    assert processing.jobs_total == 3
    assert processing.average_duration_seconds == 4.0  # (2 + 6) / 2, not / 3
    assert processing.timed_jobs == 2


def test_job_status_counts_cover_every_state(tmp_path: Path) -> None:
    jobs = [
        _job("job-1", "vid-a", status=JobStatus.SUCCEEDED),
        _job("job-2", "vid-a", status=JobStatus.FAILED),
        _job("job-3", "vid-a", status=JobStatus.CANCELLED),
        _job("job-4", "vid-a", status=JobStatus.RUNNING),
        _job("job-5", "vid-a", status=JobStatus.PENDING),
    ]
    processing = _service([], jobs).summary().processing
    assert (
        processing.jobs_succeeded,
        processing.jobs_failed,
        processing.jobs_cancelled,
        processing.jobs_running,
        processing.jobs_pending,
    ) == (1, 1, 1, 1, 1)


# --- violations ----------------------------------------------------------------------
def test_violation_breakdown_comes_from_the_persisted_histogram(tmp_path: Path) -> None:
    jobs = [
        _job(
            "job-1",
            "vid-a",
            event_ids=("evt-1", "evt-2", "evt-3"),
            violation_counts={"no_helmet": 2, "wrong_way": 1},
        ),
        _job(
            "job-2",
            "vid-b",
            event_ids=("evt-4",),
            violation_counts={"no_helmet": 1},
        ),
    ]
    violations = _service([], jobs).summary().violations

    assert violations.events_total == 4
    # Most frequent first.
    assert [(v.violation_type, v.count) for v in violations.by_type] == [
        ("no_helmet", 3),
        ("wrong_way", 1),
    ]
    assert violations.counted_jobs == 2
    assert violations.uncounted_jobs == 0


def test_reprocessing_a_video_does_not_double_count(tmp_path: Path) -> None:
    """The same video run twice yields the same content-derived ids, not twice as many."""

    jobs = [
        _job("job-1", "vid-a", event_ids=("evt-1", "evt-2"),
             violation_counts={"no_helmet": 2}),
        _job("job-2", "vid-a", event_ids=("evt-1", "evt-2"),
             violation_counts={"no_helmet": 2}),
    ]
    violations = _service([], jobs).summary().violations
    assert violations.events_total == 2
    assert [(v.violation_type, v.count) for v in violations.by_type] == [("no_helmet", 2)]


def test_a_pre_h15_run_is_reported_as_uncounted_not_as_zero(tmp_path: Path) -> None:
    """A recovered run contributes to the total but cannot contribute a breakdown."""

    jobs = [
        _job("job-1", "vid-a", event_ids=("evt-1",), violation_counts={"no_helmet": 1}),
        # Events on disk, but the snapshot predates the histogram.
        _job("job-2", "vid-b", event_ids=("evt-2", "evt-3"), violation_counts={}),
    ]
    violations = _service([], jobs).summary().violations

    assert violations.events_total == 3  # the total is still complete
    assert [(v.violation_type, v.count) for v in violations.by_type] == [("no_helmet", 1)]
    assert violations.counted_jobs == 1
    assert violations.uncounted_jobs == 1  # and the shortfall is declared


def test_a_run_that_confirmed_nothing_is_counted_not_flagged(tmp_path: Path) -> None:
    """An empty histogram with no events means 'measured zero', not 'not measured'."""

    jobs = [_job("job-1", "vid-a", event_ids=(), violation_counts={})]
    violations = _service([], jobs).summary().violations
    assert violations.counted_jobs == 1
    assert violations.uncounted_jobs == 0


def test_only_succeeded_runs_contribute_violations(tmp_path: Path) -> None:
    jobs = [
        _job("job-1", "vid-a", status=JobStatus.FAILED, event_ids=("evt-1",),
             violation_counts={"no_helmet": 1}),
    ]
    assert _service([], jobs).summary().violations.events_total == 0


# --- health --------------------------------------------------------------------------
def test_health_surfaces_operational_problems(tmp_path: Path) -> None:
    present = _video("vid-a", tmp_path, scene_hash="scene-1")
    missing = _video("vid-b", tmp_path)
    missing.path.unlink()
    jobs = [
        _job("job-1", "vid-a", status=JobStatus.FAILED),
        _job("job-2", "vid-a", started_at=None, finished_at=None),
    ]

    health = _service([present, missing], jobs, readiness="unconfigured").summary().health
    assert health.engine == "unconfigured"
    assert health.failed_jobs == 1
    assert health.videos_missing_media == 1
    assert health.videos_uncalibrated == 1
    # Only job-2: a failed run that recorded timing is still timed.
    assert health.runs_without_timing == 1


# --- activity ------------------------------------------------------------------------
def test_activity_is_newest_first_and_mixes_kinds(tmp_path: Path) -> None:
    videos = [_video("vid-a", tmp_path, uploaded_at=_T0)]
    jobs = [_job("job-1", "vid-a", finished_at=_T0 + timedelta(minutes=5), event_ids=("e",))]

    activity = _service(videos, jobs).summary().recent_activity
    assert [entry.kind for entry in activity] == ["run", "upload"]
    assert activity[0].subject_id == "job-1"
    assert "1 violation(s)" in activity[0].summary


def test_activity_omits_entries_with_no_recorded_instant(tmp_path: Path) -> None:
    """An undated row in a chronological feed is noise; the count still reports it."""

    videos = [_video("vid-a", tmp_path, uploaded_at=None)]
    jobs = [_job("job-1", "vid-a", submitted_at=None, started_at=None, finished_at=None)]

    summary = _service(videos, jobs).summary()
    assert summary.recent_activity == ()
    assert summary.repository.videos_total == 1  # still counted
    assert summary.processing.jobs_total == 1


def test_activity_is_capped(tmp_path: Path) -> None:
    jobs = [
        _job(f"job-{index}", "vid-a", finished_at=_T0 + timedelta(seconds=index))
        for index in range(40)
    ]
    assert len(_service([], jobs).summary().recent_activity) == 12


# --- storage -------------------------------------------------------------------------
def test_artifact_usage_is_measured_from_the_store(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts" / "ab"
    artifacts.mkdir(parents=True)
    (artifacts / "one.png").write_bytes(b"x" * 5)
    (artifacts / "two.png").write_bytes(b"y" * 7)

    evidence = _service([], [], artifacts_dir=tmp_path / "artifacts").summary().evidence
    assert evidence.artifacts_total == 2
    assert evidence.artifact_bytes == 12


def test_a_missing_artifact_directory_reports_zero(tmp_path: Path) -> None:
    evidence = _service([], [], artifacts_dir=tmp_path / "nope").summary().evidence
    assert (evidence.artifacts_total, evidence.artifact_bytes) == (0, 0)

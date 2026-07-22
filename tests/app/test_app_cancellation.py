"""Cooperative job cancellation across engine, registry, service, and HTTP (H7D).

Cancellation is exercised deterministically without threads: the engine's
per-frame ``should_cancel`` predicate at the unit level, and the full HTTP
lifecycle through :class:`DeferredJobExecutor`, which captures a job's work so a
test can cancel it *before* the run executes and then run it to observe the
terminal ``cancelled`` state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _app_helpers import (
    DEFAULT_RULES,
    SCENE,
    DeferredJobExecutor,
    StubEngineProvider,
    make_client,
    upload_wrong_way_video,
)
from _slice_fixtures import write_wrong_way_clip

from trafficpulse.app.registry import JobRecord, JobStatus, JobStore
from trafficpulse.engine import FileFrameSource, RunCancelledError


# --- engine (unit) -------------------------------------------------------------
def test_engine_run_raises_when_cancel_predicate_is_true(tmp_path: Path) -> None:
    engine = StubEngineProvider().create(scene=SCENE, rules=DEFAULT_RULES)
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    source = FileFrameSource(clip, camera_id=SCENE.scene.camera_id)
    with pytest.raises(RunCancelledError):
        engine.run(source, should_cancel=lambda: True)


def test_engine_run_completes_when_never_cancelled(tmp_path: Path) -> None:
    engine = StubEngineProvider().create(scene=SCENE, rules=DEFAULT_RULES)
    clip = write_wrong_way_clip(tmp_path / "clip.mp4")
    source = FileFrameSource(clip, camera_id=SCENE.scene.camera_id)
    result = engine.run(source, should_cancel=lambda: False)
    assert result.metrics.frames_processed == 30
    assert len(result.events) == 1


# --- registry (unit) -----------------------------------------------------------
def test_request_cancel_flags_a_running_job() -> None:
    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v", status=JobStatus.RUNNING))
    assert store.request_cancel("j") is True
    assert store.is_cancel_requested("j") is True
    store.mark_cancelled("j")
    record = store.get("j")
    assert record is not None and record.status is JobStatus.CANCELLED


def test_request_cancel_is_noop_on_a_terminal_job() -> None:
    store = JobStore()
    store.add(JobRecord(job_id="j", video_id="v", status=JobStatus.SUCCEEDED))
    assert store.request_cancel("j") is False
    assert store.is_cancel_requested("j") is False


def test_request_cancel_is_false_for_unknown_job() -> None:
    assert JobStore().request_cancel("nope") is False


def test_job_status_terminality() -> None:
    assert JobStatus.SUCCEEDED.is_terminal
    assert JobStatus.FAILED.is_terminal
    assert JobStatus.CANCELLED.is_terminal
    assert not JobStatus.PENDING.is_terminal
    assert not JobStatus.RUNNING.is_terminal


# --- HTTP lifecycle ------------------------------------------------------------
def test_cancel_transitions_job_to_cancelled_and_persists_nothing(tmp_path: Path) -> None:
    executor = DeferredJobExecutor()
    client = make_client(tmp_path, executor=executor)
    video_id = upload_wrong_way_video(client, tmp_path)

    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    # The work is captured but not yet run: the job is still pending.
    assert client.get(f"/api/process/{job_id}").json()["status"] == "pending"

    cancelled = client.post(f"/api/process/{job_id}/cancel")
    assert cancelled.status_code == 200

    # Running the deferred work now: the engine observes the cancel flag and stops.
    executor.run_pending()

    final = client.get(f"/api/process/{job_id}").json()
    assert final["status"] == "cancelled"
    assert final["event_count"] == 0
    assert final["error"] is None
    # A cancelled run persists nothing, so no events are listable for the video.
    listing = client.get("/api/events", params={"video_id": video_id}).json()
    assert listing["total"] == 0


def test_cancel_completed_job_is_a_noop(tmp_path: Path) -> None:
    client = make_client(tmp_path)  # synchronous executor: the job runs at submit
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    assert client.get(f"/api/process/{job_id}").json()["status"] == "succeeded"

    response = client.post(f"/api/process/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"  # unchanged


def test_cancel_unknown_job_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/process/job-nope/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "job_not_found"


def test_cancelled_job_counts_in_metrics(tmp_path: Path) -> None:
    executor = DeferredJobExecutor()
    client = make_client(tmp_path, executor=executor)
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    client.post(f"/api/process/{job_id}/cancel")
    executor.run_pending()

    metrics = client.get("/api/metrics").json()
    assert metrics["jobs_cancelled"] == 1
    assert metrics["jobs_total"] == 1
    assert metrics["jobs_succeeded"] == 0

"""Overlay-video endpoint + service behaviour (overlay integration)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config, make_scene_service, upload_wrong_way_video

from trafficpulse.app import SynchronousJobExecutor, ThreadJobExecutor
from trafficpulse.app.errors import JobNotFoundError, OverlayNotFoundError
from trafficpulse.app.registry import (
    JobRecord,
    JobStatus,
    JobStore,
    OverlayStatus,
    VideoStore,
)
from trafficpulse.app.services import ProcessingService, VideoService
from trafficpulse.persistence import EventStore


def test_status_reports_overlay_unavailable_for_a_run_without_overlay(tmp_path: Path) -> None:
    # The stub wrong-way run has no helmet observer, so no overlay is produced.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    detail = client.get(f"/api/process/{job_id}").json()
    assert detail["status"] == JobStatus.SUCCEEDED.value
    assert detail["overlay_available"] is False
    # Settled, not left pending: a client must never poll forever for an overlay
    # this run was never going to produce.
    assert detail["overlay_status"] == OverlayStatus.NONE.value


def test_overlay_is_announced_pending_before_the_job_goes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression guard for the bug that stranded the workspace on the raw upload.

    A client stops polling the instant it sees a terminal job status, so the status
    that *first* reports ``succeeded`` must already say an overlay is coming.
    Rendering is a second decode+encode pass finishing long after; with the marks in
    the other order the client could observe ``succeeded`` + "no overlay", stop
    polling, and never learn the annotated video was produced moments later.

    The window between the two marks is microseconds, so it is not observable from
    outside -- this asserts the ordering directly, which is the invariant that broke.
    """

    order: list[str] = []
    real_pending = JobStore.mark_overlay_pending
    real_succeeded = JobStore.mark_succeeded

    def spy_pending(self: JobStore, job_id: str) -> None:
        order.append("overlay_pending")
        real_pending(self, job_id)

    def spy_succeeded(self: JobStore, job_id: str, result: object) -> None:
        order.append("succeeded")
        real_succeeded(self, job_id, result)  # type: ignore[arg-type]

    monkeypatch.setattr(JobStore, "mark_overlay_pending", spy_pending)
    monkeypatch.setattr(JobStore, "mark_succeeded", spy_succeeded)

    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    assert order == ["overlay_pending", "succeeded"]


def test_a_pending_overlay_is_visible_to_a_client_while_the_render_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across a real background thread, the client sees pending -> settled.

    Exercises the production ``ThreadJobExecutor`` path the synchronous test
    executor cannot: the render is held open while the client polls, which is
    exactly the window in which the old contract reported "no overlay".
    """

    rendering = threading.Event()
    release = threading.Event()

    def blocking_render(**kwargs: object) -> None:
        rendering.set()
        assert release.wait(timeout=10.0)
        return None  # "nothing to draw" -- keeps this test off the encoder

    monkeypatch.setattr("trafficpulse.app.services.render_job_overlay", blocking_render)
    client = make_client(tmp_path, executor=ThreadJobExecutor())
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    assert rendering.wait(timeout=30.0)
    mid = client.get(f"/api/process/{job_id}").json()
    assert (mid["status"], mid["overlay_status"]) == (
        JobStatus.SUCCEEDED.value,
        OverlayStatus.PENDING.value,
    )

    release.set()
    for _ in range(200):  # the render thread settles the status shortly after
        settled = client.get(f"/api/process/{job_id}").json()
        if settled["overlay_status"] != OverlayStatus.PENDING.value:
            break
        time.sleep(0.05)
    assert settled["overlay_status"] == OverlayStatus.NONE.value


def test_a_failed_render_settles_the_overlay_without_failing_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs: object) -> None:
        raise RuntimeError("boom: encoder unavailable")

    monkeypatch.setattr("trafficpulse.app.services.render_job_overlay", boom)
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    detail = client.get(f"/api/process/{job_id}").json()
    # The overlay is presentation only: the run still succeeded and its events
    # are still queryable; only the annotated video is missing.
    assert detail["status"] == JobStatus.SUCCEEDED.value
    assert detail["overlay_status"] == OverlayStatus.FAILED.value
    assert detail["overlay_available"] is False


def test_overlay_endpoint_404_when_no_overlay(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    response = client.get(f"/api/process/{job_id}/overlay")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "overlay_not_found"


def test_overlay_endpoint_404_for_unknown_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/process/job-nope/overlay")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "job_not_found"


def _service(tmp_path: Path) -> tuple[ProcessingService, JobStore]:
    config = make_config(tmp_path)
    jobs = JobStore()
    service = ProcessingService(
        config=config,
        scenes=make_scene_service(config),
        provider=None,  # type: ignore[arg-type]  # unused by overlay_video_path
        store=EventStore(config.runs_dir),
        job_store=jobs,
        executor=SynchronousJobExecutor(),
        videos=VideoService(config, VideoStore()),
    )
    return service, jobs


def test_overlay_video_path_serves_a_recorded_artifact(tmp_path: Path) -> None:
    service, jobs = _service(tmp_path)
    jobs.add(JobRecord(job_id="job-1", video_id="vid-1", status=JobStatus.SUCCEEDED))
    artifact = tmp_path / "overlays" / "job-1.mp4"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"fake-mp4")
    jobs.set_overlay_video("job-1", artifact)

    assert service.overlay_video_path("job-1") == artifact


def test_overlay_video_path_raises_when_absent_or_unknown(tmp_path: Path) -> None:
    service, jobs = _service(tmp_path)
    jobs.add(JobRecord(job_id="job-2", video_id="vid-1", status=JobStatus.SUCCEEDED))

    try:
        service.overlay_video_path("job-2")
        raise AssertionError("expected OverlayNotFoundError")
    except OverlayNotFoundError:
        pass

    try:
        service.overlay_video_path("job-unknown")
        raise AssertionError("expected JobNotFoundError")
    except JobNotFoundError:
        pass

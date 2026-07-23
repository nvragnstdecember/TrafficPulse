"""Overlay-video endpoint + service behaviour (overlay integration)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import make_client, make_config, upload_wrong_way_video

from trafficpulse.app import SynchronousJobExecutor
from trafficpulse.app.errors import JobNotFoundError, OverlayNotFoundError
from trafficpulse.app.registry import JobRecord, JobStatus, JobStore, VideoStore
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
        scene=None,
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

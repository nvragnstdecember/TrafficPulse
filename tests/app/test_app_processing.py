"""Processing lifecycle, job status, and configuration errors (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import (
    FakeEngine,
    RaisingDetector,
    StubEngineProvider,
    make_client,
    make_config,
    make_metrics,
    upload_wrong_way_video,
)

from trafficpulse.app import AppConfig
from trafficpulse.app.registry import (
    JobRecord,
    JobStatus,
    JobStore,
    VideoStore,
)


def test_process_runs_to_completion_and_confirms_an_event(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)

    created = client.post("/api/process", json={"video_id": video_id})
    assert created.status_code == 202
    body = created.json()
    assert body["video_id"] == video_id
    job_id = body["job_id"]

    status = client.get(f"/api/process/{job_id}")
    assert status.status_code == 200
    detail = status.json()
    assert detail["status"] == JobStatus.SUCCEEDED.value
    assert detail["frames_processed"] == 30
    assert detail["frames_total"] == 30
    assert detail["progress"] == 1.0
    assert detail["fps"] == 10.0  # media-time (PTS-derived), deterministic
    assert detail["event_count"] == 1
    assert detail["error"] is None


def test_process_unknown_video_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/process", json={"video_id": "vid-does-not-exist"})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "video_not_found"


def test_status_unknown_job_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/process/job-nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "job_not_found"


def test_process_with_no_rules_configured_is_400(tmp_path: Path) -> None:
    config = make_config(tmp_path, default_rules=())
    client = make_client(tmp_path, config=config)
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_configuration"


def test_process_with_invalid_rule_for_scene_is_400(tmp_path: Path) -> None:
    # wrong_way with no direction on the two-direction example scene: the H6 rule
    # factory refuses it, and the service surfaces that as a 400.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post(
        "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_configuration"


def test_process_without_a_configured_scene_is_503(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "engine_unavailable"


def test_process_with_default_provider_and_no_inference_is_503(tmp_path: Path) -> None:
    # The default RealEngineProvider with a scene but no inference backend: its
    # create() raises EngineUnavailableError, which the service re-raises to a 503
    # (no ML framework is imported -- the None-inference branch fails first).
    from fastapi.testclient import TestClient

    from trafficpulse.app import create_app

    app = create_app(make_config(tmp_path))  # default provider, example scene, no inference
    with TestClient(app, raise_server_exceptions=False) as client:
        video_id = upload_wrong_way_video(client, tmp_path)
        response = client.post("/api/process", json={"video_id": video_id})
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "engine_unavailable"


def test_failed_job_reports_failed_status_not_500(tmp_path: Path) -> None:
    client = make_client(tmp_path, provider=StubEngineProvider(RaisingDetector))
    video_id = upload_wrong_way_video(client, tmp_path)
    created = client.post("/api/process", json={"video_id": video_id})
    # Submission succeeds (engine built); the failure happens during execution.
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    detail = client.get(f"/api/process/{job_id}").json()
    assert detail["status"] == JobStatus.FAILED.value
    assert detail["error"] and "boom" in detail["error"]


# --- job-status computation branches (unit) ------------------------------------
def _service(tmp_path: Path) -> object:
    from trafficpulse.app.registry import SynchronousJobExecutor
    from trafficpulse.app.services import ProcessingService, VideoService
    from trafficpulse.persistence import EventStore

    job_store = JobStore()
    return (
        ProcessingService(
            config=make_config(tmp_path),
            scene=None,
            provider=StubEngineProvider(),
            store=EventStore(tmp_path / "runs"),
            job_store=job_store,
            executor=SynchronousJobExecutor(),
            videos=VideoService(make_config(tmp_path), VideoStore()),
        ),
        job_store,
    )


def test_status_pending_reports_nulls(tmp_path: Path) -> None:
    service, job_store = _service(tmp_path)
    job_store.add(JobRecord(job_id="j", video_id="v", status=JobStatus.PENDING))
    status = service.status("j")  # type: ignore[attr-defined]
    assert status.frames_processed == 0
    assert status.progress is None
    assert status.fps is None
    assert status.estimated_remaining_seconds is None


def test_status_running_with_total_and_wall_fps(tmp_path: Path) -> None:
    service, job_store = _service(tmp_path)
    engine = FakeEngine(make_metrics(frames_processed=4, media_fps=5.0, wall_fps=2.0))
    job_store.add(
        JobRecord(
            job_id="j",
            video_id="v",
            status=JobStatus.RUNNING,
            frames_total=10,
            engine=engine,  # type: ignore[arg-type]
        )
    )
    status = service.status("j")  # type: ignore[attr-defined]
    assert status.progress == 0.4
    assert status.fps == 5.0
    assert status.estimated_remaining_seconds == 3.0  # (10 - 4) / 2.0


def test_status_running_without_total_has_null_progress(tmp_path: Path) -> None:
    service, job_store = _service(tmp_path)
    engine = FakeEngine(make_metrics(frames_processed=4, media_fps=5.0))
    job_store.add(
        JobRecord(
            job_id="j",
            video_id="v",
            status=JobStatus.RUNNING,
            frames_total=None,
            engine=engine,  # type: ignore[arg-type]
        )
    )
    status = service.status("j")  # type: ignore[attr-defined]
    assert status.progress is None
    assert status.fps == 5.0
    assert status.estimated_remaining_seconds is None


def test_status_succeeded_is_full_progress(tmp_path: Path) -> None:
    from trafficpulse.engine import EngineRunResult

    service, job_store = _service(tmp_path)
    result = EngineRunResult(
        source_id="s", events=(), manifests=(), metrics=make_metrics(frames_processed=10)
    )
    job_store.add(
        JobRecord(
            job_id="j",
            video_id="v",
            status=JobStatus.SUCCEEDED,
            frames_total=10,
            result=result,
        )
    )
    status = service.status("j")  # type: ignore[attr-defined]
    assert status.progress == 1.0
    assert status.frames_processed == 10


def test_from_env_config_can_build_an_app(tmp_path: Path) -> None:
    # The env path produces a usable AppConfig (host/port travel with it).
    config = AppConfig.from_env({"TRAFFICPULSE_APP_STORAGE": str(tmp_path)})
    client = make_client(tmp_path, config=config)
    assert client.get("/api/health").status_code == 200

"""Metrics endpoint: aggregate counts + reused H6 EngineMetrics (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import make_client, upload_wrong_way_video


def test_metrics_empty_before_any_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    body = client.get("/api/metrics").json()
    assert body["jobs_total"] == 0
    assert body["jobs_succeeded"] == 0
    assert body["events_total"] == 0
    assert body["latest"] is None


def test_metrics_reflect_a_completed_job(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    body = client.get("/api/metrics").json()
    assert body["jobs_total"] == 1
    assert body["jobs_succeeded"] == 1
    assert body["jobs_failed"] == 0
    assert body["events_total"] == 1
    # The latest block is the H6 EngineMetrics verbatim (not a recomputation).
    latest = body["latest"]
    assert latest is not None
    assert latest["frames_processed"] == 30
    assert latest["media_fps"] == 10.0
    assert latest["events_confirmed"] == 1


def test_metrics_count_failed_jobs(tmp_path: Path) -> None:
    from _app_helpers import RaisingDetector, StubEngineProvider

    client = make_client(tmp_path, provider=StubEngineProvider(RaisingDetector))
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    body = client.get("/api/metrics").json()
    assert body["jobs_failed"] == 1
    assert body["jobs_succeeded"] == 0

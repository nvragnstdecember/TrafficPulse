"""Startup recovery: rebuilding the runtime indices from persisted artifacts (H10)."""

from __future__ import annotations

import json
from pathlib import Path

from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient

from trafficpulse.app.recovery import (
    RUN_SNAPSHOT_NAME,
    RecoveryReport,
    RepositoryRecovery,
)
from trafficpulse.app.registry import JobStatus, JobStore, OverlayStatus, VideoStore


def _process(client: TestClient, tmp_path: Path, name: str = "clip.mp4") -> tuple[str, str]:
    """Upload + process one clip; return ``(video_id, job_id)``."""

    video_id = upload_wrong_way_video(client, tmp_path, name=name)
    job_id: str = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    return video_id, job_id


def _recover(tmp_path: Path) -> tuple[VideoStore, JobStore, RecoveryReport]:
    """Run recovery in isolation over a storage root."""

    recovery = RepositoryRecovery(make_config(tmp_path))
    videos, jobs = VideoStore(), JobStore()
    report = recovery.recover(videos=videos, jobs=jobs)
    return videos, jobs, report


# --- the success criterion --------------------------------------------------------
def test_a_restarted_application_serves_historical_runs_without_reprocessing(
    tmp_path: Path,
) -> None:
    # Process → persist → restart → everything the workspace needs still resolves,
    # with no upload and no inference on the second app.
    first = make_client(tmp_path)
    video_id, job_id = _process(first, tmp_path)
    events = first.get("/api/events").json()["items"]
    assert events, "the stub run must confirm an event for this to prove anything"
    event_id = events[0]["event_id"]

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    # the job resolves
    job = restarted.get(f"/api/process/{job_id}").json()
    assert job["status"] == JobStatus.SUCCEEDED.value
    assert job["video_id"] == video_id
    assert job["event_count"] == len(events)

    # the video's events resolve, filtered by video id -- the linkage that exists
    # nowhere in a ConfirmedEvent and is the whole reason a run snapshot is written
    recovered = restarted.get("/api/events", params={"video_id": video_id}).json()
    assert [item["event_id"] for item in recovered["items"]] == [
        item["event_id"] for item in events
    ]

    # detail + evidence resolve
    assert restarted.get(f"/api/events/{event_id}").status_code == 200
    assert restarted.get(f"/api/evidence/{event_id}").status_code == 200


def test_review_continues_after_a_restart(tmp_path: Path) -> None:
    # H9 wrote decisions to disk correctly; before H10 they were simply unreachable
    # because the event could not be located. The analyst must be able to carry on.
    first = make_client(tmp_path)
    _, _ = _process(first, tmp_path)
    event_id = first.get("/api/events").json()["items"][0]["event_id"]
    first.post(f"/api/events/{event_id}/review", json={"action": "open"})
    first.post(
        f"/api/events/{event_id}/review",
        json={"action": "approve", "note": "Confirmed before the restart"},
    )

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    body = restarted.get(f"/api/events/{event_id}/review").json()
    assert body["case"]["status"] == "approved"
    assert body["case"]["note"] == "Confirmed before the restart"
    assert [entry["action"] for entry in body["history"]] == ["open", "approve"]

    # ...and the workflow continues: reopen + re-decide on the restarted process.
    assert (
        restarted.post(f"/api/events/{event_id}/review", json={"action": "reopen"}).status_code
        == 200
    )
    final = restarted.post(
        f"/api/events/{event_id}/review", json={"action": "reject"}
    ).json()
    assert final["case"]["status"] == "rejected"
    assert [entry["action"] for entry in final["history"]] == [
        "open",
        "approve",
        "reopen",
        "reject",
    ]


def test_the_event_list_still_carries_review_status_after_a_restart(tmp_path: Path) -> None:
    first = make_client(tmp_path)
    _process(first, tmp_path)
    event_id = first.get("/api/events").json()["items"][0]["event_id"]
    first.post(f"/api/events/{event_id}/review", json={"action": "open"})
    first.post(f"/api/events/{event_id}/review", json={"action": "false_positive"})

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    badged = {
        item["event_id"]: item["review_status"]
        for item in restarted.get("/api/events").json()["items"]
    }
    assert badged[event_id] == "false_positive"


def test_the_uploaded_video_is_recovered_with_its_original_filename(tmp_path: Path) -> None:
    # The stored file is named by content id, so the client-supplied filename only
    # survives in the snapshot -- and the workspace displays it.
    first = make_client(tmp_path)
    video_id = upload_wrong_way_video(first, tmp_path, name="junction-north.mp4")

    videos, _, _ = _recover(tmp_path)
    record = videos.get(video_id)
    assert record is not None
    assert record.filename == "junction-north.mp4"
    assert record.path.is_file() and record.path.suffix == ".mp4"
    assert record.width == 320 and record.height == 240

    restarted = make_client(tmp_path, config=make_config(tmp_path))
    # Re-uploading identical bytes is now recognised as a duplicate, which is only
    # possible if the video record was recovered.
    clip = (tmp_path / "junction-north.mp4").read_bytes()
    response = restarted.post(
        "/api/video/upload", files={"file": ("junction-north.mp4", clip, "video/mp4")}
    )
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "duplicate_video"


def test_the_overlay_endpoint_resolves_after_a_restart(tmp_path: Path) -> None:
    first = make_client(tmp_path)
    _, job_id = _process(first, tmp_path)
    # The stub run produces no overlay metadata, so the honest recovered state is
    # "nothing to play" -- not a stale 'pending' that a client would poll forever.
    assert first.get(f"/api/process/{job_id}").json()["overlay_status"] == "none"

    restarted = make_client(tmp_path, config=make_config(tmp_path))
    status = restarted.get(f"/api/process/{job_id}").json()
    assert status["overlay_status"] == "none"
    assert status["overlay_available"] is False
    assert restarted.get(f"/api/process/{job_id}/overlay").status_code == 404


def test_a_ready_overlay_survives_a_restart(tmp_path: Path) -> None:
    # File presence is the authority on overlay availability after a restart.
    first = make_client(tmp_path)
    _, job_id = _process(first, tmp_path)
    overlays = make_config(tmp_path).overlays_dir
    overlays.mkdir(parents=True, exist_ok=True)
    (overlays / f"{job_id}.mp4").write_bytes(b"fake-mp4")

    restarted = make_client(tmp_path, config=make_config(tmp_path))
    status = restarted.get(f"/api/process/{job_id}").json()
    assert status["overlay_status"] == "ready"
    assert status["overlay_available"] is True


# --- recovery in isolation --------------------------------------------------------
def test_an_empty_repository_recovers_to_an_empty_index(tmp_path: Path) -> None:
    videos, jobs, report = _recover(tmp_path)
    assert report.is_empty
    assert report.skipped == ()
    assert jobs.jobs() == ()
    assert videos.get("vid-anything") is None


def test_multiple_runs_are_all_recovered_deterministically(tmp_path: Path) -> None:
    # Three runs over one upload: uploads are content-addressed, so reprocessing
    # the same clip is how a repository accumulates distinct runs.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    ids = [
        client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
        for _ in range(3)
    ]
    assert len(set(ids)) == 3

    _, jobs, report = _recover(tmp_path)
    assert report.runs_recovered == 3
    assert {record.job_id for record in jobs.jobs()} == set(ids)
    # Deterministic: a second recovery produces the same ordering.
    _, again, _ = _recover(tmp_path)
    assert [r.job_id for r in jobs.jobs()] == [r.job_id for r in again.jobs()]


def test_events_are_indexed_from_filenames_without_parsing_them(tmp_path: Path) -> None:
    # The performance guarantee: the event index is rebuilt from a directory
    # listing. Corrupting an event's *contents* must not affect indexing.
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]
    events_dir = make_config(tmp_path).runs_dir / job_id / "events"
    (events_dir / f"{event_id}.json").write_text("{ this is not json", encoding="utf-8")

    _, jobs, report = _recover(tmp_path)

    assert jobs.job_for_event(event_id) == job_id
    assert report.events_indexed >= 1


def test_an_interrupted_job_is_recovered_as_failed_not_running(tmp_path: Path) -> None:
    # A job whose worker died cannot progress. Restoring it as 'running' would
    # leave a client polling forever, so recovery settles it and says why.
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    snapshot_path = make_config(tmp_path).runs_dir / job_id / RUN_SNAPSHOT_NAME
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw["status"] = "running"
    snapshot_path.write_text(json.dumps(raw), encoding="utf-8")

    _, jobs, report = _recover(tmp_path)

    record = jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.error is not None and "restart" in record.error
    assert report.interrupted_jobs == 1


def test_a_pending_overlay_does_not_survive_as_pending(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    snapshot_path = make_config(tmp_path).runs_dir / job_id / RUN_SNAPSHOT_NAME
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw["overlay_status"] = "pending"
    snapshot_path.write_text(json.dumps(raw), encoding="utf-8")

    _, jobs, _ = _recover(tmp_path)

    record = jobs.get(job_id)
    assert record is not None
    # Nothing is rendering any more; a client must not be told to keep waiting.
    assert record.overlay_status is OverlayStatus.NONE


# --- corruption handling ----------------------------------------------------------
def test_a_corrupt_run_snapshot_is_skipped_without_failing_startup(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    good_job = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    bad_job = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    (make_config(tmp_path).runs_dir / bad_job / RUN_SNAPSHOT_NAME).write_text(
        "{ not json at all", encoding="utf-8"
    )

    _, jobs, report = _recover(tmp_path)

    assert jobs.get(good_job) is not None  # the healthy run still recovers
    assert jobs.get(bad_job) is None
    assert any(bad_job in line for line in report.skipped)


def test_a_snapshot_missing_required_fields_is_skipped(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    path = make_config(tmp_path).runs_dir / job_id / RUN_SNAPSHOT_NAME
    path.write_text(json.dumps({"job_id": job_id}), encoding="utf-8")  # no video_id

    _, jobs, report = _recover(tmp_path)

    assert jobs.get(job_id) is None
    assert any("not a valid RunSnapshot" in line for line in report.skipped)


def test_a_video_whose_media_was_deleted_is_not_recovered(tmp_path: Path) -> None:
    # Recovering it would advertise a playable upload whose bytes are gone.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    videos_dir = make_config(tmp_path).videos_dir
    for media in videos_dir.glob(f"{video_id}.*"):
        if media.suffix != ".json":
            media.unlink()

    videos, _, report = _recover(tmp_path)

    assert videos.get(video_id) is None
    assert any("no media file" in line for line in report.skipped)


def test_a_pre_h10_run_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    # A run persisted before snapshots existed has events on disk but nothing
    # recording which upload they belong to. Inventing that linkage would be
    # fabrication, so recovery reports it and moves on.
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    (make_config(tmp_path).runs_dir / job_id / RUN_SNAPSHOT_NAME).unlink()

    _, jobs, report = _recover(tmp_path)

    assert jobs.get(job_id) is None
    assert any("pre-H10 run" in line for line in report.skipped)


def test_the_review_journal_directory_is_not_mistaken_for_a_run(tmp_path: Path) -> None:
    # `runs/reviews/` is a sibling of the run directories, not one of them.
    client = make_client(tmp_path)
    _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]
    client.post(f"/api/events/{event_id}/review", json={"action": "open"})
    assert (make_config(tmp_path).runs_dir / "reviews").is_dir()

    _, _, report = _recover(tmp_path)

    assert not any("reviews" in line for line in report.skipped)


def test_a_missing_review_journal_is_simply_pending(tmp_path: Path) -> None:
    # Reviews need no recovery: an event nobody reviewed has no journal, and that
    # absence *is* the pending state.
    client = make_client(tmp_path)
    _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    body = restarted.get(f"/api/events/{event_id}/review").json()
    assert body["case"]["status"] == "pending"
    assert body["history"] == []


def test_recovery_does_not_rewrite_the_snapshots_it_reads(tmp_path: Path) -> None:
    # Startup must not perform pointless I/O -- restoring bypasses the observer.
    client = make_client(tmp_path)
    _, job_id = _process(client, tmp_path)
    path = make_config(tmp_path).runs_dir / job_id / RUN_SNAPSHOT_NAME
    before = path.stat().st_mtime_ns
    original = path.read_bytes()

    _recover(tmp_path)

    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == before

"""The analytics endpoint end to end, over a really-processed repository (H15)."""

from __future__ import annotations

import json
from pathlib import Path

from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient


def _process(client: TestClient, tmp_path: Path, *, name: str = "clip.mp4") -> str:
    video_id = upload_wrong_way_video(client, tmp_path, name=name)
    client.post("/api/process", json={"video_id": video_id})
    return video_id


def test_summary_of_an_empty_repository(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/analytics/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["repository"]["videos_total"] == 0
    assert body["violations"]["events_total"] == 0
    assert body["violations"]["by_type"] == []
    assert body["recent_activity"] == []
    assert body["latest_run"] is None
    assert body["health"]["engine"] == "ready"
    assert body["health"]["version"]


def test_summary_reflects_a_processed_repository(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _process(client, tmp_path)
    body = client.get("/api/analytics/summary").json()

    assert body["repository"]["videos_total"] == 1
    assert body["repository"]["videos_processed"] == 1
    assert body["repository"]["videos_unprocessed"] == 0
    assert body["processing"]["jobs_succeeded"] == 1
    assert body["violations"]["events_total"] > 0
    assert body["violations"]["by_type"], "a processed run yields a per-type breakdown"
    assert body["violations"]["uncounted_jobs"] == 0
    assert body["latest_run"] is not None

    # The breakdown must sum to the deduplicated total.
    assert sum(item["count"] for item in body["violations"]["by_type"]) == (
        body["violations"]["events_total"]
    )

    # Cross-surface agreement: analytics and the library describe the same repository.
    library = client.get("/api/videos").json()
    assert library["total"] == body["repository"]["videos_total"]
    assert library["items"][0]["event_count"] == body["violations"]["events_total"]


def test_processing_timing_is_recorded_and_reported(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _process(client, tmp_path)
    processing = client.get("/api/analytics/summary").json()["processing"]

    assert processing["timed_jobs"] == 1
    assert processing["average_duration_seconds"] is not None
    assert processing["average_duration_seconds"] >= 0.0
    assert processing["frames_processed"] > 0


def test_evidence_coverage_tracks_rendered_artifacts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _process(client, tmp_path)
    evidence = client.get("/api/analytics/summary").json()["evidence"]

    # H14 renders evidence for every confirmed event of a succeeded run.
    assert evidence["events_total"] > 0
    assert evidence["events_with_artifacts"] == evidence["events_total"]
    assert evidence["artifacts_total"] > 0
    assert evidence["artifact_bytes"] > 0


def test_review_progress_moves_when_an_analyst_acts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = _process(client, tmp_path)
    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]

    before = client.get("/api/analytics/summary").json()["review"]
    assert before["events_reviewed"] == 0
    assert before["events_pending"] == before["events_total"]

    client.post(
        f"/api/events/{event_id}/review",
        json={"action": "open", "reviewer": "analyst"},
    )

    after = client.get("/api/analytics/summary").json()["review"]
    assert after["events_reviewed"] == 1
    assert after["events_pending"] == after["events_total"] - 1


def test_recent_activity_mixes_uploads_runs_and_review(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = _process(client, tmp_path)
    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]
    client.post(
        f"/api/events/{event_id}/review",
        json={"action": "open", "reviewer": "analyst"},
    )

    activity = client.get("/api/analytics/summary").json()["recent_activity"]
    kinds = {entry["kind"] for entry in activity}
    assert kinds == {"upload", "run", "review"}
    # Newest first, and every entry carries a real wall-clock instant.
    stamps = [entry["at"] for entry in activity]
    assert stamps == sorted(stamps, reverse=True)
    assert not any(stamp.startswith("1970") for stamp in stamps)


def test_activity_never_presents_media_time_as_a_date(tmp_path: Path) -> None:
    """The trap H15 exists to avoid: every event's trigger_at is 1 Jan 1970."""

    client = make_client(tmp_path)
    video_id = _process(client, tmp_path)

    events = client.get("/api/events", params={"video_id": video_id}).json()["items"]
    assert events[0]["trigger_at"].startswith("1970"), "media time is epoch-anchored"

    activity = client.get("/api/analytics/summary").json()["recent_activity"]
    assert activity and all(not entry["at"].startswith("1970") for entry in activity)


def test_analytics_survives_a_restart_with_identical_figures(tmp_path: Path) -> None:
    """Recovered and live repositories must be indistinguishable (the H11 rule)."""

    first = make_client(tmp_path)
    _process(first, tmp_path)
    before = first.get("/api/analytics/summary").json()

    restarted = make_client(tmp_path)
    after = restarted.get("/api/analytics/summary").json()

    for section in ("repository", "processing", "violations", "evidence", "review"):
        assert after[section] == before[section], f"{section} changed across restart"


def test_a_pre_h15_snapshot_degrades_without_breaking(tmp_path: Path) -> None:
    """An older repository has no timing and no histogram; it must still serve."""

    client = make_client(tmp_path)
    _process(client, tmp_path)

    # Rewrite every run snapshot as a build predating H15 would have written it.
    for snapshot_path in make_config(tmp_path).runs_dir.glob("*/run.json"):
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for field in ("submitted_at", "started_at", "finished_at", "violation_counts"):
            snapshot.pop(field, None)
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    body = make_client(tmp_path).get("/api/analytics/summary").json()

    # Totals survive (they come from the event index, not the histogram) ...
    assert body["violations"]["events_total"] > 0
    # ... but the breakdown is honestly declared incomplete rather than faked.
    assert body["violations"]["by_type"] == []
    assert body["violations"]["counted_jobs"] == 0
    assert body["violations"]["uncounted_jobs"] == 1
    assert body["processing"]["average_duration_seconds"] is None
    assert body["processing"]["timed_jobs"] == 0
    assert body["health"]["runs_without_timing"] == 1


def test_multiple_videos_aggregate_across_the_repository(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _process(client, tmp_path, name="one.mp4")
    body = client.get("/api/analytics/summary").json()

    assert body["repository"]["videos_total"] == 1
    assert body["repository"]["footage_seconds"] is not None
    assert body["repository"]["storage_bytes"] > 0


def test_summary_opens_no_event_or_manifest_file(tmp_path: Path) -> None:
    """The performance contract: analytics is O(videos + jobs), never O(events).

    Asserted structurally by counting reads of the event store, which is the only
    route to an event or manifest file.
    """

    client = make_client(tmp_path)
    _process(client, tmp_path)

    from trafficpulse.persistence import EventStore

    calls: list[str] = []
    original = EventStore.load

    def counting_load(self: EventStore, run_id: str) -> object:
        calls.append(run_id)
        return original(self, run_id)

    EventStore.load = counting_load  # type: ignore[method-assign]
    try:
        assert client.get("/api/analytics/summary").status_code == 200
    finally:
        EventStore.load = original  # type: ignore[method-assign]

    assert calls == [], f"analytics deserialised events from runs {calls}"

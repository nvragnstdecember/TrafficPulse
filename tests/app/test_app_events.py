"""Event list (filter/sort/paginate) + detail (H7A)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from _app_helpers import make_client, make_metrics, upload_wrong_way_video

from trafficpulse.app.models import EventSort
from trafficpulse.app.registry import JobRecord, JobStore
from trafficpulse.app.services import EventService
from trafficpulse.contracts import ConfirmedEvent
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.engine import EngineRunResult
from trafficpulse.persistence import EventStore


# --- HTTP happy path -----------------------------------------------------------
def test_list_and_detail_over_a_real_run(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})

    listing = client.get("/api/events", params={"video_id": video_id})
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    summary = body["items"][0]
    assert summary["violation_type"] == ViolationType.WRONG_WAY.value
    assert summary["video_id"] == video_id

    detail = client.get(f"/api/events/{summary['event_id']}")
    assert detail.status_code == 200
    full = detail.json()
    assert full["event_id"] == summary["event_id"]
    assert full["rule_id"] == summary["rule_id"]
    assert "measurements" in full  # the complete contract, not the summary


def test_event_detail_unknown_id_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/events/evt-nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "event_not_found"


def test_list_empty_without_processing(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    body = client.get("/api/events").json()
    assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}


# --- filter / sort / paginate over a seeded store (deterministic) --------------
def _event(event_id: str, *, trigger_offset: float, video: str) -> ConfirmedEvent:
    at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=trigger_offset)
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id=f"cam-{video}",
        track_ids=("iou-1",),
        start_at=at,
        trigger_at=at,
        rule_id="wrong_way",
        created_at=at,
    )


def _seeded_events(tmp_path: Path) -> EventService:
    """Two videos, three events, persisted + indexed -- no engine involved."""

    store = EventStore(tmp_path / "runs")
    job_store = JobStore()
    plan = {
        "job-a": ("vid-a", [_event("evt-a1", trigger_offset=3.0, video="a"),
                            _event("evt-a2", trigger_offset=1.0, video="a")]),
        "job-b": ("vid-b", [_event("evt-b1", trigger_offset=2.0, video="b")]),
    }
    for job_id, (video_id, events) in plan.items():
        store.persist(job_id, events)  # write-once JSON + stub manifests
        job_store.add(JobRecord(job_id=job_id, video_id=video_id))
        # mark_succeeded records status, event ids, and the event->job index.
        result = EngineRunResult(
            source_id=job_id, events=tuple(events), manifests=(), metrics=make_metrics()
        )
        job_store.mark_succeeded(job_id, result)
    return EventService(store, job_store)


def test_list_sorted_by_trigger_ascending(tmp_path: Path) -> None:
    events = _seeded_events(tmp_path)
    page = events.list(video_id=None, limit=50, offset=0, sort=EventSort.TRIGGER_AT_ASC)
    assert page.total == 3
    assert [item.event_id for item in page.items] == ["evt-a2", "evt-b1", "evt-a1"]


def test_list_sorted_by_trigger_descending(tmp_path: Path) -> None:
    events = _seeded_events(tmp_path)
    page = events.list(video_id=None, limit=50, offset=0, sort=EventSort.TRIGGER_AT_DESC)
    assert [item.event_id for item in page.items] == ["evt-a1", "evt-b1", "evt-a2"]


def test_list_sorted_by_event_id(tmp_path: Path) -> None:
    events = _seeded_events(tmp_path)
    page = events.list(video_id=None, limit=50, offset=0, sort=EventSort.EVENT_ID_DESC)
    assert [item.event_id for item in page.items] == ["evt-b1", "evt-a2", "evt-a1"]


def test_list_filters_by_video(tmp_path: Path) -> None:
    events = _seeded_events(tmp_path)
    page = events.list(video_id="vid-a", limit=50, offset=0, sort=EventSort.EVENT_ID_ASC)
    assert page.total == 2
    assert {item.video_id for item in page.items} == {"vid-a"}


def test_list_paginates(tmp_path: Path) -> None:
    events = _seeded_events(tmp_path)
    page = events.list(video_id=None, limit=1, offset=1, sort=EventSort.TRIGGER_AT_ASC)
    assert page.total == 3
    assert page.limit == 1 and page.offset == 1
    assert [item.event_id for item in page.items] == ["evt-b1"]


def test_list_invalid_pagination_is_422(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/api/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/events", params={"offset": -1}).status_code == 422


def test_list_skips_succeeded_jobs_with_no_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "runs")
    job_store = JobStore()
    event = _event("evt-x", trigger_offset=1.0, video="a")
    store.persist("job-events", [event])
    job_store.add(JobRecord(job_id="job-events", video_id="v"))
    job_store.mark_succeeded(
        "job-events",
        EngineRunResult(source_id="a", events=(event,), manifests=(), metrics=make_metrics()),
    )
    # A succeeded job that confirmed nothing persisted no run -- it must be skipped.
    job_store.add(JobRecord(job_id="job-empty", video_id="v"))
    job_store.mark_succeeded(
        "job-empty",
        EngineRunResult(source_id="e", events=(), manifests=(), metrics=make_metrics()),
    )
    events = EventService(store, job_store)
    page = events.list(video_id="v", limit=50, offset=0, sort=EventSort.EVENT_ID_ASC)
    assert page.total == 1


def test_list_deduplicates_events_across_runs(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "runs")
    job_store = JobStore()
    event = _event("evt-dup", trigger_offset=1.0, video="a")
    for job_id in ("job-1", "job-2"):  # same clip reprocessed -> same event id
        store.persist(job_id, [event])
        job_store.add(JobRecord(job_id=job_id, video_id="v"))
        job_store.mark_succeeded(
            job_id,
            EngineRunResult(
                source_id=job_id, events=(event,), manifests=(), metrics=make_metrics()
            ),
        )
    events = EventService(store, job_store)
    page = events.list(video_id=None, limit=50, offset=0, sort=EventSort.EVENT_ID_ASC)
    assert page.total == 1  # deduplicated by event id

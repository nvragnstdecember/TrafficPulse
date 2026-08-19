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


# --- run scoping (R7) ------------------------------------------------------------
def _reprocessed(tmp_path: Path) -> tuple[EventService, JobStore]:
    """One video processed twice: a wide first run, a narrower second one.

    ``evt-shared`` is confirmed by both runs -- reprocessing re-confirms identical
    content-derived ids -- while ``evt-only-1`` exists solely in the first. That is
    the shape the reported bug appears in: the second run did not find it, but a
    video-scoped listing still returned it.
    """

    store = EventStore(tmp_path / "runs")
    job_store = JobStore()
    shared = _event("evt-shared", trigger_offset=1.0, video="a")
    dropped = _event("evt-only-1", trigger_offset=2.0, video="a")
    plan = {"job-1": [shared, dropped], "job-2": [shared]}
    for job_id, events in plan.items():
        store.persist(job_id, events)
        job_store.add(JobRecord(job_id=job_id, video_id="vid-a"))
        job_store.mark_succeeded(
            job_id,
            EngineRunResult(
                source_id=job_id, events=tuple(events), manifests=(), metrics=make_metrics()
            ),
        )
    return EventService(store, job_store), job_store


def _ids(page: object) -> list[str]:
    return [summary.event_id for summary in page.items]  # type: ignore[attr-defined]


def _list(events: EventService, **kwargs: object) -> object:
    params: dict[str, object] = {
        "video_id": None,
        "limit": 50,
        "offset": 0,
        "sort": EventSort.EVENT_ID_ASC,
    }
    params.update(kwargs)
    return events.list(**params)  # type: ignore[arg-type]


def test_scoping_to_a_run_returns_only_that_runs_events(tmp_path: Path) -> None:
    events, _ = _reprocessed(tmp_path)

    assert _ids(_list(events, video_id="vid-a", job_id="job-1")) == [
        "evt-only-1",
        "evt-shared",
    ]
    assert _ids(_list(events, video_id="vid-a", job_id="job-2")) == ["evt-shared"]


def test_a_narrower_rerun_does_not_show_the_superseded_runs_events(
    tmp_path: Path,
) -> None:
    """The reported bug, as a regression guard."""

    events, _ = _reprocessed(tmp_path)

    page = _list(events, video_id="vid-a", job_id="job-2")
    assert "evt-only-1" not in _ids(page)
    assert page.total == 1  # type: ignore[attr-defined]


def test_without_a_run_the_whole_history_is_still_returned(tmp_path: Path) -> None:
    """The repository view is unchanged: both runs, deduplicated by event id."""

    events, _ = _reprocessed(tmp_path)

    page = _list(events, video_id="vid-a")
    assert _ids(page) == ["evt-only-1", "evt-shared"]
    assert page.total == 2  # type: ignore[attr-defined]


def test_a_duplicated_event_is_attributed_to_the_run_serving_its_evidence(
    tmp_path: Path,
) -> None:
    """``EventSummary.job_id`` must agree with where ``locate`` reads the event.

    Both point at the newest run that produced it. They disagreed before R7 -- the
    listing labelled such an event with the *oldest* run while its detail and
    evidence were served from the newest -- so a client could not use the label to
    address anything.
    """

    events, job_store = _reprocessed(tmp_path)

    summary = next(
        item for item in _list(events, video_id="vid-a").items  # type: ignore[attr-defined]
        if item.event_id == "evt-shared"
    )
    assert summary.job_id == "job-2"
    assert summary.job_id == job_store.job_for_event("evt-shared")


def test_a_run_of_another_video_selects_nothing(tmp_path: Path) -> None:
    """A mismatched pair describes no run, so it is not silently widened."""

    events, _ = _reprocessed(tmp_path)
    seeded = _seeded_events(tmp_path / "seeded")

    assert _list(events, video_id="vid-b", job_id="job-1").total == 0  # type: ignore[attr-defined]
    assert _list(seeded, video_id="vid-a", job_id="job-b").total == 0  # type: ignore[attr-defined]


def test_an_unknown_or_unfinished_run_yields_an_empty_page(tmp_path: Path) -> None:
    """Empty, not an error -- the same shape an unknown video_id has always had."""

    events, job_store = _reprocessed(tmp_path)
    job_store.add(JobRecord(job_id="job-running", video_id="vid-a"))
    job_store.mark_running("job-running", frames_total=10)
    job_store.add(JobRecord(job_id="job-failed", video_id="vid-a"))
    job_store.mark_failed("job-failed", "boom")

    for job_id in ("job-nope", "job-running", "job-failed"):
        assert _list(events, job_id=job_id).total == 0, job_id  # type: ignore[attr-defined]


def test_run_scoping_composes_with_sorting_and_paging(tmp_path: Path) -> None:
    events, _ = _reprocessed(tmp_path)

    descending = _list(
        events, video_id="vid-a", job_id="job-1", sort=EventSort.EVENT_ID_DESC
    )
    assert _ids(descending) == ["evt-shared", "evt-only-1"]

    paged = _list(events, video_id="vid-a", job_id="job-1", limit=1, offset=1)
    assert _ids(paged) == ["evt-shared"]
    assert paged.total == 2  # type: ignore[attr-defined]


def test_the_endpoint_accepts_a_run_filter(tmp_path: Path) -> None:
    """Over HTTP: the query parameter reaches the service and narrows the page."""

    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    job_id = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    scoped = client.get("/api/events", params={"video_id": video_id, "job_id": job_id})
    assert scoped.status_code == 200
    assert [item["job_id"] for item in scoped.json()["items"]] == [job_id]

    other = client.get("/api/events", params={"video_id": video_id, "job_id": "job-nope"})
    assert other.status_code == 200
    assert other.json()["total"] == 0

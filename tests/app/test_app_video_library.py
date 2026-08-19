"""The historical video library: browse, describe, and play back stored videos (H11).

The endpoints that make persisted work *discoverable*. H10 proved a restarted
backend can still serve a video's events when a client already holds its id; these
tests prove a client can find that id in the first place, and that a recovered
video is indistinguishable from one uploaded in the same session.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from _app_helpers import (
    NO_OVERLAY_RULES,
    make_client,
    make_config,
    upload_wrong_way_video,
)
from _slice_fixtures import FRAME_COUNT, write_wrong_way_clip
from fastapi.testclient import TestClient

from trafficpulse.app.models import VideoSort
from trafficpulse.app.registry import JobStore, VideoRecord, VideoStore
from trafficpulse.app.services import VideoLibraryService


def _bare_record(video_id: str, *, uploaded_at: datetime | None) -> VideoRecord:
    """A minimal :class:`VideoRecord` for service-level ordering tests (no I/O)."""

    return VideoRecord(
        video_id=video_id,
        filename=f"{video_id}.mp4",
        path=Path(f"{video_id}.mp4"),
        size_bytes=1,
        width=320,
        height=240,
        fps=10.0,
        frame_count=30,
        duration_seconds=3.0,
        codec="mpeg4",
        uploaded_at=uploaded_at,
    )


def _process(client: TestClient, tmp_path: Path, name: str = "clip.mp4") -> tuple[str, str]:
    """Upload + process one clip; return ``(video_id, job_id)``."""

    video_id = upload_wrong_way_video(client, tmp_path, name=name)
    job_id: str = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    return video_id, job_id


def _upload_distinct(client: TestClient, tmp_path: Path, name: str, frames: int) -> str:
    """Upload a clip with distinct *content*, so it gets its own video id.

    Uploads are content-addressed, so two byte-identical clips are one video however
    they are named. Varying the frame count is the cheapest way to populate a
    library with several genuinely different videos.
    """

    clip = write_wrong_way_clip(tmp_path / name, frames=frames)
    response = client.post(
        "/api/video/upload", files={"file": (name, clip.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 201, response.text
    video_id: str = response.json()["video_id"]
    return video_id


def _library(client: TestClient, **params: object) -> dict[str, object]:
    response = client.get("/api/videos", params=params)
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


# --- the success criterion ---------------------------------------------------------
def test_a_processed_video_is_browsable_after_a_restart(tmp_path: Path) -> None:
    # Process -> persist -> restart -> browse -> select -> the analysis is all there,
    # with nothing re-uploaded and no inference on the second app. This is the whole
    # milestone in one test.
    first = make_client(tmp_path)
    video_id, job_id = _process(first, tmp_path, name="junction-north.mp4")
    events = first.get("/api/events").json()["items"]
    assert events, "the stub run must confirm an event for this to prove anything"

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    # Browse: the library enumerates work the client never told it about.
    body = _library(restarted)
    assert body["total"] == 1
    row = body["items"][0]
    assert row["video_id"] == video_id
    assert row["filename"] == "junction-north.mp4"
    assert row["status"] == "succeeded"
    assert row["job_id"] == job_id
    assert row["event_count"] == len(events)
    assert row["media_available"] is True

    # Select: every downstream endpoint resolves from the row alone.
    assert restarted.get(f"/api/videos/{video_id}/media").status_code == 200
    listed = restarted.get("/api/events", params={"video_id": video_id}).json()
    assert [item["event_id"] for item in listed["items"]] == [
        item["event_id"] for item in events
    ]
    assert restarted.get(f"/api/process/{job_id}").json()["status"] == "succeeded"


def test_a_recovered_video_lists_identically_to_a_live_one(tmp_path: Path) -> None:
    # The frontend must not be able to tell the two apart -- if it could, "recovered"
    # would become a state the UI has to explain, and the library would be a second
    # class of video rather than the same one.
    live = make_client(tmp_path)
    _process(live, tmp_path)
    before = _library(live)["items"][0]

    restarted = make_client(tmp_path, config=make_config(tmp_path))
    after = _library(restarted)["items"][0]

    assert after == before


# --- empty and degraded repositories ----------------------------------------------
def test_an_empty_repository_lists_nothing_rather_than_failing(tmp_path: Path) -> None:
    body = _library(make_client(tmp_path))

    assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_a_repository_whose_snapshots_are_corrupt_still_lists_what_survived(
    tmp_path: Path,
) -> None:
    # Corruption is contained to the entity it belongs to: the library shows the
    # healthy videos instead of the whole page failing.
    client = make_client(tmp_path)
    good, _ = _process(client, tmp_path, name="good.mp4")
    bad = _upload_distinct(client, tmp_path, "bad.mp4", frames=FRAME_COUNT - 5)
    (make_config(tmp_path).videos_dir / f"{bad}.json").write_text("{ not json", encoding="utf-8")

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    body = _library(restarted)
    assert [row["video_id"] for row in body["items"]] == [good]


def test_a_video_whose_file_was_deleted_reports_it_rather_than_disappearing(
    tmp_path: Path,
) -> None:
    # Within a live session the record outlives the bytes. Dropping the row would
    # hide a video whose events and review history are still perfectly valid, so the
    # library keeps it and says playback is unavailable.
    client = make_client(tmp_path)
    video_id, _ = _process(client, tmp_path)
    for media in make_config(tmp_path).videos_dir.glob(f"{video_id}.*"):
        if media.suffix != ".json":
            media.unlink()

    row = _library(client)["items"][0]
    assert row["media_available"] is False
    assert row["event_count"] >= 1  # the analysis is untouched

    response = client.get(f"/api/videos/{video_id}/media")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "video_media_not_found"


def test_an_unknown_video_is_a_typed_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/api/videos/vid-nope")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "video_not_found"


# --- metadata correctness ----------------------------------------------------------
def test_an_uploaded_but_unprocessed_video_reports_no_job(tmp_path: Path) -> None:
    # Null, not a fabricated 'pending': nothing was ever submitted for this video.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)

    row = client.get(f"/api/videos/{video_id}").json()

    assert row["job_id"] is None
    assert row["status"] is None
    assert row["job_count"] == 0
    assert row["event_count"] == 0
    assert row["media_available"] is True


def test_decoded_metadata_is_reported_from_the_upload(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    upload = client.get(f"/api/videos/{video_id}").json()

    assert (upload["width"], upload["height"]) == (320, 240)
    assert upload["codec"]
    assert upload["size_bytes"] > 0
    assert datetime.fromisoformat(upload["uploaded_at"]) <= datetime.now(UTC)


def test_the_opened_job_is_the_latest_successful_run(tmp_path: Path) -> None:
    # Reprocessing is how a repository accumulates runs. "Open this video" means its
    # analysis, so a later failure must not hide an earlier success.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    first = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]
    second = client.post("/api/process", json={"video_id": video_id}).json()["job_id"]

    row = client.get(f"/api/videos/{video_id}").json()

    assert row["job_count"] == 2
    assert row["job_id"] == second
    assert row["job_id"] != first
    assert row["status"] == "succeeded"


def test_events_are_counted_once_across_repeated_runs(tmp_path: Path) -> None:
    # Event ids are content-derived, so reprocessing confirms the *same* violations.
    # Summing per-run counts would tell an analyst a clip has twice the offences it
    # has -- and would disagree with the event list the same row links to.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    for _ in range(3):
        client.post("/api/process", json={"video_id": video_id})

    row = client.get(f"/api/videos/{video_id}").json()
    listed = client.get("/api/events", params={"video_id": video_id}).json()

    assert row["job_count"] == 3
    assert row["event_count"] == listed["total"]


def test_review_progress_is_reported_per_video(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id, _ = _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]

    assert client.get(f"/api/videos/{video_id}").json()["events_reviewed"] == 0

    client.post(f"/api/events/{event_id}/review", json={"action": "open"})

    row = client.get(f"/api/videos/{video_id}").json()
    assert row["events_reviewed"] == 1
    assert row["events_reviewed"] <= row["event_count"]


def test_review_progress_survives_a_restart(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id, _ = _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]
    client.post(f"/api/events/{event_id}/review", json={"action": "open"})
    client.post(f"/api/events/{event_id}/review", json={"action": "approve"})

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    assert restarted.get(f"/api/videos/{video_id}").json()["events_reviewed"] == 1


def test_a_ready_overlay_is_advertised_on_the_library_row(tmp_path: Path) -> None:
    # NO_OVERLAY_RULES so the run renders none of its own: this asserts that a file
    # appearing on disk is what advertises an overlay, which needs a run that made no
    # overlay itself.
    client = make_client(tmp_path, config=make_config(tmp_path, default_rules=NO_OVERLAY_RULES))
    video_id, job_id = _process(client, tmp_path)
    assert client.get(f"/api/videos/{video_id}").json()["overlay_available"] is False

    overlays = make_config(tmp_path).overlays_dir
    overlays.mkdir(parents=True, exist_ok=True)
    (overlays / f"{job_id}.mp4").write_bytes(b"fake-mp4")
    restarted = make_client(tmp_path, config=make_config(tmp_path))

    assert restarted.get(f"/api/videos/{video_id}").json()["overlay_available"] is True


# --- ordering + paging -------------------------------------------------------------
def test_the_default_ordering_is_most_recently_uploaded_first(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    first = _upload_distinct(client, tmp_path, "a.mp4", frames=FRAME_COUNT)
    second = _upload_distinct(client, tmp_path, "b.mp4", frames=FRAME_COUNT - 2)
    third = _upload_distinct(client, tmp_path, "c.mp4", frames=FRAME_COUNT - 4)

    ids = [row["video_id"] for row in _library(client)["items"]]

    assert ids == [third, second, first]


def test_ordering_is_reversible_and_available_by_filename(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    _upload_distinct(client, tmp_path, "zebra.mp4", frames=FRAME_COUNT)
    _upload_distinct(client, tmp_path, "alpha.mp4", frames=FRAME_COUNT - 2)

    oldest = [row["filename"] for row in _library(client, sort="uploaded_at")["items"]]
    by_name = [row["filename"] for row in _library(client, sort="filename")["items"]]

    assert oldest == ["zebra.mp4", "alpha.mp4"]
    assert by_name == ["alpha.mp4", "zebra.mp4"]


def test_a_pre_h11_upload_recovers_its_instant_from_the_stored_file(
    tmp_path: Path,
) -> None:
    # A snapshot written before the timestamp existed still has to be placeable on
    # the time axis, or every older video would sink to the bottom of the library.
    # The file's mtime is when its bytes were written, which *is* when the upload was
    # accepted -- a recovered fact, not a substituted one.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    snapshot = make_config(tmp_path).videos_dir / f"{video_id}.json"
    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    assert raw.pop("uploaded_at") is not None
    snapshot.write_text(json.dumps(raw), encoding="utf-8")

    restarted = make_client(tmp_path, config=make_config(tmp_path))

    row = restarted.get(f"/api/videos/{video_id}").json()
    assert row["uploaded_at"] is not None
    media = next(
        path
        for path in make_config(tmp_path).videos_dir.glob(f"{video_id}.*")
        if path.suffix != ".json"
    )
    recovered = datetime.fromisoformat(row["uploaded_at"])
    assert recovered == datetime.fromtimestamp(media.stat().st_mtime, UTC)


def test_a_video_with_no_known_instant_sorts_last_in_both_directions() -> None:
    # Unknown is not "the beginning of time" and not "now". Substituting either
    # would silently interleave undatable videos through somebody's library; sorting
    # them to the end says what is actually true in either direction.
    videos, jobs = VideoStore(), JobStore()
    videos.restore(_bare_record("vid-undated", uploaded_at=None))
    videos.restore(_bare_record("vid-older", uploaded_at=datetime(2026, 1, 1, tzinfo=UTC)))
    videos.restore(_bare_record("vid-newer", uploaded_at=datetime(2026, 6, 1, tzinfo=UTC)))
    library = VideoLibraryService(videos, jobs)

    newest_first = library.list(limit=10, offset=0, sort=VideoSort.UPLOADED_AT_DESC)
    oldest_first = library.list(limit=10, offset=0, sort=VideoSort.UPLOADED_AT_ASC)

    assert [row.video_id for row in newest_first.items] == [
        "vid-newer",
        "vid-older",
        "vid-undated",
    ]
    assert [row.video_id for row in oldest_first.items] == [
        "vid-older",
        "vid-newer",
        "vid-undated",
    ]


def test_paging_is_deterministic_and_reports_the_full_total(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for index in range(3):
        _upload_distinct(client, tmp_path, f"clip-{index}.mp4", frames=FRAME_COUNT - index)

    first = _library(client, limit=2, offset=0)
    second = _library(client, limit=2, offset=2)

    assert first["total"] == second["total"] == 3
    assert len(first["items"]) == 2 and len(second["items"]) == 1
    assert first["limit"] == 2 and second["offset"] == 2
    ids = [row["video_id"] for row in first["items"] + second["items"]]
    assert len(set(ids)) == 3
    assert ids == [row["video_id"] for row in _library(client)["items"]]


def test_paging_bounds_are_validated(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    assert client.get("/api/videos", params={"limit": 0}).status_code == 422
    assert client.get("/api/videos", params={"limit": 500}).status_code == 422
    assert client.get("/api/videos", params={"offset": -1}).status_code == 422
    assert client.get("/api/videos", params={"sort": "nonsense"}).status_code == 422


# --- media playback ----------------------------------------------------------------
def test_the_stored_source_video_streams_back_byte_for_byte(tmp_path: Path) -> None:
    # The upload is never modified: the annotated overlay is a separate artifact, and
    # what an analyst plays for an un-overlaid run is exactly what they uploaded.
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    original = (tmp_path / "clip.mp4").read_bytes()

    response = client.get(f"/api/videos/{video_id}/media")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == original


def test_media_supports_range_requests_so_the_player_can_seek(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)

    response = client.get(f"/api/videos/{video_id}/media", headers={"Range": "bytes=0-15"})

    assert response.status_code == 206
    assert len(response.content) == 16


# --- listing cost ------------------------------------------------------------------
def test_listing_does_not_deserialise_events_or_review_journals(tmp_path: Path) -> None:
    # The performance contract: a listing is metadata only. Corrupting every event
    # record and every review journal must not affect it -- if the library opened
    # them, this would raise instead of listing.
    client = make_client(tmp_path)
    video_id, job_id = _process(client, tmp_path)
    event_id = client.get("/api/events").json()["items"][0]["event_id"]
    client.post(f"/api/events/{event_id}/review", json={"action": "open"})

    runs = make_config(tmp_path).runs_dir
    for record in (runs / job_id / "events").glob("*.json"):
        record.write_text("{ not an event", encoding="utf-8")
    for journal in (runs / "reviews").glob("*.jsonl"):
        journal.write_text("not a review entry\n", encoding="utf-8")

    restarted = make_client(tmp_path, config=make_config(tmp_path))
    row = restarted.get(f"/api/videos/{video_id}").json()

    assert row["event_count"] >= 1
    assert row["events_reviewed"] == 1

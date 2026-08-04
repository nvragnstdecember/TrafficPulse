"""The analytics performance contract (H16).

H15 claimed the dashboard was O(videos + jobs). Two paths broke it: artifact
storage statistics walked the whole artifact store, and the activity feed opened
every review journal in the repository -- on every request, polled every 30
seconds. These tests pin the fixes as *behaviour*, not as comments.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _app_helpers import make_client, upload_wrong_way_video
from fastapi.testclient import TestClient

from trafficpulse.contracts.enums import ArtifactKind
from trafficpulse.evidence import ArtifactStore
from trafficpulse.persistence import ReviewStore


def _process(client: TestClient, tmp_path: Path) -> str:
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    return video_id


# --- artifact usage --------------------------------------------------------------
def test_usage_is_measured_once_then_maintained(tmp_path: Path) -> None:
    """The store is the only writer, so the figure is kept rather than rediscovered."""

    store = ArtifactStore(tmp_path)
    assert store.usage() == (0, 0)

    store.put(b"x" * 10, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    assert store.usage() == (1, 10)

    store.put(b"y" * 5, kind=ArtifactKind.BEFORE_FRAME, media_type="image/png")
    assert store.usage() == (2, 15)


def test_storing_identical_bytes_does_not_double_count(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.put(b"same", kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    store.put(b"same", kind=ArtifactKind.AFTER_FRAME, media_type="image/png")
    assert store.usage() == (1, 4)  # one file on disk, counted once


def test_a_cold_store_measures_what_is_actually_on_disk(tmp_path: Path) -> None:
    """A fresh process trusts the filesystem, not a number it did not compute."""

    seeded = ArtifactStore(tmp_path)
    seeded.put(b"abc", kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")

    restarted = ArtifactStore(tmp_path)  # new process, cold cache
    assert restarted.usage() == (1, 3)


def test_usage_does_not_walk_the_store_on_every_ask(tmp_path: Path) -> None:
    """The regression: an O(artifacts) filesystem walk per dashboard request."""

    store = ArtifactStore(tmp_path)
    store.put(b"abc", kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    store.usage()  # warm the figure

    calls: list[Path] = []
    original = Path.rglob

    def counting_rglob(self: Path, pattern: str) -> object:
        calls.append(self)
        return original(self, pattern)

    Path.rglob = counting_rglob  # type: ignore[method-assign]
    try:
        for _ in range(5):
            store.usage()
    finally:
        Path.rglob = original  # type: ignore[method-assign]

    assert calls == [], "usage re-walked the artifact store"


# --- review activity -------------------------------------------------------------
def test_recent_journals_are_selected_without_reading_them(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    assert store.recently_reviewed_event_ids(5) == ()

    directory = tmp_path / "reviews"
    directory.mkdir(parents=True)
    for index in range(6):
        (directory / f"evt-{index}.jsonl").write_text("{}\n", encoding="utf-8")

    recent = store.recently_reviewed_event_ids(3)
    assert len(recent) == 3
    assert set(recent) <= {f"evt-{index}" for index in range(6)}


def test_a_zero_limit_reads_nothing(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    (tmp_path / "reviews").mkdir(parents=True)
    (tmp_path / "reviews" / "evt-1.jsonl").write_text("{}\n", encoding="utf-8")
    assert store.recently_reviewed_event_ids(0) == ()


def test_selection_is_deterministic(tmp_path: Path) -> None:
    """Coarse mtime resolution must not make the feed reshuffle between reads."""

    store = ReviewStore(tmp_path)
    directory = tmp_path / "reviews"
    directory.mkdir(parents=True)
    for index in range(4):
        (directory / f"evt-{index}.jsonl").write_text("{}\n", encoding="utf-8")

    assert store.recently_reviewed_event_ids(3) == store.recently_reviewed_event_ids(3)


@pytest.mark.parametrize("reviewed", [1, 8])
def test_the_activity_feed_opens_only_a_bounded_number_of_journals(
    tmp_path: Path, reviewed: int
) -> None:
    """Whole-repository journal reads per request were the H16 finding."""

    client = make_client(tmp_path)
    video_id = _process(client, tmp_path)
    events = client.get("/api/events", params={"video_id": video_id}).json()["items"]
    for summary in events[:reviewed]:
        client.post(
            f"/api/events/{summary['event_id']}/review",
            json={"action": "open", "reviewer": "analyst"},
        )

    opened: list[str] = []
    original = ReviewStore.history

    def counting_history(self: ReviewStore, event_id: str) -> object:
        opened.append(event_id)
        return original(self, event_id)

    ReviewStore.history = counting_history  # type: ignore[method-assign]
    try:
        assert client.get("/api/analytics/summary").status_code == 200
    finally:
        ReviewStore.history = original  # type: ignore[method-assign]

    # Never more than the feed can display, whatever the repository holds.
    assert len(opened) <= 12


def test_the_summary_still_opens_no_event_file(tmp_path: Path) -> None:
    """The H15 contract, re-asserted after the H16 optimisation."""

    from trafficpulse.persistence import EventStore

    client = make_client(tmp_path)
    _process(client, tmp_path)

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

    assert calls == []


def test_the_optimisation_did_not_change_what_is_reported(tmp_path: Path) -> None:
    """Correctness is preserved: the same figures, computed more cheaply."""

    client = make_client(tmp_path)
    _process(client, tmp_path)
    body = client.get("/api/analytics/summary").json()

    evidence = body["evidence"]
    assert evidence["artifacts_total"] > 0
    assert evidence["artifact_bytes"] > 0
    assert evidence["events_with_artifacts"] == evidence["events_total"]

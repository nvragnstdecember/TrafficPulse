"""Analyst review lifecycle over the HTTP API (H9)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient

from trafficpulse.contracts.enums import ReviewStatus


def _reviewable_event(client: TestClient, tmp_path: Path) -> str:
    """Run a job and return the id of an event that can be reviewed."""

    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    items = client.get("/api/events").json()["items"]
    assert items, "the stub run must confirm at least one event to review"
    event_id: str = items[0]["event_id"]
    return event_id


def _decide(client: TestClient, event_id: str, action: str, **body: object):
    return client.post(
        f"/api/events/{event_id}/review", json={"action": action, **body}
    )


# --- reading -------------------------------------------------------------------
def test_an_unreviewed_event_reports_pending_with_no_history(tmp_path: Path) -> None:
    # Not a 404: "nobody has looked at this yet" is a real state, and the client
    # needs to render it without special-casing a missing resource.
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)

    body = client.get(f"/api/events/{event_id}/review").json()

    assert body["case"]["status"] == ReviewStatus.PENDING.value
    assert body["case"]["event_id"] == event_id
    assert body["history"] == []


def test_review_of_an_unknown_event_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/events/evt-nope/review")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "event_not_found"


# --- the lifecycle ---------------------------------------------------------------
def test_the_full_decision_path_is_recorded(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)

    assert _decide(client, event_id, "open", reviewer="analyst-a").status_code == 200
    _decide(client, event_id, "note", reviewer="analyst-a", note="Plate legible")
    final = _decide(
        client, event_id, "approve", reviewer="analyst-a", reason="Clear violation"
    ).json()

    assert final["case"]["status"] == ReviewStatus.APPROVED.value
    assert final["case"]["reviewer_id"] == "analyst-a"
    assert final["case"]["note"] == "Plate legible"  # survived the approval
    assert final["case"]["reason"] == "Clear violation"
    assert final["case"]["decided_at"] is not None
    assert [entry["action"] for entry in final["history"]] == ["open", "note", "approve"]


def test_an_illegal_transition_is_a_conflict_not_a_validation_error(tmp_path: Path) -> None:
    # 409, because the request is well-formed and it is the *state* that refuses it.
    # A client must be able to tell "you sent nonsense" from "somebody else already
    # decided this" -- the latter must never silently overwrite their call.
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)

    response = _decide(client, event_id, "approve")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "invalid_transition"
    # ...and nothing was written.
    assert client.get(f"/api/events/{event_id}/review").json()["history"] == []


def test_a_decided_case_cannot_be_re_decided_without_reopening(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)
    _decide(client, event_id, "open")
    _decide(client, event_id, "approve")

    assert _decide(client, event_id, "reject").status_code == 409

    assert _decide(client, event_id, "reopen").status_code == 200
    assert _decide(client, event_id, "reject").status_code == 200
    body = client.get(f"/api/events/{event_id}/review").json()
    assert body["case"]["status"] == ReviewStatus.REJECTED.value
    # The original approval is still in the record -- corrections append.
    assert [entry["action"] for entry in body["history"]] == [
        "open",
        "approve",
        "reopen",
        "reject",
    ]


def test_a_note_does_not_decide_a_case(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)
    _decide(client, event_id, "open")

    body = _decide(client, event_id, "note", note="Checking the plate").json()

    assert body["case"]["status"] == ReviewStatus.IN_REVIEW.value
    assert body["case"]["decided_at"] is None


def test_false_positive_is_recorded_distinctly_from_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)
    _decide(client, event_id, "open")

    body = _decide(
        client, event_id, "false_positive", reason="Shadow misread as a rider"
    ).json()

    assert body["case"]["status"] == ReviewStatus.FALSE_POSITIVE.value
    assert body["case"]["status"] != ReviewStatus.REJECTED.value


def test_an_unknown_action_is_rejected_by_validation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)

    response = _decide(client, event_id, "obliterate")

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_deciding_an_unknown_event_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert _decide(client, "evt-nope", "open").status_code == 404


# --- list integration ------------------------------------------------------------
def test_the_event_list_badges_each_row_with_its_review_status(tmp_path: Path) -> None:
    # So a dashboard can badge and filter without an N+1 fetch per row.
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)

    before = client.get("/api/events").json()["items"]
    assert {item["review_status"] for item in before} == {ReviewStatus.PENDING.value}

    _decide(client, event_id, "open")
    _decide(client, event_id, "approve")

    after = {
        item["event_id"]: item["review_status"]
        for item in client.get("/api/events").json()["items"]
    }
    assert after[event_id] == ReviewStatus.APPROVED.value


# --- persistence -----------------------------------------------------------------
def test_decisions_survive_a_new_application(tmp_path: Path) -> None:
    # The success criterion: a decision is on disk, not in process memory. A fresh
    # app over the same storage must see it.
    config = make_config(tmp_path)
    client = make_client(tmp_path, config=config)
    event_id = _reviewable_event(client, tmp_path)
    _decide(client, event_id, "open", reviewer="analyst-a")
    _decide(client, event_id, "approve", reviewer="analyst-a", note="Confirmed")

    # A brand-new app instance over the same storage root. Since H10 it rebuilds
    # its job/event index at startup, so the review is reachable immediately --
    # no re-upload and no reprocessing.
    fresh = make_client(tmp_path, config=make_config(tmp_path))

    body = fresh.get(f"/api/events/{event_id}/review").json()
    assert body["case"]["status"] == ReviewStatus.APPROVED.value
    assert body["case"]["note"] == "Confirmed"
    assert [entry["action"] for entry in body["history"]] == ["open", "approve"]


def test_the_journal_is_written_beside_the_events_never_into_them(tmp_path: Path) -> None:
    # The architectural invariant: review state must never be able to collide with
    # a write-once event record.
    client = make_client(tmp_path)
    event_id = _reviewable_event(client, tmp_path)
    _decide(client, event_id, "open")

    runs = make_config(tmp_path).runs_dir
    assert (runs / "reviews" / f"{event_id}.jsonl").is_file()
    # Every append-only sidecar is a *sibling* of the per-run tree, never inside it:
    # the review journal (H9) and the rendered-artifact journal (H14) both live at
    # the root, so neither can collide with a write-once run directory.
    sidecars = {"reviews", "rendered"}
    for run_dir in runs.iterdir():
        if run_dir.name in sidecars:
            continue
        assert not any(path.suffix == ".jsonl" for path in run_dir.rglob("*"))

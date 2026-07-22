"""Evidence endpoint: manifest references only, no media (H7A)."""

from __future__ import annotations

from pathlib import Path

from _app_helpers import make_client, upload_wrong_way_video


def test_evidence_returns_the_manifest_with_frame_references(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    video_id = upload_wrong_way_video(client, tmp_path)
    client.post("/api/process", json={"video_id": video_id})
    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]

    response = client.get(f"/api/evidence/{event_id}")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["event_id"] == event_id
    trigger = manifest["trigger_frame"]
    assert trigger is not None
    assert trigger["locator"].startswith("frames/")
    # References only: no rendered media, no integrity hash was computed.
    assert trigger["sha256"] is None
    assert manifest["rule_trace"]  # a reviewable rule trace is present


def test_evidence_unknown_event_is_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/evidence/evt-nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "event_not_found"

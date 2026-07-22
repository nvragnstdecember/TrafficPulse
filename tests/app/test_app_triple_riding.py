"""End-to-end triple-riding detection through the HTTP application (v1.1 U3).

Proves the integration hop: a triple-riding video, uploaded and processed by the
*app*, surfaces a ``triple_riding`` ConfirmedEvent and its evidence through the
same ``/api/events`` and ``/api/evidence`` endpoints the React workspace already
consumes -- with no frontend change (the workspace already renders the
``triple_riding`` violation type).

Everything under test is reused: the engine's triple-riding rule wiring, the U1
perception + P4-U4 association, the U3 reasoner, persistence, and evidence. Unlike
no-helmet, no classifier is needed (rider counting is pure geometry), so the app
path needs no extra injection. The scripted rider-count detector replays a
motorcycle carrying N riders; coverage of the reasoner/smoothing internals lives
in ``tests/rules`` and ``tests/pipeline`` and is not repeated here.
"""

from __future__ import annotations

from pathlib import Path

from _app_helpers import StubEngineProvider, make_client, make_config
from _helmet_fixtures import helmet_detector_config
from _triple_fixtures import scripted_rider_count_detector, write_triple_riding_clip

from trafficpulse.app.registry import JobStatus
from trafficpulse.engine import TripleRidingRuleConfig


def _triple_client(tmp_path: Path, *, riders: int):
    provider = StubEngineProvider(
        detector_factory=lambda: scripted_rider_count_detector(riders=riders),
        detector_config=helmet_detector_config(),  # maps motorbike/person -> classes
    )
    config = make_config(tmp_path, default_rules=(TripleRidingRuleConfig(),))
    return make_client(tmp_path, provider=provider, config=config)


def _upload_clip(client: object, tmp_path: Path, *, riders: int) -> str:
    clip = write_triple_riding_clip(tmp_path / "triple.mp4", riders=riders)
    response = client.post(  # type: ignore[attr-defined]
        "/api/video/upload", files={"file": ("triple.mp4", clip.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 201, response.text
    return response.json()["video_id"]


def _process(client: object, video_id: str) -> dict:
    created = client.post("/api/process", json={"video_id": video_id})  # type: ignore[attr-defined]
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    return client.get(f"/api/process/{job_id}").json()  # type: ignore[attr-defined]


def test_triple_riding_video_produces_a_triple_riding_event(tmp_path: Path) -> None:
    client = _triple_client(tmp_path, riders=3)
    video_id = _upload_clip(client, tmp_path, riders=3)

    status = _process(client, video_id)
    assert status["status"] == JobStatus.SUCCEEDED.value
    assert status["event_count"] == 1

    listing = client.get("/api/events", params={"video_id": video_id}).json()
    assert listing["total"] == 1
    summary = listing["items"][0]
    assert summary["violation_type"] == "triple_riding"
    # The event names the motorcycle and its rider tracks (4 tracks: bike + 3).
    assert len(summary["track_ids"]) == 4


def test_triple_riding_event_detail_and_evidence(tmp_path: Path) -> None:
    client = _triple_client(tmp_path, riders=3)
    video_id = _upload_clip(client, tmp_path, riders=3)
    _process(client, video_id)

    event_id = client.get("/api/events", params={"video_id": video_id}).json()["items"][0][
        "event_id"
    ]

    detail = client.get(f"/api/events/{event_id}").json()
    assert detail["violation_type"] == "triple_riding"
    assert detail["trigger_at"]
    assert detail["confidence"]
    max_riders = next(m for m in detail["measurements"] if m["name"] == "max_rider_count")
    assert max_riders["value"] == 3.0

    evidence = client.get(f"/api/evidence/{event_id}").json()
    assert evidence["event_id"] == event_id
    assert evidence["trigger_frame"] is not None
    assert evidence["evidence_package_id"] and evidence["created_at"]


def test_two_riders_produce_no_event(tmp_path: Path) -> None:
    client = _triple_client(tmp_path, riders=2)
    video_id = _upload_clip(client, tmp_path, riders=2)
    status = _process(client, video_id)
    assert status["status"] == JobStatus.SUCCEEDED.value
    assert status["event_count"] == 0
    assert client.get("/api/events", params={"video_id": video_id}).json()["total"] == 0

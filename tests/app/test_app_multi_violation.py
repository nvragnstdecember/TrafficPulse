"""Multiple violation types in one run, end to end (R2 + R5).

The final-validation claim, exercised through the real HTTP surface: one upload,
one job, several shipped rules, several violation types confirmed, persisted, and
reported -- with the counts staying truthful when a client declares the same rule
twice.

Everything runs on a scripted stub detector over a real decoded clip: no torch, no
checkpoint. The scene is the illegal-stopping test scene (the example scene with
its no-stopping zone patched into the tiny clip's pixel space), which supports
wrong-way *and* illegal-stopping, so one track can genuinely commit both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _app_helpers import StubEngineProvider, make_client, make_config
from _stopping_fixtures import (
    illegal_stopping_test_scene,
    stopping_detector_config,
    write_illegal_stopping_clip,
)
from fastapi.testclient import TestClient

from trafficpulse.detector import RawDetection, StubDetector

# One vehicle: descends for 2.0 s (wrong-way against the scene's "north" legal
# direction), then holds inside the no-stopping zone for the rest of the clip
# (illegal stopping). One track, two violation types.
FRAMES = 70
_DESCENT_FRAMES = 20


def _box(frame_index: int) -> tuple[float, float, float, float]:
    bottom = 40.0 + frame_index * 8.0 if frame_index < _DESCENT_FRAMES else 200.0
    return (140.0, bottom - 30.0, 180.0, bottom)


def _detector() -> StubDetector:
    return StubDetector(
        per_frame={
            i: (RawDetection(label="car", score=0.9, box=_box(i)),) for i in range(FRAMES)
        }
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """An app whose resolved scene supports wrong-way and illegal-stopping."""

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        illegal_stopping_test_scene().model_dump_json(), encoding="utf-8"
    )
    # No ``default_rules``: the point of R5 is that the server derives them.
    config = make_config(tmp_path / "storage", scene_path=scene_path, default_rules=())
    return make_client(
        tmp_path / "storage",
        config=config,
        provider=StubEngineProvider(_detector, detector_config=stopping_detector_config()),
    )


def _upload(client: TestClient, tmp_path: Path) -> str:
    clip = write_illegal_stopping_clip(tmp_path / "multi.mp4", frames=FRAMES)
    response = client.post(
        "/api/video/upload",
        files={"file": ("multi.mp4", clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    video_id: str = response.json()["video_id"]
    return video_id


def _process(client: TestClient, video_id: str, rules: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"video_id": video_id}
    if rules is not None:
        body["rules"] = rules
    response = client.post("/api/process", json=body)
    assert response.status_code == 202, response.text
    status: dict[str, Any] = client.get(
        f"/api/process/{response.json()['job_id']}"
    ).json()
    assert status["status"] == "succeeded", status
    return status


def _types(client: TestClient, **params: Any) -> set[str]:
    listing = client.get("/api/events", params=params)
    assert listing.status_code == 200, listing.text
    return {item["violation_type"] for item in listing.json()["items"]}


# --- R5: derived rules produce a genuinely multi-violation run ----------------------
def test_a_default_run_derives_its_rules_and_confirms_several_types(
    client: TestClient, tmp_path: Path
) -> None:
    """One request naming no rules -> several violation types in one run."""

    video_id = _upload(client, tmp_path)
    supported = client.get(f"/api/videos/{video_id}").json()["supported_violations"]
    assert {"wrong_way", "illegal_stopping"} <= set(supported)

    status = _process(client, video_id)

    events = client.get("/api/events", params={"video_id": video_id}).json()
    types = {item["violation_type"] for item in events["items"]}
    assert {"wrong_way", "illegal_stopping"} <= types, events
    assert status["event_count"] == events["total"]
    # Distinct events, one manifest each -- nothing collapsed across types.
    assert len({item["event_id"] for item in events["items"]}) == events["total"]
    for item in events["items"]:
        evidence = client.get(f"/api/evidence/{item['event_id']}")
        assert evidence.status_code == 200, evidence.text
        assert evidence.json()["event_id"] == item["event_id"]


def test_a_default_run_never_selects_an_unsupported_or_unshipped_rule(
    client: TestClient, tmp_path: Path
) -> None:
    """Scene-aware, not "run everything": speeding and red-light stay out.

    ``speeding`` has no shipped reasoner, and red-light needs a per-run signal
    schedule no default can supply -- so neither can appear in a derived run even
    though this scene declares junction geometry.
    """

    video_id = _upload(client, tmp_path)
    _process(client, video_id)

    events = client.get("/api/events", params={"video_id": video_id}).json()
    types = {item["violation_type"] for item in events["items"]}
    assert "speeding" not in types
    assert "red_light_jumping" not in types


def test_explicitly_requested_rules_are_still_honoured_verbatim(
    client: TestClient, tmp_path: Path
) -> None:
    """R5 must not swallow an explicit request: one rule asked for, one rule run."""

    video_id = _upload(client, tmp_path)
    _process(client, video_id, rules=[{"kind": "illegal_stopping"}])

    events = client.get("/api/events", params={"video_id": video_id}).json()
    assert {item["violation_type"] for item in events["items"]} == {"illegal_stopping"}


def test_an_explicit_rule_the_scene_cannot_satisfy_still_fails_fast(
    client: TestClient, tmp_path: Path
) -> None:
    """Fail-fast is preserved for what the client actually asked for."""

    video_id = _upload(client, tmp_path)
    response = client.post(
        "/api/process",
        json={"video_id": video_id, "rules": [{"kind": "red_light_jumping"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_configuration"


# --- R2: duplicate rule declarations must not inflate anything ----------------------
def test_duplicate_rule_declarations_do_not_inflate_counts(
    client: TestClient, tmp_path: Path
) -> None:
    """The same rule twice is the same violation twice -- counted once.

    ``event_id`` is content-derived, so a duplicated rule re-confirms an identical
    event. The write-once store already absorbed it; this pins the *counts* that
    previously did not.
    """

    video_id = _upload(client, tmp_path)
    duplicated = _process(
        client,
        video_id,
        rules=[
            {"kind": "illegal_stopping"},
            {"kind": "illegal_stopping"},
            {"kind": "wrong_way", "direction_id": "dir-north"},
            {"kind": "wrong_way", "direction_id": "dir-north"},
        ],
    )

    events = client.get("/api/events", params={"video_id": video_id}).json()
    assert {item["violation_type"] for item in events["items"]} == {
        "illegal_stopping",
        "wrong_way",
    }
    # One event per violation, not two.
    assert events["total"] == 2
    assert duplicated["event_count"] == 2

    violations = client.get("/api/analytics/summary").json()["violations"]
    assert {v["violation_type"]: v["count"] for v in violations["by_type"]} == {
        "illegal_stopping": 1,
        "wrong_way": 1,
    }
    assert violations["events_total"] == 2
    assert violations["events_total"] == sum(v["count"] for v in violations["by_type"])


def test_duplicate_declarations_persist_exactly_one_record_per_event(
    client: TestClient, tmp_path: Path
) -> None:
    """Persistence stays correct: one event file + one manifest file per event."""

    video_id = _upload(client, tmp_path)
    response = client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [{"kind": "illegal_stopping"}, {"kind": "illegal_stopping"}],
        },
    )
    job_id = response.json()["job_id"]

    run_dir = tmp_path / "storage" / "runs" / job_id
    events_on_disk = sorted(p.name for p in (run_dir / "events").glob("*.json"))
    manifests_on_disk = sorted(p.name for p in (run_dir / "manifests").glob("*.json"))
    assert len(events_on_disk) == 1
    assert events_on_disk == manifests_on_disk

    stored = json.loads((run_dir / "events" / events_on_disk[0]).read_text(encoding="utf-8"))
    assert stored["violation_type"] == "illegal_stopping"


def test_distinct_events_are_never_folded_together(
    client: TestClient, tmp_path: Path
) -> None:
    """Deduplication is by ``event_id`` only -- two real violations stay two."""

    video_id = _upload(client, tmp_path)
    _process(
        client,
        video_id,
        rules=[
            {"kind": "illegal_stopping"},
            {"kind": "illegal_stopping"},
            {"kind": "wrong_way", "direction_id": "dir-north"},
        ],
    )

    events = client.get("/api/events", params={"video_id": video_id}).json()
    assert events["total"] == 2
    assert len({item["event_id"] for item in events["items"]}) == 2


# --- run scoping over a real reprocess (R7) -----------------------------------------
def test_a_reprocess_can_be_reviewed_without_the_superseded_runs_events(
    client: TestClient, tmp_path: Path
) -> None:
    """The workflow R7 exists for, end to end.

    A video is processed for both violations, then reprocessed for illegal-stopping
    alone -- the narrowing an analyst does after deciding the wrong-way geometry was
    wrong. Scoped to the second run the wrong-way event is gone; the repository view
    still holds it, because it genuinely happened.
    """

    video_id = _upload(client, tmp_path)
    first = _process(
        client,
        video_id,
        rules=[
            {"kind": "wrong_way", "direction_id": "dir-north"},
            {"kind": "illegal_stopping"},
        ],
    )
    second = _process(client, video_id, rules=[{"kind": "illegal_stopping"}])
    assert first["job_id"] != second["job_id"]

    assert _types(client, video_id=video_id, job_id=first["job_id"]) == {
        "wrong_way",
        "illegal_stopping",
    }
    assert _types(client, video_id=video_id, job_id=second["job_id"]) == {
        "illegal_stopping"
    }
    # The history is untouched: both runs' findings remain addressable.
    assert _types(client, video_id=video_id) == {"wrong_way", "illegal_stopping"}


def test_a_listed_events_run_is_the_run_its_evidence_is_served_from(
    client: TestClient, tmp_path: Path
) -> None:
    """``job_id`` on a summary must address something -- across a reprocess too."""

    video_id = _upload(client, tmp_path)
    _process(client, video_id, rules=[{"kind": "illegal_stopping"}])
    latest = _process(client, video_id, rules=[{"kind": "illegal_stopping"}])

    for item in client.get("/api/events", params={"video_id": video_id}).json()["items"]:
        # Re-listing scoped to the run the summary names must return that same event.
        scoped = client.get(
            "/api/events", params={"video_id": video_id, "job_id": item["job_id"]}
        ).json()
        assert item["event_id"] in {row["event_id"] for row in scoped["items"]}
        assert item["job_id"] == latest["job_id"]
        assert client.get(f"/api/evidence/{item['event_id']}").status_code == 200

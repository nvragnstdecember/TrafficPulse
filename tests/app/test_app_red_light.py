"""Red-light jumping through the HTTP application (H13).

The milestone's success path, executed: calibrate a junction, declare the run's
signal timing, process, and get a confirmed event that survives a restart and is
reviewable — with no configuration file and no reasoner change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _app_helpers import make_client, make_config, upload_wrong_way_video
from _slice_fixtures import HEIGHT, WIDTH
from fastapi.testclient import TestClient

from trafficpulse.contracts.scene import ZoneType

# Mirrors tests/pipeline/test_red_light_pipeline.py: the synthetic clip's rectangle
# crosses y=120 at t=1.2 s and reaches the junction (y>=150) at t=1.7 s.
STOP_LINE_Y = 120.0
JUNCTION_TOP = 150.0
LANE = "zone-approach"
JUNCTION = "zone-junction"


def _full_frame() -> list[list[float]]:
    return [[0.0, 0.0], [float(WIDTH), 0.0], [float(WIDTH), float(HEIGHT)], [0.0, float(HEIGHT)]]


def _junction_draft(**overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "scene_name": "Signalised junction",
        "camera_id": "cam-junction",
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "zones": [
            {
                "zone_id": LANE,
                "zone_type": ZoneType.LANE.value,
                "polygon": _full_frame(),
            },
            {
                "zone_id": JUNCTION,
                "zone_type": ZoneType.INTERSECTION.value,
                "polygon": [
                    [100.0, JUNCTION_TOP],
                    [220.0, JUNCTION_TOP],
                    [220.0, 235.0],
                    [100.0, 235.0],
                ],
            },
        ],
        "stop_lines": [
            {
                "stop_line_id": "sl-1",
                "a": [100.0, STOP_LINE_Y],
                "b": [220.0, STOP_LINE_Y],
                "crossing_dx": 0.0,
                "crossing_dy": 1.0,
                "signal_group_id": "sg-1",
                "zone_ids": [JUNCTION],
            }
        ],
        "signal_groups": [
            {
                "signal_group_id": "sg-1",
                "roi_polygon": [[5.0, 5.0], [45.0, 5.0], [45.0, 60.0]],
                "zone_ids": [JUNCTION],
            }
        ],
    }
    draft.update(overrides)
    return draft


def _red_light_rule(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "red_light_jumping", "schedule": schedule}


def _calibrated(client: TestClient, tmp_path: Path) -> str:
    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.put(f"/api/videos/{video_id}/scene", json=_junction_draft())
    assert response.status_code == 200, response.text
    return video_id


# --- the success criterion ---------------------------------------------------------
def test_calibrate_configure_timing_process_and_confirm(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(client, tmp_path)

    job = client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [_red_light_rule([{"at_seconds": 0.0, "state": "red"}])],
        },
    )
    assert job.status_code == 202, job.text
    job_id = job.json()["job_id"]
    assert client.get(f"/api/process/{job_id}").json()["status"] == "succeeded"

    events = client.get("/api/events", params={"video_id": video_id}).json()["items"]
    assert [e["violation_type"] for e in events] == ["red_light_jumping"]

    # Evidence comes from the existing pipeline, unchanged.
    manifest = client.get(f"/api/evidence/{events[0]['event_id']}")
    assert manifest.status_code == 200
    assert manifest.json()["trigger_frame"] is not None

    # And the analyst review workflow accepts it like any other event.
    decided = client.post(
        f"/api/events/{events[0]['event_id']}/review", json={"action": "open"}
    )
    assert decided.status_code == 200
    assert decided.json()["case"]["status"] == "in_review"


def test_a_green_schedule_confirms_nothing(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(client, tmp_path)

    job = client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [_red_light_rule([{"at_seconds": 0.0, "state": "green"}])],
        },
    )

    assert job.status_code == 202, job.text
    assert client.get("/api/events", params={"video_id": video_id}).json()["total"] == 0


def test_a_light_change_between_the_line_and_the_junction_still_confirms(
    tmp_path: Path,
) -> None:
    # The milestone's hard requirement, through the full HTTP stack.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(client, tmp_path)

    client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [
                _red_light_rule(
                    [
                        {"at_seconds": 0.0, "state": "red"},
                        {"at_seconds": 1.5, "state": "green"},
                    ]
                )
            ],
        },
    )

    assert client.get("/api/events", params={"video_id": video_id}).json()["total"] == 1


# --- capability discovery ------------------------------------------------------------
def test_drawing_a_junction_advertises_red_light_on_the_video(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    before = client.get(f"/api/videos/{video_id}").json()["supported_violations"]
    summary = client.put(f"/api/videos/{video_id}/scene", json=_junction_draft()).json()
    after = client.get(f"/api/videos/{video_id}").json()["supported_violations"]

    assert "red_light_jumping" not in before
    assert "red_light_jumping" in summary["supported_violations"]
    assert "red_light_jumping" in after


def test_a_scene_without_a_stop_line_does_not_advertise_red_light(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    no_junction = _junction_draft(
        zones=[
            {
                "zone_id": LANE,
                "zone_type": ZoneType.LANE.value,
                "polygon": _full_frame(),
            }
        ],
        stop_lines=[],
        signal_groups=[],
    )

    summary = client.put(f"/api/videos/{video_id}/scene", json=no_junction).json()

    assert "red_light_jumping" not in summary["supported_violations"]


# --- configuration errors ---------------------------------------------------------------
def test_a_red_light_rule_without_a_schedule_is_a_clean_400(tmp_path: Path) -> None:
    # Silently confirming nothing is the failure mode this refusal exists to prevent:
    # with no phases every instant resolves to 'unknown'.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(client, tmp_path)

    response = client.post(
        "/api/process",
        json={"video_id": video_id, "rules": [{"kind": "red_light_jumping"}]},
    )

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["type"] == "invalid_configuration"
    assert "schedule" in body["message"]


def test_a_red_light_rule_on_an_uncalibrated_junction_is_a_clean_400(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    # A lane-only scene: no stop line, so red-light cannot be built for it.
    client.put(
        f"/api/videos/{video_id}/scene",
        json=_junction_draft(
            zones=[
                {
                    "zone_id": LANE,
                    "zone_type": ZoneType.LANE.value,
                    "polygon": [
                        [0.0, 0.0],
                        [float(WIDTH), 0.0],
                        [float(WIDTH), float(HEIGHT)],
                        [0.0, float(HEIGHT)],
                    ],
                }
            ],
            stop_lines=[],
            signal_groups=[],
        ),
    )

    response = client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [_red_light_rule([{"at_seconds": 0.0, "state": "red"}])],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_configuration"


def test_an_out_of_order_schedule_is_rejected_by_validation(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(client, tmp_path)

    response = client.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [
                _red_light_rule(
                    [
                        {"at_seconds": 5.0, "state": "red"},
                        {"at_seconds": 1.0, "state": "green"},
                    ]
                )
            ],
        },
    )

    assert response.status_code in (400, 422)


# --- persistence + recovery ------------------------------------------------------------
def test_a_red_light_event_and_its_scene_survive_a_restart(tmp_path: Path) -> None:
    first = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(first, tmp_path)
    first.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [_red_light_rule([{"at_seconds": 0.0, "state": "red"}])],
        },
    )
    original = first.get("/api/events").json()["items"]
    assert original

    restarted = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))

    recovered = restarted.get("/api/events", params={"video_id": video_id}).json()["items"]
    assert [e["event_id"] for e in recovered] == [e["event_id"] for e in original]
    # The scene the event names is still fetchable -- provenance, not a claim. The
    # hash lives on the full ConfirmedEvent, not on the compact list summary.
    detail = restarted.get(f"/api/events/{recovered[0]['event_id']}").json()
    scene = restarted.get(f"/api/scenes/{detail['scene_config_hash']}")
    assert scene.status_code == 200
    assert scene.json()["stop_lines"][0]["stop_line_id"] == "sl-1"
    assert "red_light_jumping" in restarted.get(
        f"/api/videos/{video_id}"
    ).json()["supported_violations"]


def test_review_continues_for_a_red_light_event_after_a_restart(tmp_path: Path) -> None:
    first = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = _calibrated(first, tmp_path)
    first.post(
        "/api/process",
        json={
            "video_id": video_id,
            "rules": [_red_light_rule([{"at_seconds": 0.0, "state": "red"}])],
        },
    )
    event_id = first.get("/api/events").json()["items"][0]["event_id"]
    first.post(f"/api/events/{event_id}/review", json={"action": "open"})

    restarted = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    decided = restarted.post(
        f"/api/events/{event_id}/review",
        json={"action": "approve", "note": "Clear red-light entry"},
    )

    assert decided.status_code == 200
    assert decided.json()["case"]["status"] == "approved"

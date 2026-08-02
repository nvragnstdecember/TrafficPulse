"""Per-video scene management and calibration (H12).

The milestone's claim, executed: a scene belongs to a *video*, survives a
restart, and is what enables the geometry-dependent rules -- with no configuration
file and no change to any reasoner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _app_helpers import make_client, make_config, upload_wrong_way_video
from fastapi.testclient import TestClient

from trafficpulse.contracts.scene import ZoneType

# The synthetic clip is 320x240 and its vehicle travels *down* the frame, so a
# scene whose legal direction is "up" makes it a sustained wrong-way run.
WIDTH, HEIGHT = 320, 240
LANE = "zone-lane"
NO_STOPPING = "zone-nostop"


def _full_frame() -> list[list[float]]:
    return [[0.0, 0.0], [float(WIDTH), 0.0], [float(WIDTH), float(HEIGHT)], [0.0, float(HEIGHT)]]


def _draft(**overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "scene_name": "Test junction",
        "camera_id": "cam-test",
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "zones": [
            {"zone_id": LANE, "zone_type": ZoneType.LANE.value, "polygon": _full_frame()}
        ],
        "direction": {"dx": 0.0, "dy": -1.0, "zone_id": LANE},
    }
    draft.update(overrides)
    return draft


def _with_no_stopping_zone() -> dict[str, Any]:
    return _draft(
        zones=[
            {"zone_id": LANE, "zone_type": ZoneType.LANE.value, "polygon": _full_frame()},
            {
                "zone_id": NO_STOPPING,
                "zone_type": ZoneType.NO_STOPPING.value,
                "polygon": [[10.0, 10.0], [300.0, 10.0], [300.0, 220.0], [10.0, 220.0]],
            },
        ]
    )


def _calibrate(client: TestClient, video_id: str, draft: dict[str, Any]) -> dict[str, Any]:
    response = client.put(f"/api/videos/{video_id}/scene", json=draft)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- the success criterion ---------------------------------------------------------
def test_calibrate_restart_recover_and_wrong_way_confirms(tmp_path: Path) -> None:
    # Upload -> calibrate -> persist -> restart -> recover -> process -> wrong way.
    # No configuration file names this scene and no reasoner changed; the only new
    # thing is that the video carries its own geometry.
    first = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(first, tmp_path)
    summary = _calibrate(first, video_id, _draft())
    assert "wrong_way" in summary["supported_violations"]

    restarted = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))

    # The binding survived, and so did the revision it names.
    recovered = restarted.get(f"/api/videos/{video_id}/scene").json()
    assert recovered["scene_hash"] == summary["scene_hash"]

    job = restarted.post(
        "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
    )
    assert job.status_code == 202, job.text
    job_id = job.json()["job_id"]
    assert restarted.get(f"/api/process/{job_id}").json()["status"] == "succeeded"

    events = restarted.get("/api/events", params={"video_id": video_id}).json()["items"]
    assert [event["violation_type"] for event in events] == ["wrong_way"]


def test_illegal_stopping_is_enabled_by_drawing_a_no_stopping_zone(tmp_path: Path) -> None:
    # The rule that a server-wide scene could never enable for an arbitrary upload:
    # where stopping is illegal is a fact about the site, so it has to be drawn.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    before = client.get(f"/api/videos/{video_id}").json()
    assert before["supported_violations"] == []

    summary = _calibrate(client, video_id, _with_no_stopping_zone())

    assert summary["has_no_stopping_zone"] is True
    assert "illegal_stopping" in summary["supported_violations"]
    # And the engine accepts it: the rule builds against this scene and runs.
    job = client.post(
        "/api/process",
        json={"video_id": video_id, "rules": [{"kind": "illegal_stopping"}]},
    )
    assert job.status_code == 202, job.text
    assert client.get(f"/api/process/{job.json()['job_id']}").json()["status"] == "succeeded"


def test_two_videos_are_reasoned_over_their_own_geometry(tmp_path: Path) -> None:
    # The singleton this milestone removed: one process, two cameras, two scenes.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    from _slice_fixtures import FRAME_COUNT, write_wrong_way_clip

    first = upload_wrong_way_video(client, tmp_path, name="a.mp4")
    clip = write_wrong_way_clip(tmp_path / "b.mp4", frames=FRAME_COUNT - 5)
    second = client.post(
        "/api/video/upload", files={"file": ("b.mp4", clip.read_bytes(), "video/mp4")}
    ).json()["video_id"]

    # Opposite legal directions: the same downward motion is a violation under one
    # and perfectly legal under the other.
    against = _calibrate(client, first, _draft(direction={"dx": 0.0, "dy": -1.0, "zone_id": LANE}))
    with_flow = _calibrate(
        client, second, _draft(direction={"dx": 0.0, "dy": 1.0, "zone_id": LANE})
    )
    assert against["scene_hash"] != with_flow["scene_hash"]

    for video_id in (first, second):
        response = client.post(
            "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
        )
        assert response.status_code == 202, response.text

    opposing = client.get("/api/events", params={"video_id": first}).json()
    aligned = client.get("/api/events", params={"video_id": second}).json()
    assert opposing["total"] >= 1
    assert aligned["total"] == 0  # travelling with the declared flow is not a violation


# --- scene as a resource -----------------------------------------------------------
def test_a_stored_scene_resolves_by_the_hash_events_carry(tmp_path: Path) -> None:
    # Provenance stops being a claim: the hash stamped into a ConfirmedEvent is an
    # address, and fetching it returns the exact geometry that produced the event.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    _calibrate(client, video_id, _draft())
    client.post("/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]})

    event_id = client.get("/api/events").json()["items"][0]["event_id"]
    event = client.get(f"/api/events/{event_id}").json()

    scene = client.get(f"/api/scenes/{event['scene_config_hash']}")
    assert scene.status_code == 200, scene.text
    body = scene.json()
    assert body["frame"]["reference_width"] == WIDTH
    assert body["legal_directions"][0]["vector"]["dy"] == -1.0


def test_recalibrating_supersedes_the_binding_but_not_the_old_revision(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    original = _calibrate(client, video_id, _draft())

    edited = _calibrate(client, video_id, _draft(direction={"dx": 1.0, "dy": 0.0, "zone_id": LANE}))

    assert edited["scene_hash"] != original["scene_hash"]
    assert client.get(f"/api/videos/{video_id}/scene").json()["scene_hash"] == edited["scene_hash"]
    # The superseded revision is still fetchable -- events reasoned under it must
    # not be left pointing at content that no longer exists.
    assert client.get(f"/api/scenes/{original['scene_hash']}").status_code == 200


def test_saving_an_unchanged_drawing_changes_nothing(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    first = _calibrate(client, video_id, _draft())
    second = _calibrate(client, video_id, _draft())

    assert first == second
    assert len(list(make_config(tmp_path).scenes_dir.glob("*.json"))) == 1


def test_an_uncalibrated_video_reports_no_scene(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    response = client.get(f"/api/videos/{video_id}/scene")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "scene_not_found"
    assert client.get(f"/api/videos/{video_id}").json()["scene_hash"] is None


def test_unknown_scene_and_unknown_video_are_typed_404s(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    assert client.get(f"/api/scenes/{'0' * 64}").status_code == 404
    assert client.get("/api/videos/vid-nope/scene").status_code == 404
    assert client.put("/api/videos/vid-nope/scene", json=_draft()).status_code == 404


# --- validation --------------------------------------------------------------------
def test_a_draft_can_be_checked_without_being_saved(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    response = client.post(f"/api/videos/{video_id}/scene/validate", json=_draft())

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert "wrong_way" in body["supported_violations"]
    assert body["scene_hash"]
    # Nothing was stored, and the video is still uncalibrated.
    assert not make_config(tmp_path).scenes_dir.exists()
    assert client.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_validating_predicts_the_hash_the_save_would_produce(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)

    predicted = client.post(f"/api/videos/{video_id}/scene/validate", json=_draft()).json()
    saved = _calibrate(client, video_id, _draft())

    assert predicted["scene_hash"] == saved["scene_hash"]


def test_geometry_drawn_off_frame_is_reported_as_a_state_not_a_request_error(
    tmp_path: Path,
) -> None:
    # A half-finished or mis-drawn calibration is something the UI renders while the
    # analyst is still working, so validate answers 200 with the reason.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    off_frame = _draft(
        zones=[
            {
                "zone_id": LANE,
                "zone_type": ZoneType.LANE.value,
                "polygon": [[0.0, 0.0], [9999.0, 0.0], [10.0, 10.0]],
            }
        ],
        direction=None,
    )

    response = client.post(f"/api/videos/{video_id}/scene/validate", json=off_frame)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]
    assert body["scene_hash"] is None


def test_a_drawing_against_the_wrong_frame_size_is_refused(tmp_path: Path) -> None:
    # Geometry drawn on a differently-sized frame lands in the wrong place; only
    # the cases that overflow would be caught by the bounds check alone.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    mismatched = _draft(frame_width=1920, frame_height=1080)

    saved = client.put(f"/api/videos/{video_id}/scene", json=mismatched)
    checked = client.post(f"/api/videos/{video_id}/scene/validate", json=mismatched)

    assert saved.status_code == 400
    assert saved.json()["error"]["type"] == "invalid_configuration"
    assert checked.json()["valid"] is False
    assert "does not match" in checked.json()["errors"][0]


# --- the fallback --------------------------------------------------------------------
def test_an_uncalibrated_video_falls_back_to_the_servers_configured_scene(
    tmp_path: Path,
) -> None:
    # A single-camera deployment that configured one scene keeps working exactly as
    # it did before H12; the global scene is now a default, not the only answer.
    client = make_client(tmp_path)  # make_config supplies the example scene

    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.post("/api/process", json={"video_id": video_id})

    assert response.status_code == 202
    assert client.get(f"/api/videos/{video_id}").json()["scene_hash"] is None


def test_a_videos_own_scene_wins_over_the_configured_one(tmp_path: Path) -> None:
    client = make_client(tmp_path)  # the example scene is configured as fallback
    video_id = upload_wrong_way_video(client, tmp_path)
    _calibrate(client, video_id, _draft())

    # The example scene declares two legal directions, so a bare wrong_way rule
    # cannot resolve one and 400s. The calibrated scene declares exactly one, so
    # the same request now succeeds -- which is only possible if the binding won.
    response = client.post(
        "/api/process", json={"video_id": video_id, "rules": [{"kind": "wrong_way"}]}
    )

    assert response.status_code == 202, response.text


# --- recovery ------------------------------------------------------------------------
def test_a_binding_whose_scene_was_deleted_degrades_to_uncalibrated(tmp_path: Path) -> None:
    # Keeping a dangling binding would send processing looking for geometry that is
    # gone; dropping it is a state the analyst can see and fix.
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    video_id = upload_wrong_way_video(client, tmp_path)
    summary = _calibrate(client, video_id, _draft())
    (make_config(tmp_path).scenes_dir / f"{summary['scene_hash']}.json").unlink()

    restarted = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))

    assert restarted.get(f"/api/videos/{video_id}").json()["scene_hash"] is None
    assert restarted.get(f"/api/videos/{video_id}/scene").status_code == 404


def test_the_library_reports_calibration_state_per_video(tmp_path: Path) -> None:
    client = make_client(tmp_path, config=make_config(tmp_path, scene_path=None))
    from _slice_fixtures import FRAME_COUNT, write_wrong_way_clip

    calibrated = upload_wrong_way_video(client, tmp_path, name="a.mp4")
    clip = write_wrong_way_clip(tmp_path / "b.mp4", frames=FRAME_COUNT - 5)
    bare = client.post(
        "/api/video/upload", files={"file": ("b.mp4", clip.read_bytes(), "video/mp4")}
    ).json()["video_id"]
    _calibrate(client, calibrated, _with_no_stopping_zone())

    rows = {row["video_id"]: row for row in client.get("/api/videos").json()["items"]}

    assert rows[calibrated]["scene_hash"] is not None
    # Triple riding needs no geometry, so an authored scene always supports it;
    # no_helmet is absent because this deployment configures no helmet classifier,
    # which is a deployment fact the scene cannot override.
    assert set(rows[calibrated]["supported_violations"]) == {
        "wrong_way",
        "illegal_stopping",
        "triple_riding",
    }
    assert rows[bare]["scene_hash"] is None
    assert rows[bare]["supported_violations"] == []

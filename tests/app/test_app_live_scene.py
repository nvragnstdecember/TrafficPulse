"""What a live camera is reasoned about, and what that entitles it to run.

The subject here is the seam between live mode and the existing calibration
machinery. Two claims are worth testing separately from the session mechanics:

* an **uncalibrated** camera gets a scene that claims only its frame size, so the
  geometry-dependent violations are unavailable -- and are reported as unavailable,
  with a reason, rather than silently absent from an empty event feed;
* a **calibrated** camera reasons through the analyst's own stored scene, which is
  the only way live mode ever runs wrong-way or illegal-stopping. Nothing is
  derived from a camera view to get there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _live_helpers import HEIGHT, WIDTH, LiveStubProvider, frame_message, live_client, live_config

from trafficpulse.app.errors import SceneNotFoundError
from trafficpulse.app.live import (
    LiveConfig,
    LiveProtocolError,
    LiveSessionManager,
    live_rule_plan,
    provisional_live_scene,
)
from trafficpulse.app.registry import VideoStore
from trafficpulse.app.services import SceneService, VideoService
from trafficpulse.contracts import SceneConfig, ZoneType
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.persistence import SceneStore
from trafficpulse.scenes import (
    CALIBRATION_SOURCE_ANALYST,
    DirectionDraft,
    SceneDraft,
    ZoneDraft,
    build_scene,
    full_frame_polygon,
)


def _calibrated_scene() -> SceneConfig:
    """A scene an analyst could have drawn for this camera: a lane and its direction.

    Deliberately authored at the camera's own frame size. Wrong-way reasoning is
    scoped to its governing lane's polygon, so a scene measured against a different
    frame would put that polygon somewhere else in this one and the rule would run
    and confirm nothing -- which is exactly the silent failure the size guard exists
    to prevent.
    """

    draft = SceneDraft(
        scene_name="live-calibrated",
        camera_id="cam-fixed-1",
        frame_width=WIDTH,
        frame_height=HEIGHT,
        zones=(
            ZoneDraft(
                zone_id="zone-lane",
                zone_type=ZoneType.LANE,
                polygon=full_frame_polygon(WIDTH, HEIGHT),
            ),
        ),
        direction=DirectionDraft(dx=0.0, dy=-1.0, zone_id="zone-lane"),
    )
    return build_scene(
        draft, scene_id="scene-fixed-1", calibration_source=CALIBRATION_SOURCE_ANALYST
    )


def _manager(tmp_path: Path) -> tuple[LiveSessionManager, SceneStore]:
    config = live_config(tmp_path)
    videos = VideoStore()
    store = SceneStore(tmp_path / "scenes")
    scenes = SceneService(store, VideoService(config, videos), videos)
    manager = LiveSessionManager(
        config=config,
        live_config=LiveConfig(),
        provider=LiveStubProvider(),  # type: ignore[arg-type]
        scenes=scenes,
    )
    return manager, store


# --- the provisional scene ---------------------------------------------------------
def test_a_provisional_live_scene_claims_only_the_frame() -> None:
    scene = provisional_live_scene(
        width=WIDTH, height=HEIGHT, camera_id="cam-live", scene_id="scene-live"
    )
    assert (scene.frame.reference_width, scene.frame.reference_height) == (WIDTH, HEIGHT)
    # Nothing observable-from-nothing is invented: no direction, no stop line, no
    # signal timing, and exactly one zone covering the view.
    assert scene.legal_directions == ()
    assert scene.stop_lines == ()
    assert len(scene.zones) == 1
    assert scene.zones[0].zone_type is ZoneType.LANE


def test_the_plan_for_a_provisional_scene_names_every_missing_prerequisite() -> None:
    scene = provisional_live_scene(
        width=WIDTH, height=HEIGHT, camera_id="cam-live", scene_id="scene-live"
    )
    plan = live_rule_plan(scene, no_helmet_available=True)
    unavailable = {entry.violation_type: entry.reason for entry in plan.unavailable}

    assert ViolationType.TRIPLE_RIDING in plan.supported
    assert ViolationType.NO_HELMET in plan.supported
    assert set(unavailable) == {
        ViolationType.WRONG_WAY,
        ViolationType.ILLEGAL_STOPPING,
        ViolationType.RED_LIGHT_JUMPING,
    }
    # Each reason names the missing evidence, so a viewer can act on it.
    assert "legal travel direction" in unavailable[ViolationType.WRONG_WAY]
    assert "no-stopping zone" in unavailable[ViolationType.ILLEGAL_STOPPING]
    assert "signal timing" in unavailable[ViolationType.RED_LIGHT_JUMPING]


def test_a_turban_blind_deployment_cannot_run_the_helmet_rule_live() -> None:
    """The capability guard governs live mode exactly as it governs a job."""

    scene = provisional_live_scene(
        width=WIDTH, height=HEIGHT, camera_id="cam-live", scene_id="scene-live"
    )
    plan = live_rule_plan(scene, no_helmet_available=False)
    assert ViolationType.NO_HELMET not in plan.supported
    reasons = {entry.violation_type: entry.reason for entry in plan.unavailable}
    assert "still classified and displayed" in reasons[ViolationType.NO_HELMET]


# --- the calibrated scene -----------------------------------------------------------
def test_a_calibrated_camera_runs_the_geometry_rules_live(tmp_path: Path) -> None:
    manager, store = _manager(tmp_path)
    scene_hash = store.put(_calibrated_scene())
    try:
        session = manager.create(width=WIDTH, height=HEIGHT, scene_hash=scene_hash)
        assert ViolationType.WRONG_WAY in session.plan.supported
        assert session.camera_id == "cam-fixed-1"
        # And it is the analyst's scene, not a live-mode reconstruction of one.
        assert session.scene.calibration.source == CALIBRATION_SOURCE_ANALYST
    finally:
        manager.close_all()


def test_a_calibrated_session_reports_itself_as_calibrated(tmp_path: Path) -> None:
    manager, store = _manager(tmp_path)
    scene_hash = store.put(_calibrated_scene())
    try:
        manager.create(width=WIDTH, height=HEIGHT, scene_hash=scene_hash)
        summary = manager.summaries()[0]
        assert summary.scene_calibrated is True
        assert summary.camera_id == "cam-fixed-1"
    finally:
        manager.close_all()


def test_a_scene_for_a_different_frame_size_is_refused(tmp_path: Path) -> None:
    """Refused, not applied: applying it would fail silently, which is worse."""

    manager, store = _manager(tmp_path)
    scene_hash = store.put(_calibrated_scene())
    try:
        with pytest.raises(LiveProtocolError, match="would land in the wrong place"):
            manager.create(width=640, height=480, scene_hash=scene_hash)
        assert manager.summaries() == ()
    finally:
        manager.close_all()


def test_an_unknown_scene_hash_is_reported_as_missing(tmp_path: Path) -> None:
    manager, _store = _manager(tmp_path)
    try:
        with pytest.raises(SceneNotFoundError):
            manager.create(width=WIDTH, height=HEIGHT, scene_hash="0" * 64)
    finally:
        manager.close_all()


def test_the_socket_reports_a_missing_scene_instead_of_failing_opaquely(
    tmp_path: Path,
) -> None:
    with live_client(tmp_path) as client, client.websocket_connect("/api/live/ws") as ws:
        ws.send_json(
            {"type": "start", "width": WIDTH, "height": HEIGHT, "scene_hash": "0" * 64}
        )
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "scene_not_found"


def test_live_frames_run_the_calibrated_scenes_rules_end_to_end(tmp_path: Path) -> None:
    """A calibrated live session processes frames through the geometry rule set.

    The point is not that this synthetic stream *confirms* a wrong-way event -- the
    scripted detector drives the motorcycle along the legal direction -- but that
    the rule is genuinely running over live frames, which is what makes a real
    wrong-way vehicle in front of a calibrated camera confirmable at all.
    """

    config = live_config(tmp_path)
    with live_client(tmp_path, config=config) as client:
        # The application roots its scene store at the storage dir (see create_app).
        store = SceneStore(config.storage_dir)
        scene_hash = store.put(_calibrated_scene())
        with client.websocket_connect("/api/live/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "scene_hash": scene_hash,
                }
            )
            opened = ws.receive_json()
            assert opened["type"] == "session", opened
            assert ViolationType.WRONG_WAY in opened["running_violations"]
            assert opened["scene_calibrated"] is True

            for index in range(3):
                ws.send_json(frame_message(index))
                while True:
                    message = ws.receive_json()
                    if message["type"] == "result":
                        break
                    assert message["type"] == "events", message

    assert message["frame_index"] == 2
    assert message["stats"]["frames_processed"] == 3

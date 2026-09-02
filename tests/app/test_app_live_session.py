"""Live camera monitoring: session semantics, isolation, back-pressure, cleanup.

These tests drive the real live stack over stub model backends (see
``_live_helpers``). What they are asserting is that a live stream is processed by
the *same* pipeline an uploaded video is, with the persistent state a camera needs
and none of the state a camera must not accumulate -- so each one is written
against an observable consequence (a track id that survives a frame, an event that
is confirmed only after its rule's persistence window, a session that is gone from
the process after the socket closes), never against an internal call.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from _live_helpers import (
    HEIGHT,
    WIDTH,
    LiveStubProvider,
    frame_message,
    jpeg_frame,
    live_client,
    live_config,
    multi_rider_provider,
    scripted_helmet_classifier,
)
from starlette.websockets import WebSocketDisconnect

from trafficpulse.app.live import LiveConfig, PendingFrame
from trafficpulse.app.live.errors import LiveFrameError
from trafficpulse.contracts.enums import ViolationType

# A cadence that reasons on every frame, so a test does not have to sleep to see an
# event the rule has already earned. It changes *when* events surface, never which.
EAGER = LiveConfig(finalize_interval_seconds=0.0)


def _start(ws: object, *, width: int = WIDTH, height: int = HEIGHT) -> dict[str, object]:
    ws.send_json({"type": "start", "width": width, "height": height})  # type: ignore[attr-defined]
    message: dict[str, object] = ws.receive_json()  # type: ignore[attr-defined]
    assert message["type"] == "session", message
    return message


def _pump(ws: object, frames: int, *, start: int = 0) -> list[dict[str, object]]:
    """Send ``frames`` frames one at a time and collect each frame's result.

    Strictly one frame in flight, so every frame is processed and the assertions
    below are about the pipeline rather than about which frames happened to be
    dropped. Back-pressure gets its own test, where dropping is the subject.
    """

    results: list[dict[str, object]] = []
    for index in range(start, start + frames):
        ws.send_json(frame_message(index))  # type: ignore[attr-defined]
        while True:
            message = ws.receive_json()  # type: ignore[attr-defined]
            if message["type"] == "result":
                results.append(message)
                break
            assert message["type"] == "events", message
            results.append(message)
    return results


def _frame_results(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    return [message for message in messages if message["type"] == "result"]


# --- readiness ------------------------------------------------------------------
def test_status_reports_ready_when_a_backend_is_configured(tmp_path: Path) -> None:
    with live_client(tmp_path) as client:
        body = client.get("/api/live/status").json()
    assert body["ready"] is True
    assert body["inference_configured"] is True
    assert body["drawing_backend_available"] is True
    assert body["max_sessions"] >= 1


def test_status_explains_an_unconfigured_backend(tmp_path: Path) -> None:
    """A deployment with no detector says so *before* a camera is opened."""

    config = live_config(tmp_path, inference=None)
    with live_client(tmp_path, config=config) as client:
        body = client.get("/api/live/status").json()
    assert body["ready"] is False
    assert body["inference_configured"] is False
    assert "no inference backend" in body["detail"].lower()


# --- session opening -------------------------------------------------------------
def test_session_announces_what_it_runs_and_what_it_cannot(tmp_path: Path) -> None:
    """The opening message states the unavailable violations *with their reasons*.

    The point of the assertion: an empty live event feed must be readable as "this
    camera is not being checked for wrong-way" rather than as "no wrong-way
    happened", and only the reason makes that distinction available to a viewer.
    """

    with live_client(tmp_path) as client, client.websocket_connect("/api/live/ws") as ws:
        session = _start(ws)

    assert session["width"] == WIDTH and session["height"] == HEIGHT
    assert session["scene_calibrated"] is False
    assert ViolationType.TRIPLE_RIDING in session["running_violations"]
    unavailable = {
        entry["violation_type"]: entry["reason"]
        for entry in session["unavailable_violations"]
    }
    # Geometry is never derived from a camera view, so these stay off and say why.
    assert ViolationType.WRONG_WAY in unavailable
    assert "legal travel direction" in unavailable[ViolationType.WRONG_WAY]
    assert ViolationType.ILLEGAL_STOPPING in unavailable
    assert ViolationType.RED_LIGHT_JUMPING in unavailable
    assert all(reason.strip() for reason in unavailable.values())


def test_a_session_must_open_with_start(tmp_path: Path) -> None:
    with live_client(tmp_path) as client, client.websocket_connect("/api/live/ws") as ws:
        ws.send_json({"type": "stop"})
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "live_protocol_error"


def test_an_unparseable_message_is_a_clean_protocol_error(tmp_path: Path) -> None:
    with live_client(tmp_path) as client, client.websocket_connect("/api/live/ws") as ws:
        ws.send_text("{not json")
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "live_protocol_error"


def test_binary_messages_are_refused_with_an_explanation(tmp_path: Path) -> None:
    with live_client(tmp_path) as client, client.websocket_connect("/api/live/ws") as ws:
        ws.send_bytes(b"\x00\x01\x02")
        message = ws.receive_json()
    assert message["type"] == "error"
    assert "text messages only" in message["message"]


def test_an_unavailable_backend_refuses_the_session_with_a_reason(tmp_path: Path) -> None:
    config = live_config(tmp_path, inference=None)
    with (
        live_client(tmp_path, config=config) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        ws.send_json({"type": "start", "width": WIDTH, "height": HEIGHT})
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "live_unavailable"


# --- frame processing -------------------------------------------------------------
def test_frames_are_detected_tracked_and_annotated(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        results = _frame_results(_pump(ws, 4))

    assert [result["frame_index"] for result in results] == [0, 1, 2, 3]
    assert [result["sequence"] for result in results] == [0, 1, 2, 3]
    # The scripted detector puts a motorcycle and a rider on every frame, and both
    # reach the client as tracked objects with real boxes.
    classes = {track["object_class"] for track in results[-1]["tracks"]}
    assert classes == {"motorcycle", "person"}
    for track in results[-1]["tracks"]:
        x1, y1, x2, y2 = track["bbox"]
        assert x2 > x1 and y2 > y1
    # Something was drawn: live annotation goes through the shipped overlay
    # framework, so a run that observed riders publishes elements to draw.
    assert results[-1]["annotated"]


def test_track_ids_persist_across_processed_frames(tmp_path: Path) -> None:
    """The defining property of a session: one motorcycle keeps one identity."""

    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        results = _frame_results(_pump(ws, 6))

    def motorcycle_id(result: dict[str, object]) -> str:
        ids = [
            track["track_id"]
            for track in result["tracks"]
            if track["object_class"] == "motorcycle"
        ]
        assert len(ids) == 1, ids
        return str(ids[0])

    identities = {motorcycle_id(result) for result in results[1:]}
    assert len(identities) == 1, f"the motorcycle was re-identified every frame: {identities}"


def test_rider_association_runs_and_reports_helmet_state(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        results = _frame_results(_pump(ws, 5))

    riders = results[-1]["riders"]
    assert riders, "the rider associator produced no association for a rider on a bike"
    rider = riders[0]
    assert rider["motorcycle_track_id"]
    assert rider["helmet_label"] == "no_helmet"
    assert rider["rider_count"] == 1
    # A single rider is the only case the shipped system attributes a driver in.
    assert rider["driver_resolved"] is True


def test_multi_rider_leaves_the_driver_unresolved(tmp_path: Path) -> None:
    """Three riders on one motorcycle: the driver is reported as unresolved.

    Not a live-mode rule -- the tracker supplies no velocity, so no layer of this
    system can say which rider is driving, and live mode reports that rather than
    picking one.
    """

    with (
        live_client(tmp_path, provider=multi_rider_provider(3), live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        results = _frame_results(_pump(ws, 6))

    riders = results[-1]["riders"]
    assert len(riders) == 3, riders
    assert {rider["rider_count"] for rider in riders} == {3}
    assert all(rider["driver_resolved"] is False for rider in riders)


def test_a_confirmed_violation_reaches_the_client_as_the_real_event(
    tmp_path: Path,
) -> None:
    """A live triple-riding event is the very ``ConfirmedEvent`` the rule minted.

    It also has to *earn* confirmation: the rule's persistence window is a second of
    media time, so the first frames confirm nothing. A live event appearing on frame
    one would mean the semantics had been replaced with a per-frame assertion.
    """

    with (
        live_client(tmp_path, provider=multi_rider_provider(3), live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        messages = _pump(ws, 16)

    events = [
        event
        for message in messages
        if message["type"] == "events"
        for event in message["events"]
    ]
    assert events, "no live event was confirmed for a sustained three-rider motorcycle"
    # The live session runs the whole resolved rule set, not one rule: three
    # helmetless riders on one motorcycle earn both violations, exactly as the same
    # footage would through an uploaded-video job.
    kinds = {event["violation_type"] for event in events}
    assert ViolationType.TRIPLE_RIDING in kinds, kinds

    triple = next(
        event for event in events if event["violation_type"] == ViolationType.TRIPLE_RIDING
    )
    assert triple["rule_id"]
    assert triple["track_ids"]
    assert triple["start_at"] < triple["trigger_at"], "the event did not accrue over time"
    # The full contract travels, not a reduced live shape.
    assert "confidence" in triple and "scene_config_hash" in triple and "models" in triple
    # Announced once each, not re-announced on every subsequent frame.
    assert len({event["event_id"] for event in events}) == len(events)


def test_a_frame_of_the_wrong_size_warns_without_ending_the_session(
    tmp_path: Path,
) -> None:
    """A resolution change is refused per-frame, not silently reasoned about."""

    import base64

    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        ws.send_json(
            {
                "type": "frame",
                "sequence": 0,
                "capture_seconds": 0.0,
                "data": base64.b64encode(jpeg_frame(0, width=160, height=120)).decode(),
            }
        )
        warning = ws.receive_json()
        # The session survives and keeps processing correctly-sized frames.
        results = _frame_results(_pump(ws, 1, start=1))

    assert warning["type"] == "warning"
    assert warning["code"] == "live_frame_error"
    assert "160x120" in warning["message"]
    assert results[0]["type"] == "result"


def test_an_inference_failure_ends_the_session_with_a_typed_error(
    tmp_path: Path,
) -> None:
    from _app_helpers import RaisingDetector

    provider = LiveStubProvider(
        detector_factory=RaisingDetector,
        classifier=scripted_helmet_classifier(),
    )
    with (
        live_client(tmp_path, provider=provider, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        ws.send_json(frame_message(0))
        message = ws.receive_json()

    assert message["type"] == "error"
    assert message["code"] == "live_inference_error"


# --- metrics ---------------------------------------------------------------------
def test_reported_metrics_are_measured_not_fabricated(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        results = _frame_results(_pump(ws, 3))

    stats = results[-1]["stats"]
    assert stats["frames_received"] == 3
    assert stats["frames_processed"] == 3
    assert stats["active_tracks"] == 2
    # Latency and FPS come from a real clock over real work: present, positive, and
    # consistent with one another rather than with the camera's capture rate.
    assert stats["latency_ms_last"] is not None and stats["latency_ms_last"] >= 0.0
    assert stats["inference_fps"] is not None and stats["inference_fps"] > 0.0


def test_the_reported_rate_is_throughput_not_the_inverse_of_latency(
    tmp_path: Path,
) -> None:
    """The distinction that decides whether the number on screen is honest.

    Two frames can be in flight, so a frame's end-to-end latency includes the wait
    behind the one before it. Reporting ``1 / latency`` as the frame rate would
    publish roughly *half* the frames per second the system genuinely completes --
    understating the pipeline while claiming to measure it. This drives the session
    with an injected clock where the two answers differ by construction: 1 s of
    processing per frame, but 2 s of measured end-to-end latency.
    """

    manager, session = _session(tmp_path)
    try:
        for index in range(4):
            # ``received_at`` a full second before the clock's current value, so each
            # frame's end-to-end latency is one second longer than its processing.
            session.process(
                PendingFrame(
                    data=jpeg_frame(index),
                    capture_seconds=index * 0.1,
                    client_sequence=index,
                    received_at=time.monotonic() - 1.0,
                )
            )
        stats = session.stats()
    finally:
        manager.close_all()

    assert stats.frames_processed == 4
    assert stats.latency_ms_mean is not None and stats.latency_ms_mean >= 1000.0
    assert stats.processing_ms_mean is not None
    # Latency carries a whole fabricated second of queue wait that processing does
    # not, and the published rate follows processing, not latency.
    assert stats.processing_ms_mean < stats.latency_ms_mean
    assert stats.inference_fps is not None
    assert stats.inference_fps > 1000.0 / stats.latency_ms_mean


def test_no_rate_is_claimed_from_a_single_frame(tmp_path: Path) -> None:
    """One frame measures a duration, not a rate; the field stays null."""

    manager, session = _session(tmp_path)
    try:
        first = session.process(_pending(0))
        assert first.stats.inference_fps is None
        assert first.stats.processing_ms_mean is not None
        second = session.process(_pending(1))
        assert second.stats.inference_fps is not None
    finally:
        manager.close_all()


# --- back-pressure (unit level, where dropping is the subject) ----------------------
def _session(tmp_path: Path, **live_overrides: object):
    """One live session built through the real manager, for direct driving."""

    from trafficpulse.app.live import LiveSessionManager
    from trafficpulse.app.registry import VideoStore
    from trafficpulse.app.services import SceneService, VideoService
    from trafficpulse.persistence import SceneStore

    config = live_config(tmp_path)
    store = VideoStore()
    scenes = SceneService(SceneStore(tmp_path), VideoService(config, store), store)
    manager = LiveSessionManager(
        config=config,
        live_config=LiveConfig(**live_overrides),  # type: ignore[arg-type]
        provider=LiveStubProvider(classifier=scripted_helmet_classifier()),  # type: ignore[arg-type]
        scenes=scenes,
    )
    return manager, manager.create(width=WIDTH, height=HEIGHT)


def _pending(index: int) -> PendingFrame:
    return PendingFrame(
        data=jpeg_frame(index),
        capture_seconds=index * 0.1,
        client_sequence=index,
        received_at=time.monotonic(),
    )


def test_the_queue_holds_one_frame_and_drops_the_stale_one(tmp_path: Path) -> None:
    """Back-pressure: latency cannot accumulate because the backlog cannot exist."""

    manager, session = _session(tmp_path)
    try:
        assert session.offer(_pending(0)) is False
        assert session.offer(_pending(1)) is True, "the older frame was not displaced"
        assert session.offer(_pending(2)) is True
        taken = session.take()
        assert taken is not None
        # What survived is the newest frame, not the oldest: a live view must show
        # the road now, not the road from three inferences ago.
        assert taken.client_sequence == 2
        assert session.take() is None
        stats = session.stats()
        assert stats.frames_received == 3
        assert stats.frames_dropped == 2
    finally:
        manager.close_all()


def test_a_backwards_capture_clock_drops_the_frame_rather_than_restamping_it(
    tmp_path: Path,
) -> None:
    """A clock artefact costs one frame, never a fabricated inter-frame interval.

    Restamping the frame to "just after the last one" would be the convenient fix
    and the wrong one: the tracker's motion, and every duration threshold the
    reasoners build on it, would then rest on an interval nobody measured.
    """

    manager, session = _session(tmp_path)
    try:
        session.process(_pending(5))
        with pytest.raises(LiveFrameError, match="not after the previous frame"):
            session.process(
                PendingFrame(
                    data=jpeg_frame(6),
                    capture_seconds=0.1,  # the client's clock went backwards
                    client_sequence=6,
                    received_at=time.monotonic(),
                )
            )
        assert session.stats().frames_out_of_order == 1
        # The session is unharmed: the next in-order frame processes normally.
        assert session.process(_pending(6)).capture_seconds == pytest.approx(0.6)
    finally:
        manager.close_all()


def test_an_undecodable_frame_is_a_frame_error_not_a_crash(tmp_path: Path) -> None:
    manager, session = _session(tmp_path)
    try:
        with pytest.raises(LiveFrameError):
            session.process(
                PendingFrame(
                    data=b"not an image",
                    capture_seconds=0.0,
                    client_sequence=0,
                    received_at=time.monotonic(),
                )
            )
    finally:
        manager.close_all()


def test_an_oversized_payload_is_refused_before_decoding(tmp_path: Path) -> None:
    manager, session = _session(tmp_path, max_frame_bytes=1024)
    try:
        with pytest.raises(LiveFrameError, match="over the 1024-byte limit"):
            session.process(_pending(0))
    finally:
        manager.close_all()


def test_the_window_rolls_over_and_bounds_accumulated_state(tmp_path: Path) -> None:
    """The memory bound is real: at the boundary the engine's history is released."""

    manager, session = _session(tmp_path, window_frames=30, finalize_interval_seconds=0.0)
    try:
        rollovers = 0
        for index in range(31):
            result = session.process(_pending(index))
            rollovers += int(result.window_rolled_over)
        assert rollovers == 1
        assert session.stats().windows_completed == 1
        # The window that just started holds only the frames processed since the
        # boundary -- which is exactly what stops an all-day session growing.
        assert session.stats().window_frames_processed == 1
        assert len(session.track_states()) <= 2
        assert session.stats().frames_processed == 31
    finally:
        manager.close_all()


# --- isolation, capacity and cleanup ----------------------------------------------
def test_two_sessions_share_no_tracker_state(tmp_path: Path) -> None:
    """Structural isolation: each session gets its own engine from the provider."""

    provider = LiveStubProvider(classifier=scripted_helmet_classifier())
    with (
        live_client(tmp_path, provider=provider, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as first,
        client.websocket_connect("/api/live/ws") as second,
    ):
        first_session = _start(first)
        second_session = _start(second)
        # Drive one session much further than the other; a shared tracker or a
        # shared history would make the second session's first frame inherit it.
        first_results = _frame_results(_pump(first, 8))
        second_results = _frame_results(_pump(second, 1))

        assert first_session["session_id"] != second_session["session_id"]
        assert provider.created == 2, "the two sessions did not get separate engines"
        assert first_results[-1]["stats"]["frames_processed"] == 8
        assert second_results[-1]["stats"]["frames_processed"] == 1
        assert second_results[-1]["frame_index"] == 0

        listing = client.get("/api/live/sessions").json()
        assert len(listing["sessions"]) == 2


def test_the_session_cap_is_enforced_with_a_clear_refusal(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=LiveConfig(max_sessions=1)) as client,
        client.websocket_connect("/api/live/ws") as first,
    ):
        _start(first)
        with client.websocket_connect("/api/live/ws") as second:
            second.send_json({"type": "start", "width": WIDTH, "height": HEIGHT})
            message = second.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "live_capacity_error"
    assert "slot" in message["message"]


def test_closing_the_socket_releases_the_session(tmp_path: Path) -> None:
    """Cleanup is the connection's lifetime: there is no way to leak an engine."""

    with live_client(tmp_path, live=EAGER) as client:
        with client.websocket_connect("/api/live/ws") as ws:
            _start(ws)
            _pump(ws, 2)
            assert len(client.get("/api/live/sessions").json()["sessions"]) == 1
        # The context manager closed the socket without a 'stop' message -- the
        # disconnect path has to clean up exactly as the graceful path does.
        assert client.get("/api/live/sessions").json()["sessions"] == []


def test_a_graceful_stop_reports_the_session_summary(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        _pump(ws, 2)
        ws.send_json({"type": "stop"})
        message = ws.receive_json()
        assert message["type"] == "stopped"
        assert message["stats"]["frames_processed"] == 2
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert client.get("/api/live/sessions").json()["sessions"] == []


def test_a_second_start_on_one_socket_is_refused(tmp_path: Path) -> None:
    with (
        live_client(tmp_path, live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        ws.send_json({"type": "start", "width": WIDTH, "height": HEIGHT})
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["code"] == "live_protocol_error"


def test_a_live_session_persists_no_frame_and_no_event(tmp_path: Path) -> None:
    """Privacy and repository integrity: live mode writes nothing to disk.

    A live event has no source file to render evidence from, so putting one in the
    write-once event store would create a record whose manifest could never be
    resolved. The assertion is the whole storage root, not just the run tree,
    because a stray camera frame anywhere would be the same failure.
    """

    with (
        live_client(tmp_path, provider=multi_rider_provider(3), live=EAGER) as client,
        client.websocket_connect("/api/live/ws") as ws,
    ):
        _start(ws)
        messages = _pump(ws, 16)

    assert any(message["type"] == "events" for message in messages), (
        "the test proves nothing unless the session actually confirmed an event"
    )
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written == [], f"a live session wrote {written}"

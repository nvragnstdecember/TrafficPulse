"""Live camera endpoints: pre-flight readiness, the session listing, and the socket.

The socket handler is the only place in the application where two things run at
once for one client, and the reason is the whole point of live mode: inference is
much slower than capture, so receiving and processing cannot be the same loop. If
they were, the socket would stop being read while a frame was in the detector, the
kernel buffer would fill with frames nobody wanted any more, and the picture on
screen would fall further behind the road with every frame.

So the handler runs two cooperating tasks over one session:

* the **receiver** reads messages and hands each frame to the session's single
  pending slot -- a frame arriving while one is already waiting replaces it, and the
  displaced frame is counted as dropped;
* the **worker** takes whatever is pending, runs it through the engine on a thread
  (inference is blocking CPU work and must never occupy the event loop), and sends
  back the annotated result and any newly confirmed events.

Whichever finishes first ends the other, so a client disconnect stops inference
immediately and an inference failure closes the socket immediately -- there is no
state in which one of the two is still running alone.

Closing is the whole cleanup
----------------------------
The session's lifetime is the connection's lifetime. Every exit path -- clean stop,
client disconnect, protocol error, inference failure, server shutdown -- runs
through the same ``finally``, which closes the session and removes it from the
manager. There is no way to navigate away from the page and leave an engine
running.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..dependencies import LiveManagerDep, LiveManagerWsDep
from ..errors import SceneNotFoundError
from ..live.errors import LiveError, LiveFrameError, LiveProtocolError
from ..live.manager import LiveSessionManager
from ..live.protocol import (
    ErrorMessage,
    EventsMessage,
    FrameMessage,
    LiveSessionListResponse,
    LiveSessionModel,
    ReadinessModel,
    StartMessage,
    StopMessage,
    StoppedMessage,
    WarningMessage,
    parse_client_message,
    readiness_model,
    result_message,
    session_message,
    stats_model,
)
from ..live.session import LiveSession, PendingFrame

_logger = logging.getLogger("trafficpulse.live")

router = APIRouter(tags=["live"])

#: How long the worker waits for a frame before looping. Bounds how quickly a stop
#: is noticed when the camera has gone quiet; short enough to be imperceptible,
#: long enough that an idle session costs nothing.
_IDLE_POLL_SECONDS = 0.25

#: WebSocket close code for a session the server ended on purpose, with a reason
#: already sent as an ``error`` message. 1011 ("internal error") would misreport a
#: refused configuration, and 1000 would misreport a failure as a clean stop.
_CLOSE_SESSION_ERROR = 4400


@router.get(
    "/api/live/status",
    response_model=ReadinessModel,
    summary="Whether this deployment can monitor a live camera",
    description="Pre-flight for live mode: whether an inference backend and a "
    "drawing backend are present, how many session slots are free, and a complete "
    "sentence explaining any 'no'. A client asks this *before* requesting camera "
    "permission, so a deployment that cannot monitor live says so first.",
)
def live_status(live: LiveManagerDep) -> ReadinessModel:
    return readiness_model(live.readiness())


@router.get(
    "/api/live/sessions",
    response_model=LiveSessionListResponse,
    summary="Live sessions currently running in this process",
    description="Operator visibility into live mode. Sessions are in-memory and "
    "ephemeral: nothing here survives a restart, and no camera frame or live event "
    "is stored anywhere.",
)
def live_sessions(live: LiveManagerDep) -> LiveSessionListResponse:
    return LiveSessionListResponse(
        sessions=tuple(
            LiveSessionModel(
                session_id=summary.session_id,
                camera_id=summary.camera_id,
                width=summary.width,
                height=summary.height,
                scene_calibrated=summary.scene_calibrated,
                frames_processed=summary.frames_processed,
                uptime_seconds=summary.uptime_seconds,
            )
            for summary in live.summaries()
        ),
        max_sessions=live.live_config.max_sessions,
    )


@router.websocket("/api/live/ws")
async def live_socket(websocket: WebSocket, live: LiveManagerWsDep) -> None:
    """One live monitoring session, for the lifetime of this connection."""

    await websocket.accept()
    session: LiveSession | None = None
    try:
        session = await _open_session(websocket, live)
        await _run_session(websocket, session)
    except LiveError as exc:
        _logger.info("live socket refused/ended: %s: %s", exc.error_type, exc.message)
        await _fail(websocket, exc.error_type, exc.message)
    except SceneNotFoundError as exc:
        await _fail(websocket, exc.error_type, exc.message)
    except WebSocketDisconnect:
        _logger.info("live client disconnected")
    except Exception:  # noqa: BLE001 - a socket handler must never leak a traceback
        _logger.exception("unhandled error in a live session")
        await _fail(websocket, "internal_error", "an internal error ended the session")
    finally:
        # The single cleanup path for every exit above.
        if session is not None:
            live.close(session.session_id)
        # Both halves have to be connected for a close to be legal, and both cases
        # arise: ``application_state`` is DISCONNECTED once *we* have sent a close
        # (the failure paths do), and ``client_state`` is DISCONNECTED once the peer
        # has gone -- at which point the server has already answered its close frame
        # and sending another is a RuntimeError inside the ASGI server.
        if _is_connected(websocket):
            await websocket.close()


async def _open_session(
    websocket: WebSocket, live: LiveSessionManager
) -> LiveSession:
    """Read the opening ``start`` message and build the session it asks for."""

    message = parse_client_message(await _receive_text(websocket))
    if not isinstance(message, StartMessage):
        raise LiveProtocolError(
            f"a live session must open with a 'start' message, got {message.type!r}"
        )
    # Off the event loop, for the same reason inference is: building the engine
    # realises the real backends, and loading a checkpoint from a cold cache takes
    # tens of seconds. On the loop that stalls *every* connection on the server --
    # long enough that a client's own keepalive gives up on the handshake before the
    # first session is ever announced.
    session = await asyncio.to_thread(
        live.create,
        width=message.width,
        height=message.height,
        scene_hash=message.scene_hash,
    )
    await websocket.send_text(
        session_message(
            session_id=session.session_id,
            camera_id=session.camera_id,
            width=session.width,
            height=session.height,
            scene_hash=message.scene_hash,
            scene_calibrated=message.scene_hash is not None,
            plan=session.plan,
            window_frames=live.live_config.window_frames,
        ).model_dump_json()
    )
    return session


async def _run_session(websocket: WebSocket, session: LiveSession) -> None:
    """Drive the receiver and the worker until either finishes (see the docstring)."""

    stop = asyncio.Event()
    wake = asyncio.Event()
    send_lock = asyncio.Lock()

    async def send(payload: str) -> None:
        # One writer at a time: the worker sends results while the receiver may be
        # sending a warning, and interleaved frames would corrupt the stream.
        async with send_lock:
            await websocket.send_text(payload)

    async def receive_loop() -> None:
        while not stop.is_set():
            message = parse_client_message(await _receive_text(websocket))
            if isinstance(message, StopMessage):
                return
            if isinstance(message, StartMessage):
                raise LiveProtocolError(
                    "this session is already started; open a new connection for a "
                    "new session"
                )
            assert isinstance(message, FrameMessage)
            displaced = session.offer(
                PendingFrame(
                    data=message.decoded(),
                    capture_seconds=message.capture_seconds,
                    client_sequence=message.sequence,
                    received_at=time.monotonic(),
                )
            )
            if displaced:
                _logger.debug(
                    "live session %s dropped a stale frame", session.session_id
                )
            wake.set()

    async def work_loop() -> None:
        while not stop.is_set():
            # Cleared *before* taking, so a frame offered during the take cannot be
            # lost to a clear that follows it.
            wake.clear()
            frame = session.take()
            if frame is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(wake.wait(), timeout=_IDLE_POLL_SECONDS)
                continue
            try:
                # Inference is blocking CPU work: off the event loop, always, or the
                # socket stops being read for the duration of a detector pass.
                result = await asyncio.to_thread(session.process, frame)
            except LiveFrameError as exc:
                # One unusable frame, not a broken session.
                await send(
                    WarningMessage(
                        code=exc.error_type, message=exc.message
                    ).model_dump_json()
                )
                continue
            await send(result_message(result).model_dump_json())
            if result.new_events:
                await send(
                    EventsMessage(events=result.new_events).model_dump_json()
                )

    receiver = asyncio.create_task(receive_loop(), name="live-receive")
    worker = asyncio.create_task(work_loop(), name="live-work")
    try:
        done, _pending = await asyncio.wait(
            {receiver, worker}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        stop.set()
        wake.set()
        for task in (receiver, worker):
            task.cancel()
        await asyncio.gather(receiver, worker, return_exceptions=True)

    # Re-raise whatever ended the session, so the handler's typed error path
    # reports it to the client instead of the session ending silently.
    for task in done:
        exception = task.exception()
        if exception is not None:
            raise exception

    await websocket.send_text(
        StoppedMessage(
            session_id=session.session_id, stats=stats_model(session.stats())
        ).model_dump_json()
    )


def _is_connected(websocket: WebSocket) -> bool:
    """Whether a frame can still legally be sent on this socket."""

    return (
        websocket.application_state is WebSocketState.CONNECTED
        and websocket.client_state is WebSocketState.CONNECTED
    )


async def _receive_text(websocket: WebSocket) -> str:
    """Receive one text message, refusing anything else with a typed error."""

    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(message.get("code", 1000)))
    text = message.get("text")
    if text is None:
        raise LiveProtocolError(
            "the live protocol carries JSON text messages only; frames travel "
            "base64-encoded inside a 'frame' message"
        )
    return str(text)


async def _fail(websocket: WebSocket, code: str, message: str) -> None:
    """Tell the client why the session ended, then close. Never raises."""

    if not _is_connected(websocket):
        return
    try:
        await websocket.send_text(
            ErrorMessage(code=code, message=message).model_dump_json()
        )
        # The close reason is capped at 123 bytes by the protocol, so the full
        # message travels in the error frame above and this is only a label.
        await websocket.close(code=_CLOSE_SESSION_ERROR, reason=code[:120])
    except Exception:  # noqa: BLE001 - the peer may already be gone
        _logger.debug("could not deliver the live failure to the client", exc_info=True)


__all__ = ["router"]

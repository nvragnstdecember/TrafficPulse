"""The live camera WebSocket protocol: every message, typed.

One socket carries a whole session -- control in both directions and frames in
one -- because the session *is* the connection: opening the socket opens the
session and closing it closes the session, so there is no way to leak a backend
session by navigating away, and no id to keep in sync between two channels.

Typed on both sides, deliberately
----------------------------------
Client messages are parsed into these models before anything acts on them, so a
malformed payload is a clean protocol error with a message rather than a
``KeyError`` inside the inference path. Server messages are pydantic models for the
same reason every HTTP response is: the wire shape is declared in one place, and
the frontend's types are written against it.

Frames travel as base64 inside JSON
-----------------------------------
A binary frame would save the base64 overhead's ~33%, at the cost of a second
framing convention (which binary message belongs to which control message) on both
sides. At the rates live mode actually runs -- a handful of frames per second, tens
of kilobytes each -- that overhead is irrelevant and the simplicity is not, so
frames are a field like any other. If measurement ever says otherwise, this is the
only module that changes.

What is deliberately absent
---------------------------
There is no message that carries a violation *decision* from the client, and no
message that lets a client tune a threshold, a rule, or the classifier. Live mode's
semantics come from the scene and the deployment; the socket carries pixels and
timing, never policy.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ...contracts import ConfirmedEvent
from ...contracts.enums import ViolationType
from .errors import LiveProtocolError
from .manager import LiveReadiness
from .scene import LiveRulePlan
from .session import (
    LiveFrameResult,
    LiveMotorcycleView,
    LiveRiderView,
    LiveStats,
    LiveTrackView,
)


class _LiveModel(BaseModel):
    """Frozen, strict base for every live message (extra fields are refused)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- client -> server -----------------------------------------------------------
class StartMessage(_LiveModel):
    """Open a monitoring session for a camera of this exact frame size."""

    type: Literal["start"] = "start"
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    scene_hash: str | None = Field(
        default=None,
        description="A stored scene revision to reason through, for a camera an "
        "analyst has calibrated. Omitted, the session gets a provisional scene that "
        "claims only the frame size, and the violations needing geometry stay "
        "explicitly unavailable.",
    )


class FrameMessage(_LiveModel):
    """One captured camera frame."""

    type: Literal["frame"] = "frame"
    sequence: int = Field(ge=0, description="The client's own frame counter, echoed back.")
    capture_seconds: float = Field(
        ge=0.0,
        description="Capture time relative to the start of capture, from the client's "
        "own monotonic clock. Required: the server never invents a frame's timestamp.",
    )
    data: str = Field(description="Base64-encoded JPEG.")

    def decoded(self) -> bytes:
        """The raw encoded frame bytes.

        Raises:
            LiveProtocolError: the field is not valid base64.
        """

        try:
            return base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise LiveProtocolError(f"frame payload is not valid base64: {exc}") from exc


class StopMessage(_LiveModel):
    """End the session cleanly (the socket closes straight after)."""

    type: Literal["stop"] = "stop"


ClientMessage = Annotated[
    StartMessage | FrameMessage | StopMessage, Field(discriminator="type")
]
_CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str) -> ClientMessage:
    """Parse one client message, or raise a clean protocol error.

    Raises:
        LiveProtocolError: the text is not a valid message of a known type.
    """

    try:
        return _CLIENT_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        errors = exc.errors()
        if not errors:  # pragma: no cover - pydantic always reports at least one
            raise LiveProtocolError("unrecognised live message") from exc
        first = errors[0]
        location = ".".join(str(part) for part in first["loc"]) or "message"
        raise LiveProtocolError(
            f"unrecognised live message: {location}: {first['msg']}"
        ) from exc


# --- server -> client -----------------------------------------------------------
class UnavailableViolationModel(_LiveModel):
    """One violation this session will not evaluate, and the reason."""

    violation_type: ViolationType
    reason: str


class SessionMessage(_LiveModel):
    """Sent once, before any frame: what this session is and what it will do.

    Carries the unavailable set as well as the running one on purpose. A live view
    that only listed what it runs would let a viewer read an empty event feed as
    "no violations happened", when the truthful reading may be "wrong-way is not
    being evaluated on this camera at all".
    """

    type: Literal["session"] = "session"
    session_id: str
    camera_id: str
    width: int
    height: int
    scene_hash: str | None
    scene_calibrated: bool
    running_violations: tuple[ViolationType, ...]
    unavailable_violations: tuple[UnavailableViolationModel, ...]
    window_frames: int = Field(
        description="Processed frames per analysis window. At the boundary the engine "
        "is reset to bound memory: track ids restart and a violation whose support "
        "straddles the boundary is not confirmed."
    )


class TrackModel(_LiveModel):
    """One tracked object on a processed frame."""

    track_id: str
    object_class: str
    status: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


class MotorcycleModel(_LiveModel):
    """One motorcycle's occupancy, and whether a driver can be attributed at all."""

    motorcycle_track_id: str
    rider_count: int
    driver_resolved: bool


class RiderModel(_LiveModel):
    """One associated rider, with helmet state where the classifier read one."""

    rider_track_id: str
    motorcycle_track_id: str
    rider_count: int
    driver_resolved: bool
    helmet_label: str | None
    helmet_confidence: float | None
    helmet_gated: bool


class StatsModel(_LiveModel):
    """Measured session metrics. Every field is counted or timed, never estimated."""

    frames_received: int
    frames_dropped: int
    frames_processed: int
    frames_rejected: int
    frames_out_of_order: int
    active_tracks: int
    events_emitted: int
    windows_completed: int
    window_frames_processed: int
    uptime_seconds: float
    inference_fps: float | None
    processing_ms_mean: float | None
    latency_ms_mean: float | None
    latency_ms_last: float | None


class ResultMessage(_LiveModel):
    """The outcome of one processed frame."""

    type: Literal["result"] = "result"
    frame_index: int
    sequence: int
    capture_seconds: float
    tracks: tuple[TrackModel, ...]
    motorcycles: tuple[MotorcycleModel, ...]
    riders: tuple[RiderModel, ...]
    annotated: str | None = Field(
        description="Base64 JPEG of the frame with this system's own overlay drawn "
        "on it, or null when the run published nothing to draw on that frame -- the "
        "client then keeps showing its camera preview rather than a blank frame."
    )
    window_rolled_over: bool
    stats: StatsModel


class EventsMessage(_LiveModel):
    """Newly confirmed violations, as the very ``ConfirmedEvent`` the rules minted.

    The contract is sent verbatim rather than reduced to a display shape: a live
    event is not a lesser kind of event, and anything a reviewer would need to
    justify it -- the rule that confirmed it, its confidence components, the
    observation window, the model provenance -- travels with it.
    """

    type: Literal["events"] = "events"
    events: tuple[ConfirmedEvent, ...]


class WarningMessage(_LiveModel):
    """Something went wrong with one frame; the session continues."""

    type: Literal["warning"] = "warning"
    code: str
    message: str


class ErrorMessage(_LiveModel):
    """Something went wrong with the session; the socket closes after this."""

    type: Literal["error"] = "error"
    code: str
    message: str


class StoppedMessage(_LiveModel):
    """The session has ended and its state is released."""

    type: Literal["stopped"] = "stopped"
    session_id: str
    stats: StatsModel


class ReadinessModel(_LiveModel):
    """The pre-flight answer: can this deployment monitor a live camera at all?"""

    ready: bool
    detail: str
    active_sessions: int
    max_sessions: int
    inference_configured: bool
    drawing_backend_available: bool
    helmet_classifier_configured: bool


class LiveSessionModel(_LiveModel):
    """One open session, in the operator-facing listing."""

    session_id: str
    camera_id: str
    width: int
    height: int
    scene_calibrated: bool
    frames_processed: int
    uptime_seconds: float


class LiveSessionListResponse(_LiveModel):
    """Every live session this process is currently running."""

    sessions: tuple[LiveSessionModel, ...]
    max_sessions: int


# --- conversions ----------------------------------------------------------------
def stats_model(stats: LiveStats) -> StatsModel:
    return StatsModel(
        frames_received=stats.frames_received,
        frames_dropped=stats.frames_dropped,
        frames_processed=stats.frames_processed,
        frames_rejected=stats.frames_rejected,
        frames_out_of_order=stats.frames_out_of_order,
        active_tracks=stats.active_tracks,
        events_emitted=stats.events_emitted,
        windows_completed=stats.windows_completed,
        window_frames_processed=stats.window_frames_processed,
        uptime_seconds=stats.uptime_seconds,
        inference_fps=stats.inference_fps,
        processing_ms_mean=stats.processing_ms_mean,
        latency_ms_mean=stats.latency_ms_mean,
        latency_ms_last=stats.latency_ms_last,
    )


def _track_model(track: LiveTrackView) -> TrackModel:
    return TrackModel(
        track_id=track.track_id,
        object_class=track.object_class,
        status=track.status,
        bbox=track.bbox,
        confidence=track.confidence,
    )


def _motorcycle_model(motorcycle: LiveMotorcycleView) -> MotorcycleModel:
    return MotorcycleModel(
        motorcycle_track_id=motorcycle.motorcycle_track_id,
        rider_count=motorcycle.rider_count,
        driver_resolved=motorcycle.driver_resolved,
    )


def _rider_model(rider: LiveRiderView) -> RiderModel:
    return RiderModel(
        rider_track_id=rider.rider_track_id,
        motorcycle_track_id=rider.motorcycle_track_id,
        rider_count=rider.rider_count,
        driver_resolved=rider.driver_resolved,
        helmet_label=rider.helmet_label,
        helmet_confidence=rider.helmet_confidence,
        helmet_gated=rider.helmet_gated,
    )


def result_message(result: LiveFrameResult) -> ResultMessage:
    """The wire form of one processed frame."""

    return ResultMessage(
        frame_index=result.frame_index,
        sequence=result.client_sequence,
        capture_seconds=result.capture_seconds,
        tracks=tuple(_track_model(track) for track in result.tracks),
        motorcycles=tuple(
            _motorcycle_model(motorcycle) for motorcycle in result.motorcycles
        ),
        riders=tuple(_rider_model(rider) for rider in result.riders),
        annotated=(
            base64.b64encode(result.annotated_jpeg).decode("ascii")
            if result.annotated_jpeg is not None
            else None
        ),
        window_rolled_over=result.window_rolled_over,
        stats=stats_model(result.stats),
    )


def session_message(
    *,
    session_id: str,
    camera_id: str,
    width: int,
    height: int,
    scene_hash: str | None,
    scene_calibrated: bool,
    plan: LiveRulePlan,
    window_frames: int,
) -> SessionMessage:
    """The wire form of a session's opening announcement."""

    return SessionMessage(
        session_id=session_id,
        camera_id=camera_id,
        width=width,
        height=height,
        scene_hash=scene_hash,
        scene_calibrated=scene_calibrated,
        running_violations=plan.supported,
        unavailable_violations=tuple(
            UnavailableViolationModel(
                violation_type=entry.violation_type, reason=entry.reason
            )
            for entry in plan.unavailable
        ),
        window_frames=window_frames,
    )


def readiness_model(readiness: LiveReadiness) -> ReadinessModel:
    return ReadinessModel(
        ready=readiness.ready,
        detail=readiness.detail,
        active_sessions=readiness.active_sessions,
        max_sessions=readiness.max_sessions,
        inference_configured=readiness.inference_configured,
        drawing_backend_available=readiness.drawing_backend_available,
        helmet_classifier_configured=readiness.helmet_classifier_configured,
    )

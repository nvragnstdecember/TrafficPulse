"""Typed application errors + their HTTP mapping (H7A).

Every failure the API layer raises is an :class:`AppError` carrying its own HTTP
status and a stable ``error_type`` slug. A single exception handler (registered
in :mod:`trafficpulse.app.api`) turns any :class:`AppError` into the consistent
JSON envelope ``{"error": {"type", "message"}}`` -- so routers never build error
responses by hand, no traceback ever reaches a client, and the status/type/shape
are uniform across every endpoint.

Lower-layer errors are **translated**, not leaked: the services catch the typed
errors of the composed layers (``VideoIngestionError``, ``SceneConfigurationError``,
``PersistenceError``, ``DetectorError`` ...) and re-raise the matching
:class:`AppError`, so the HTTP contract stays decoupled from which internal
component failed.
"""

from __future__ import annotations


class AppError(Exception):
    """Base application error: an HTTP status plus a stable, safe message.

    ``status_code`` is the response status; ``error_type`` defaults to the class
    name (a stable machine-readable slug); ``message`` is a client-safe string
    (never a traceback or internal path dump).
    """

    status_code: int = 500
    error_type: str = "AppError"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequestError(AppError):
    """A malformed or semantically invalid request (400)."""

    status_code = 400
    error_type = "bad_request"


class UnsupportedMediaError(AppError):
    """The uploaded file is not a supported / readable video (400)."""

    status_code = 400
    error_type = "unsupported_media"


class InvalidConfigurationError(AppError):
    """The requested processing configuration is invalid for the scene (400)."""

    status_code = 400
    error_type = "invalid_configuration"


class VideoNotFoundError(AppError):
    """No uploaded video matches the requested id (404)."""

    status_code = 404
    error_type = "video_not_found"


class VideoMediaNotFoundError(AppError):
    """The video is known but its stored file is gone (404).

    Deliberately distinct from :class:`VideoNotFoundError`: the repository still
    holds the upload's metadata, its runs, and its events -- only the playable
    bytes have been removed from disk. A client can keep showing the video's
    analysis and simply omit playback, which it could not do if a deleted file
    were reported as an unknown video.
    """

    status_code = 404
    error_type = "video_media_not_found"


class SceneNotFoundError(AppError):
    """No stored scene matches the requested hash, or a video has none (404).

    Covers both "this repository has never stored that revision" and "this video
    has not been calibrated". They are the same answer to a client -- there is no
    scene to show you -- and distinguishing them would only invite a caller to
    branch on something it cannot act on differently.
    """

    status_code = 404
    error_type = "scene_not_found"


class JobNotFoundError(AppError):
    """No processing job matches the requested id (404)."""

    status_code = 404
    error_type = "job_not_found"


class EventNotFoundError(AppError):
    """No confirmed event matches the requested id (404)."""

    status_code = 404
    error_type = "event_not_found"


class OverlayNotFoundError(AppError):
    """No rendered overlay video is available for the requested job (404).

    A job may legitimately have none: it is still running, it failed, or its run
    produced no overlay metadata (no observed riders). The original video is always
    playable regardless."""

    status_code = 404
    error_type = "overlay_not_found"


class DuplicateVideoError(AppError):
    """An identical video (same content) has already been uploaded (409)."""

    status_code = 409
    error_type = "duplicate_video"

    def __init__(self, message: str, *, video_id: str) -> None:
        super().__init__(message)
        self.video_id = video_id


class InvalidTransitionError(AppError):
    """The requested review action is not legal from the event's current state (409).

    A conflict rather than a validation error: the request is well-formed and the
    action exists, but the case has moved on. Returning 409 is what lets a client
    distinguish "you sent nonsense" from "somebody else decided this while you were
    looking at it" -- the latter must never silently overwrite their decision.
    """

    status_code = 409
    error_type = "invalid_transition"


class PayloadTooLargeError(AppError):
    """The upload exceeds the configured maximum size (413)."""

    status_code = 413
    error_type = "payload_too_large"


class EngineUnavailableError(AppError):
    """The inference backend is not configured or cannot be built (503).

    Distinct from a client error: the request was well-formed, but the server's
    real detector backend (RT-DETR checkpoint / optional extra) is unavailable.
    """

    status_code = 503
    error_type = "engine_unavailable"

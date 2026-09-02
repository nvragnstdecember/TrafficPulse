"""Typed live-mode failures.

Live mode's failures reach a client over a WebSocket, not over HTTP, so they carry
a stable slug rather than a status code -- but they follow the same rule the HTTP
layer follows: every failure is typed, every message is client-safe, and nothing is
swallowed. The socket close that follows one of these always carries its slug and
message, so no live session ever ends without saying why.
"""

from __future__ import annotations


class LiveError(Exception):
    """Base live-session failure: a stable slug plus a client-safe message."""

    error_type: str = "live_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LiveProtocolError(LiveError):
    """The client sent something the session protocol does not allow."""

    error_type = "live_protocol_error"


class LiveFrameError(LiveError):
    """A frame could not be decoded, or violates the configured size guards."""

    error_type = "live_frame_error"


class LiveCapacityError(LiveError):
    """The process already holds as many live sessions as it will run."""

    error_type = "live_capacity_error"


class LiveUnavailableError(LiveError):
    """Live mode cannot start: no inference backend, or no drawing backend."""

    error_type = "live_unavailable"


class LiveInferenceError(LiveError):
    """Inference raised while processing a live frame.

    Terminal for the session. A detector that failed once on a live stream has no
    defined recovery -- the tracker's state is now built on a gap it does not know
    about -- so the session is closed with this error rather than continued in a
    state whose track identities quietly mean less than they claim.
    """

    error_type = "live_inference_error"

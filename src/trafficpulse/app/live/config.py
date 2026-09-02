"""Deployment knobs for live camera monitoring (frozen, strict).

Every number here is a *pacing* or *bounding* decision, never a semantic one: no
value in this file changes what a detector, tracker, associator, classifier or
reasoner concludes. The detector's score threshold, the tracker's IoU, the
reasoners' persistence windows are all resolved exactly where an uploaded video
resolves them -- through the engine provider and the scene -- so live mode cannot
drift from file mode by configuration.

Why a window rather than an unbounded session
---------------------------------------------
The engine accumulates per-track history for the whole stream and rebuilds its
reasoners over that history on every ``finalize`` (the P3-U2 idempotence
guarantee). That is exactly right for a clip of known length and unbounded for a
camera that runs all day: memory and per-finalize cost both grow linearly with
uptime. :attr:`LiveConfig.window_frames` bounds both by resetting the engine after
that many *processed* frames.

The cost of that bound is stated rather than hidden: a reset returns the tracker to
its initial state, so track ids restart at the boundary and a violation whose
support straddles it is not confirmed. Nothing is faked across the seam -- after a
reset the system genuinely does not know the motorcycle is the same one, and the UI
says so.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LiveConfig(BaseModel):
    """Frozen pacing/bounding configuration for live camera sessions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_sessions: int = Field(default=2, ge=1, le=16)
    """How many live sessions one process will hold at once.

    Low by design. Each session owns a full engine -- its own detector handle,
    tracker and observers -- and inference is the process's scarce resource, so a
    third concurrent session does not make anything faster; it makes every session
    slower and multiplies model memory. A request past the cap is refused with a
    clear message rather than accepted into a queue nobody is watching."""

    window_frames: int = Field(default=600, ge=30, le=100_000)
    """Processed frames per analysis window before the engine is reset.

    See the module docstring. 600 frames is roughly eight minutes at the ~1.2 fps
    this project actually measures on CPU, and a few minutes on a GPU -- long
    enough that a rollover is rare during a demo, short enough that an all-day
    session cannot grow without bound."""

    finalize_interval_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    """Minimum wall time between reasoning passes.

    ``finalize`` is a whole-window pass, so running it after every processed frame
    would make reasoning cost grow with window length on the critical path. Events
    are therefore surfaced at this cadence rather than instantly. It delays a live
    event by at most this much; it changes no event's content, because every event
    field is a pure function of the history prefix up to its trigger."""

    max_frame_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    """Largest accepted encoded camera frame. A guard on an untrusted client, not a
    quality setting: a browser frame at 640x480/JPEG is tens of kilobytes."""

    max_frame_pixels: int = Field(default=1920 * 1080, ge=64 * 64)
    """Largest accepted decoded frame area, as a decompression-bomb guard: the byte
    limit above bounds the *compressed* size, which a hostile image can hide behind."""

    jpeg_quality: int = Field(default=70, ge=1, le=100)
    """Encoding quality of the annotated frame sent back to the browser."""

    latency_samples: int = Field(default=30, ge=1, le=1000)
    """How many recent frames the reported inference latency/FPS average over.

    A rolling window rather than a session total, so the number on screen tracks
    what the system is doing *now* -- which is the only useful reading when a
    backend warms up, or slows down under a busier scene."""

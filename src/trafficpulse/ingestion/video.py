"""Deterministic local-video ingestion (P1-U5).

Establishes the boundary between a local video source and the rest of
TrafficPulse: it opens a file, validates it, exposes stable source metadata, and
iterates decoded frames as immutable ``FrameRecord`` values with deterministic
source/frame identity and media-relative timestamps. It is *ingestion only* --
no detection, tracking, observation, rule, event, or evidence logic.

Backend
-------
PyAV (the architecture-selected backend for PTS-accurate decode,
architecture-review §151/§17; ``docs/windows-verification.md``). PyAV's wheels
bundle FFmpeg, so no system FFmpeg is required on Windows or CI. Decoded frames
are exposed as RGB ``uint8`` arrays.

Timestamps
----------
``FrameRecord.timestamp_seconds`` is **media-relative** (seconds from the start
of the stream), computed **only** from the presentation timestamp:
``pts * time_base``. The architecture requires media time to come from PTS only
(VFR discipline, architecture-review §17: "dt from PTS only ... anomalous-dt
segments rejected"); there is **no** nominal-FPS fallback, which would fabricate
media time and be wrong for variable-frame-rate sources. A frame that lacks a
usable PTS is rejected with :class:`MissingTimestampError` rather than assigned a
fabricated timestamp. Timestamps are deterministic for repeated reads and
non-decreasing (PyAV yields frames in presentation order). They are deliberately
**not** an absolute ``AwareDatetime``: a video provides only media-relative time,
and fabricating a capture date would corrupt the absolute-datetime distinction
the domain contracts keep (architecture-review §239). Mapping to absolute capture
time is a later concern requiring external metadata.

Identity
--------
Source and frame identity are deterministic SHA-256 digests -- no wall clock, no
randomness, no builtin ``hash()``. Source identity is path-based (see
:func:`open_video`); it is a source *label*, not evidence-integrity hashing.
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import av.container
import numpy as np
from numpy.typing import NDArray

# RGB, 8-bit, 3 channels: the documented ingestion payload format.
_PIXEL_FORMAT = "rgb24"


# --- errors ------------------------------------------------------------------
class VideoIngestionError(Exception):
    """Base class for all video-ingestion errors."""


class SourceNotFoundError(VideoIngestionError):
    """The source path does not exist."""


class NotARegularFileError(VideoIngestionError):
    """The source path exists but is not a regular file (e.g. a directory)."""


class UnreadableVideoError(VideoIngestionError):
    """The source exists but cannot be opened or decoded as video (corrupt)."""


class NoDecodableFramesError(VideoIngestionError):
    """The source opens but exposes no decodable video frames."""


class MissingTimestampError(VideoIngestionError):
    """A decoded frame has no usable presentation timestamp (PTS).

    Media time must come from PTS only (architecture-review §17); a frame lacking
    a PTS is rejected rather than assigned a fabricated (e.g. nominal-FPS) time.
    """


# --- timestamps --------------------------------------------------------------
def _media_timestamp(pts: int | None, time_base: Fraction | None) -> float:
    """Return media-relative seconds ``pts * time_base`` (PTS only, no fallback).

    Raises:
        MissingTimestampError: if ``pts`` or ``time_base`` is ``None``.
    """

    if pts is None or time_base is None:
        raise MissingTimestampError(
            "decoded frame has no usable presentation timestamp (PTS); "
            "media time must come from PTS only"
        )
    return float(pts * time_base)


# --- identity ----------------------------------------------------------------
def _source_id(canonical_path: str) -> str:
    return "vsrc-" + hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:16]


def _frame_id(source_id: str, frame_index: int) -> str:
    preimage = f"{source_id}\x1f{frame_index}"
    return "vfrm-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


# --- runtime boundary types --------------------------------------------------
@dataclass(frozen=True)
class VideoSourceMetadata:
    """Stable, deterministic metadata describing one opened video source.

    A runtime type (not a domain contract): no existing frozen contract
    represents a video source. ``fps``, ``frame_count``, and ``duration_seconds``
    are ``None`` when the container does not report them reliably.
    """

    source_id: str
    path: str
    camera_id: str | None
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    codec: str


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame at the ingestion boundary.

    ``image`` is an RGB ``uint8`` array of shape ``(height, width, 3)``,
    C-contiguous, owned by this record (safe to keep after the reader closes).
    It is excluded from equality and repr so records compare by their stable
    identity/metadata fields rather than by pixel content.
    """

    source_id: str
    camera_id: str | None
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    image: NDArray[np.uint8] = field(compare=False, repr=False)


# --- reader ------------------------------------------------------------------
class VideoReader:
    """Single-pass, deterministic sequential frame reader over one source.

    Construct via :func:`open_video`. Supports the context-manager protocol and
    an explicit :meth:`close`. Iteration yields ``FrameRecord`` values with
    ``frame_index`` starting at 0 and increasing by one. Fully iterating the
    reader releases the underlying container automatically; partial iteration
    should be paired with :meth:`close` (or a ``with`` block). Iteration raises
    :class:`MissingTimestampError` if a decoded frame has no usable PTS.
    """

    def __init__(
        self,
        container: "av.container.InputContainer",
        time_base: Fraction | None,
        decoder: Iterator["av.VideoFrame"],
        first_frame: "av.VideoFrame",
        metadata: VideoSourceMetadata,
    ) -> None:
        self.metadata = metadata
        self._container = container
        self._time_base = time_base
        self._decoder = decoder
        self._first_frame: av.VideoFrame | None = first_frame
        self._started = False
        self._closed = False

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FrameRecord]:
        if self._closed:
            raise VideoIngestionError("cannot iterate a closed VideoReader")
        if self._started:
            raise VideoIngestionError("VideoReader is single-pass; open a new reader to re-read")
        self._started = True
        return self._generate()

    def _generate(self) -> Iterator[FrameRecord]:
        try:
            index = 0
            frame = self._first_frame
            self._first_frame = None
            while frame is not None:
                yield self._to_record(frame, index)
                index += 1
                try:
                    frame = next(self._decoder)
                except StopIteration:
                    frame = None
                except av.FFmpegError as exc:
                    raise UnreadableVideoError(f"decode failed at frame {index}: {exc}") from exc
        finally:
            self.close()

    def _to_record(self, frame: "av.VideoFrame", index: int) -> FrameRecord:
        timestamp = _media_timestamp(frame.pts, self._time_base)
        image: NDArray[np.uint8] = np.ascontiguousarray(
            frame.to_ndarray(format=_PIXEL_FORMAT), dtype=np.uint8
        )
        return FrameRecord(
            source_id=self.metadata.source_id,
            camera_id=self.metadata.camera_id,
            frame_id=_frame_id(self.metadata.source_id, index),
            frame_index=index,
            timestamp_seconds=timestamp,
            width=frame.width,
            height=frame.height,
            image=image,
        )

    def close(self) -> None:
        """Release the underlying container. Idempotent."""

        if self._closed:
            return
        self._closed = True
        self._container.close()


# --- factory -----------------------------------------------------------------
def open_video(
    path: str | Path,
    *,
    camera_id: str | None = None,
    source_id: str | None = None,
) -> VideoReader:
    """Open a local video file and return a validated :class:`VideoReader`.

    The reader is guaranteed to expose at least one decodable frame. On any
    failure the underlying container is closed, so no half-open reader escapes.

    Args:
        path: local filesystem path to a video file.
        camera_id: optional logical camera id (from scene configuration) recorded
            on the metadata and every frame; never inferred from the file.
        source_id: optional explicit source identity; defaults to a deterministic
            digest of the resolved path (stable per file location, not content-
            addressed -- a moved/copied file gets a different id).

    Raises:
        SourceNotFoundError: the path does not exist.
        NotARegularFileError: the path is not a regular file.
        UnreadableVideoError: the file cannot be opened/decoded as video.
        NoDecodableFramesError: the file opens but has no decodable video frames.
    """

    candidate = Path(path)
    if not candidate.exists():
        raise SourceNotFoundError(f"video source does not exist: {candidate}")
    if not candidate.is_file():
        raise NotARegularFileError(f"video source is not a regular file: {candidate}")

    canonical = candidate.resolve().as_posix()
    resolved_source_id = source_id if source_id is not None else _source_id(canonical)

    try:
        container = av.open(str(candidate))
    except (av.FFmpegError, OSError) as exc:
        raise UnreadableVideoError(f"cannot open video {candidate}: {exc}") from exc

    try:
        video_streams = container.streams.video
        if not video_streams:
            raise NoDecodableFramesError(f"no video stream in {candidate}")
        stream = video_streams[0]
        decoder = container.decode(stream)
        try:
            first_frame = next(decoder)
        except StopIteration:
            raise NoDecodableFramesError(
                f"video stream in {candidate} yielded no frames"
            ) from None
        except av.FFmpegError as exc:
            raise UnreadableVideoError(f"cannot decode {candidate}: {exc}") from exc
        metadata = _build_metadata(resolved_source_id, canonical, camera_id, stream, container)
    except BaseException:
        container.close()
        raise

    return VideoReader(container, stream.time_base, decoder, first_frame, metadata)


def _build_metadata(
    source_id: str,
    canonical_path: str,
    camera_id: str | None,
    stream: "av.VideoStream",
    container: "av.container.InputContainer",
) -> VideoSourceMetadata:
    average_rate = stream.average_rate
    duration = container.duration
    return VideoSourceMetadata(
        source_id=source_id,
        path=canonical_path,
        camera_id=camera_id,
        width=stream.width,
        height=stream.height,
        fps=float(average_rate) if average_rate else None,
        frame_count=stream.frames or None,
        duration_seconds=float(duration) / 1_000_000.0 if duration is not None else None,
        codec=stream.codec_context.name,
    )

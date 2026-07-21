"""Frame sources for the engine: files today, live adapters through one seam (H6).

:class:`FrameSource` is the engine's input abstraction: anything that can yield
an ordered stream of P1-U5 ``FrameRecord`` values. Two implementations ship:

* :class:`FileFrameSource` -- a local video file (mp4 and anything the bundled
  FFmpeg decodes), built **on** the existing P1-U5 ``open_video`` ingestion --
  PTS-only media time, deterministic identity, typed errors -- not a second
  decoder. Each :meth:`~FileFrameSource.frames` call opens a fresh single-pass
  reader, so one source object supports deterministic replay.
* :class:`IterableFrameSource` -- the **live-source adapter seam**: it wraps any
  iterator of ``FrameRecord``s (a webcam capture loop, an RTSP client, a test
  script) and enforces the stream discipline the engine's tracker seam requires
  (strictly ascending ``frame_index`` and non-decreasing timestamps), failing
  loudly with :class:`FrameSourceError` instead of fabricating order.

Webcam / RTSP posture (deliberate, documented)
----------------------------------------------
The webcam abstraction **is** this seam: a device-capture loop is a producer of
``FrameRecord``s (:func:`frame_record_from_array` stamps deterministic identity
onto raw arrays given a producer-supplied timestamp), wrapped in
:class:`IterableFrameSource`. No device-capture class ships in this unit: PyAV
device input needs platform capture formats (dshow/v4l2/avfoundation) that
cannot run deterministically -- or at all -- in CI, and shipping an untestable
class would violate the project's no-untested-code posture. RTSP arrives the
same way later: a network-reader source implementing :class:`FrameSource`
(PyAV opens network URLs) whose integration needs a reachable stream and is
therefore an opt-in, separately-tested unit. Nothing in the engine assumes a
file: it consumes only this protocol.

Timestamps are producer-supplied, never fabricated: a file source reads PTS; a
live producer must pass its own capture-relative seconds. This is the same
media-time honesty rule ingestion establishes (no nominal-FPS fallback).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..ingestion.video import FrameRecord, VideoSourceMetadata, open_video
from .errors import FrameSourceError


@runtime_checkable
class FrameSource(Protocol):
    """An ordered, re-iterable stream of ``FrameRecord``s with a stable identity."""

    @property
    def source_id(self) -> str:
        """Deterministic identity of this source (stamped on its frames)."""
        ...

    def frames(self) -> Iterator[FrameRecord]:
        """Yield the stream in order. Each call starts a fresh iteration when the
        underlying medium supports replay (files); a live adapter may be
        single-shot and must then raise on a second call rather than yield a
        different stream."""
        ...


def frame_record_from_array(
    image: NDArray[np.uint8],
    *,
    source_id: str,
    frame_index: int,
    timestamp_seconds: float,
    camera_id: str | None = None,
) -> FrameRecord:
    """Stamp deterministic identity onto one live-captured RGB frame.

    The identity scheme mirrors P1-U5 ingestion (``vfrm-`` + SHA-256 over
    ``source_id`` + index), so live frames and file frames are indistinguishable
    downstream. ``timestamp_seconds`` is the **producer's** capture-relative
    media time -- required, never fabricated here. ``image`` must be an RGB
    ``uint8`` array of shape ``(height, width, 3)`` (the ingestion payload
    contract).

    Raises:
        FrameSourceError: if the image is not an RGB uint8 (H, W, 3) array, or
            the index/timestamp is negative.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise FrameSourceError(
            "live frame image must be an RGB uint8 array of shape (height, width, 3); "
            f"got dtype={image.dtype!s}, shape={image.shape!r}"
        )
    if frame_index < 0:
        raise FrameSourceError(f"frame_index must be non-negative, got {frame_index}")
    if timestamp_seconds < 0:
        raise FrameSourceError(
            f"timestamp_seconds must be non-negative, got {timestamp_seconds}"
        )
    preimage = f"{source_id}\x1f{frame_index}"
    return FrameRecord(
        source_id=source_id,
        camera_id=camera_id,
        frame_id="vfrm-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16],
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        image=image,
    )


class FileFrameSource:
    """A local video file behind the :class:`FrameSource` seam (P1-U5 inside).

    Construction validates the file eagerly (opens it once and closes it), so a
    missing/corrupt path fails at build time with the ingestion taxonomy, not
    mid-stream. Each :meth:`frames` call opens a fresh reader -- deterministic
    replay from one object.
    """

    def __init__(self, path: str | Path, *, camera_id: str | None = None) -> None:
        self._path = Path(path)
        self._camera_id = camera_id
        with open_video(self._path, camera_id=camera_id) as reader:
            self._metadata = reader.metadata

    @property
    def source_id(self) -> str:
        return self._metadata.source_id

    @property
    def metadata(self) -> VideoSourceMetadata:
        """The P1-U5 ``VideoSourceMetadata`` captured at construction."""

        return self._metadata

    def frames(self) -> Iterator[FrameRecord]:
        with open_video(self._path, camera_id=self._camera_id) as reader:
            yield from reader


class IterableFrameSource:
    """Any ``FrameRecord`` iterable behind the seam -- the live-source adapter.

    Single-shot by design (a live stream cannot be replayed): a second
    :meth:`frames` call raises :class:`FrameSourceError` instead of silently
    yielding a different stream. The wrapped iterable's order is validated --
    strictly ascending ``frame_index``, non-decreasing ``timestamp_seconds`` --
    because the downstream tracker seam requires it and a live adapter bug must
    surface here, loudly, not as a tracker error mid-pipeline.
    """

    def __init__(self, records: Iterable[FrameRecord], *, source_id: str) -> None:
        if not source_id:
            raise FrameSourceError("source_id must be non-empty")
        self._records = iter(records)
        self._source_id = source_id
        self._consumed = False

    @property
    def source_id(self) -> str:
        return self._source_id

    def frames(self) -> Iterator[FrameRecord]:
        if self._consumed:
            raise FrameSourceError(
                f"live source {self._source_id!r} is single-shot and was already consumed"
            )
        self._consumed = True
        return self._validated()

    def _validated(self) -> Iterator[FrameRecord]:
        previous_index: int | None = None
        previous_ts: float | None = None
        for record in self._records:
            if previous_index is not None and record.frame_index <= previous_index:
                raise FrameSourceError(
                    f"source {self._source_id!r} frame_index went {previous_index} -> "
                    f"{record.frame_index}; frame indices must be strictly ascending"
                )
            if previous_ts is not None and record.timestamp_seconds < previous_ts:
                raise FrameSourceError(
                    f"source {self._source_id!r} timestamp went {previous_ts} -> "
                    f"{record.timestamp_seconds}; media time must be non-decreasing"
                )
            previous_index = record.frame_index
            previous_ts = record.timestamp_seconds
            yield record

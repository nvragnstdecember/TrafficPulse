"""Local-video ingestion for TrafficPulse (P1-U5).

The deterministic boundary between a local video source and the rest of the
system: open + validate a file, expose stable source metadata, and iterate
decoded frames as immutable ``FrameRecord`` values with deterministic identity
and media-relative timestamps. Ingestion only -- it knows nothing about
detection, tracking, observations, rules, events, or evidence.
"""

from .video import (
    FrameRecord,
    MissingTimestampError,
    NoDecodableFramesError,
    NotARegularFileError,
    SourceNotFoundError,
    UnreadableVideoError,
    VideoIngestionError,
    VideoReader,
    VideoSourceMetadata,
    open_video,
)

__all__ = [
    "open_video",
    "VideoReader",
    "VideoSourceMetadata",
    "FrameRecord",
    # errors
    "VideoIngestionError",
    "SourceNotFoundError",
    "NotARegularFileError",
    "UnreadableVideoError",
    "NoDecodableFramesError",
    "MissingTimestampError",
]

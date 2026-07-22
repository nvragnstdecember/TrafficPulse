"""Repository-safe synthetic-clip fixtures for the v1.1 U3 triple-riding e2e tests.

Generates a tiny real video at test time with PyAV (the no-download, no-network,
no-committed-binary pattern the sibling fixtures use) plus the matching scripted
``StubDetector`` -- a motorcycle carrying N riders whose boxes overlap it past the
association ``min_overlap``, drifting slowly so the real ``IouTracker`` holds one
id per entity.

Triple riding is pure geometry (rider counting), so unlike ``_helmet_fixtures``
there is no classifier: the reused perception + association layers count riders
directly. The committed ``example-scene.yaml`` works unmodified -- its
``triple_riding`` block (rider_count_threshold 3, min_persistence 1.0 s) is what
the slice reads. The detector-config/scene helpers are reused from
``_helmet_fixtures`` (both need the same motorbike/person label map and scene).

Uniquely named (``_triple_fixtures``) for pytest's prepend import mode.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from trafficpulse.detector import RawDetection, StubDetector

WIDTH = 320
HEIGHT = 240
FPS = 10  # PTS steps of 0.1 s -> the 1.0 s min_persistence is reached in ~10 frames
FRAME_COUNT = 30  # enough to pass min_persistence with margin

_STEP_X = 1.0  # slow drift: consecutive boxes overlap far above the tracker's IoU
_START_X = 30.0

# A wide motorcycle so several riders sit side by side astride it.
_BIKE_W, _BIKE_H = 120.0, 50.0
_BIKE_BOTTOM = 200.0

# Each rider: a tall box overlapping the bike's upper part. Three fit across the
# bike's width; each clears the association min_overlap (0.30 IoMin) comfortably.
_RIDER_W = 40.0
_RIDER_TOP = 90.0
_RIDER_BOTTOM = 180.0
_RIDER_OFFSETS = (5.0, 45.0, 80.0)  # x offsets within the bike for up to three riders
_RIDER_COLORS = ((220, 160, 40), (40, 220, 160), (160, 40, 220))


def _bike_box(frame_index: int) -> tuple[float, float, float, float]:
    x1 = _START_X + _STEP_X * frame_index
    return (x1, _BIKE_BOTTOM - _BIKE_H, x1 + _BIKE_W, _BIKE_BOTTOM)


def _rider_box(frame_index: int, rider: int) -> tuple[float, float, float, float]:
    x1 = _START_X + _STEP_X * frame_index + _RIDER_OFFSETS[rider]
    return (x1, _RIDER_TOP, x1 + _RIDER_W, _RIDER_BOTTOM)


def write_triple_riding_clip(path: Path, *, riders: int = 3, frames: int = FRAME_COUNT) -> Path:
    """Encode a tiny mp4 of a motorcycle carrying ``riders`` riders. Returns ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for index in range(frames):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bx1, by1, bx2, by2 = (int(round(v)) for v in _bike_box(index))
        image[by1:by2, bx1:bx2] = (40, 60, 220)  # bike
        for rider in range(riders):
            rx1, ry1, rx2, ry2 = (int(round(v)) for v in _rider_box(index, rider))
            image[ry1:ry2, rx1:rx2] = _RIDER_COLORS[rider]
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def scripted_rider_count_detector(riders: int = 3, frames: int = FRAME_COUNT) -> StubDetector:
    """A ``StubDetector`` emitting one ``motorbike`` + ``riders`` ``person`` boxes per frame.

    Keyed by ``frame.frame_index`` and matched to the rendered rectangles. The
    native ``"motorbike"`` label mirrors what the real RT-DETR checkpoint emits, so
    the fixture exercises the same label map the real path uses.
    """

    per_frame = {
        i: (
            RawDetection(label="motorbike", score=0.9, box=_bike_box(i)),
            *(RawDetection(label="person", score=0.9, box=_rider_box(i, r)) for r in range(riders)),
        )
        for i in range(frames)
    }
    return StubDetector(per_frame=per_frame)

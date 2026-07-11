"""Repository-safe synthetic-clip fixtures for the P2-U6 illegal-stopping e2e tests.

Generates a tiny real video **at test time** with PyAV (the same no-download,
no-network, no-committed-binary pattern the P1-U5 ingestion tests and the P1-U12
``_slice_fixtures`` use) plus the matching scripted ``StubDetector`` and a
purpose-built **test** ``SceneConfig``.

Why a purpose-built test scene
------------------------------
Unlike wrong-way (which reasons on a *global* legal-direction vector, so the P1-U12
clip works against the unmodified example scene), illegal stopping is gated on
**zone membership**: the vehicle's bbox bottom-center must fall inside a
``no_stopping`` polygon. The example scene's ``zone-no-stop`` lives at 1920x1080
pixel coordinates a tiny 320x240 clip cannot reach, so this module derives a test
scene from the example scene by patching **only** the ``zone-no-stop`` polygon into
the clip's pixel space and lowering ``stationary_duration`` (to keep the clip
short). Every point stays within the example scene's declared 1920x1080 frame, so
the ``SceneConfig`` bounds validator still passes. This is a **test fixture only**;
the example scene's ``zone-no-stop`` remains the analogue for the real/demo path.

Motion shape (``enter-then-stop``)
----------------------------------
:func:`write_illegal_stopping_clip` renders a rectangle that **moves down into**
the no-stopping zone over a few frames, then **holds position** inside it for the
rest of the clip -- so the real ``IouTracker`` associates one moving-then-stopped
track, the P2-U3 stationarity window flips to stationary once the trailing window
fills with held samples, and the P2-U4 reasoner accumulates dwell to exactly one
confirmed ``ILLEGAL_STOPPING`` event. :func:`write_drive_through_clip` renders a
rectangle that *keeps moving* through the zone (never stationary) for the no-event
case. Neither fixture tricks a real detector: the paired ``StubDetector`` replays a
caller-authored script matched to the rendered rectangle (a COCO RT-DETR does not
fire the vehicle class on these synthetic pixels).

Uniquely named (``_stopping_fixtures`` -- not ``_slice_fixtures`` or
``_pipeline_helpers``) so pytest's prepend import mode never collides across the
tests tree.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import yaml

from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector

# --- clip geometry -----------------------------------------------------------
WIDTH = 320
HEIGHT = 240
FPS = 10  # PTS steps of 0.1 s -> a 2.0 s dwell is reached in ~20 held frames

_BOX_W = 40.0
_BOX_H = 30.0

# Held (stopped) position: bottom-center (160, 200) sits inside NO_STOP_POLYGON.
_HOLD_X1 = 140.0
_HOLD_X2 = _HOLD_X1 + _BOX_W  # 180
_HOLD_BOTTOM = 200.0

# Entry: the box descends _STEP_DOWN px/frame for _ENTER_FRAMES frames, then holds.
# _STEP_DOWN (8) vs _BOX_H (30) keeps consecutive boxes overlapping (IoU ~0.58 >
# the 0.3 default) so the real IouTracker holds a single id across the motion.
_ENTER_FRAMES = 6
_STEP_DOWN = 8.0

# Total frames: dwell opens once the stationarity window fills (~frame 10 at
# STATIONARY_WINDOW=5) and confirms 2.0 s later (~frame 30); 40 leaves margin.
FRAME_COUNT = 40

# Drive-through (no-event) motion: a box that keeps moving down through the zone.
_DRIVE_FRAMES = 18
_DRIVE_BOTTOM0 = 140.0
_DRIVE_STEP = 6.0

# No-stopping zone polygon in the clip's pixel space (a trapezoid spanning
# x in ~[100, 220], y in [120, 220]); every point is within the example scene's
# 1920x1080 declared frame, so the SceneConfig bounds validator passes.
NO_STOP_POLYGON = [[100, 220], [220, 220], [210, 120], [110, 120]]

# Test-scene dwell threshold (seconds): small so the clip stays short. Provisional,
# like the example scene's 10.0 s -- this is a test fixture, not a tuned value.
STATIONARY_DURATION_S = 2.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SCENE_PATH = _REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"


def _stop_box(frame_index: int) -> tuple[float, float, float, float]:
    """Box for the enter-then-stop motion: descend into the zone, then hold."""

    if frame_index >= _ENTER_FRAMES:
        bottom = _HOLD_BOTTOM
    else:
        bottom = _HOLD_BOTTOM - (_ENTER_FRAMES - frame_index) * _STEP_DOWN
    return (_HOLD_X1, bottom - _BOX_H, _HOLD_X2, bottom)


def _drive_box(frame_index: int) -> tuple[float, float, float, float]:
    """Box for the drive-through motion: keep moving down through the zone."""

    bottom = _DRIVE_BOTTOM0 + _DRIVE_STEP * frame_index
    return (_HOLD_X1, bottom - _BOX_H, _HOLD_X2, bottom)


def _write_clip(
    path: Path,
    boxes: list[tuple[float, float, float, float]],
    *,
    fps: int = FPS,
) -> Path:
    """Encode a tiny mpeg4/mp4 clip of a red rectangle at ``boxes[i]`` on frame ``i``.

    Deterministic pixels; portable in PyAV's bundled FFmpeg (the P1-U5 test codec).
    """

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for box in boxes:
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        image[y1:y2, x1:x2] = (220, 40, 40)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def write_illegal_stopping_clip(path: Path, *, frames: int = FRAME_COUNT) -> Path:
    """Write the enter-then-stop clip (one rectangle stopping in the zone). Returns ``path``."""

    return _write_clip(path, [_stop_box(i) for i in range(frames)])


def write_drive_through_clip(path: Path, *, frames: int = _DRIVE_FRAMES) -> Path:
    """Write the drive-through clip (rectangle moving through, never stopping)."""

    return _write_clip(path, [_drive_box(i) for i in range(frames)])


def _scripted_detector(
    boxes: list[tuple[float, float, float, float]],
) -> StubDetector:
    """A ``StubDetector`` scripted to emit one ``car`` at ``boxes[i]`` on frame ``i``.

    Keyed by ``frame.frame_index`` so it aligns with the decoded frame order. It
    replays a known script; it performs no inference and never reads the pixels.
    """

    per_frame = {
        i: (RawDetection(label="car", score=0.9, box=box),) for i, box in enumerate(boxes)
    }
    return StubDetector(per_frame=per_frame)


def scripted_stopping_detector(frames: int = FRAME_COUNT) -> StubDetector:
    """Scripted detector matched to :func:`write_illegal_stopping_clip`."""

    return _scripted_detector([_stop_box(i) for i in range(frames)])


def scripted_drive_through_detector(frames: int = _DRIVE_FRAMES) -> StubDetector:
    """Scripted detector matched to :func:`write_drive_through_clip`."""

    return _scripted_detector([_drive_box(i) for i in range(frames)])


def stopping_detector_config() -> DetectorConfig:
    """Detector config mapping the scripted ``car`` label to ``ObjectClass.CAR``."""

    return DetectorConfig(label_map={"car": ObjectClass.CAR})


def illegal_stopping_test_scene() -> SceneConfig:
    """A test ``SceneConfig``: the example scene with the no-stop zone in clip space.

    Patches **only** the ``zone-no-stop`` polygon (into the clip's pixel space) and
    the ``stationary_duration`` value (to :data:`STATIONARY_DURATION_S`); everything
    else -- frame reference, other zones, calibration -- is the example scene
    verbatim, so the change is minimal and the scene stays valid.
    """

    raw = yaml.safe_load(_EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    scene = SceneConfig.model_validate(raw).model_dump(mode="json")
    for zone in scene["zones"]:
        if zone["zone_id"] == "zone-no-stop":
            zone["polygon"] = [list(pt) for pt in NO_STOP_POLYGON]
    for block in scene["rule_parameters"]:
        if block["violation_type"] == "illegal_stopping":
            for param in block["parameters"]:
                if param["id"] == "stationary_duration":
                    param["value"] = STATIONARY_DURATION_S
    return SceneConfig.model_validate(scene)

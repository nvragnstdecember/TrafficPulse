"""Repository-safe synthetic-clip fixtures for the P4-U6 no-helmet e2e tests.

Generates a tiny real video **at test time** with PyAV (the same no-download,
no-network, no-committed-binary pattern ``_slice_fixtures`` and
``_stopping_fixtures`` use) plus the matching scripted ``StubDetector`` and
scripted ``StubHelmetClassifier``.

Why no test scene is needed
---------------------------
Unlike illegal stopping (gated on zone membership, so ``_stopping_fixtures`` must
patch a polygon into the clip's pixel space), no-helmet reasoning is **not
zone-gated**: it reasons over rider observations and time only. The committed
``configs/scenes/example-scene.yaml`` therefore works unmodified -- its
``no_helmet`` block (``min_persistence`` 1.0 s, ``max_observation_gap`` 2.0 s) is
exactly what the slice reads. This mirrors wrong-way, which also runs against the
unmodified example scene.

Motion shape (``ride-through``)
-------------------------------
:func:`write_no_helmet_clip` renders a motorcycle rectangle with a rider rectangle
above it, drifting slowly across frame -- slowly enough that consecutive boxes
overlap well past the tracker's 0.3 IoU default, so the real ``IouTracker`` holds
one bike id and one rider id for the whole clip. The rider box overlaps the bike
box enough to clear the association ``min_overlap``, so the real P4-U4 association
derivation links them.

Nothing here tricks a real detector: the paired ``StubDetector`` replays a
caller-authored script matched to the rendered rectangles (a COCO RT-DETR does not
fire on these synthetic pixels), and the paired ``StubHelmetClassifier`` replays a
caller-authored helmet label (no model can read a helmet off a coloured rectangle).
The real detector/classifier path is exercised on real footage instead -- see
``demo/gate0_rtdetr_validation.py`` (P4-U1) and the opt-in
``tests/pipeline/test_no_helmet_e2e_real.py``.

Uniquely named (``_helmet_fixtures``) so pytest's prepend import mode never
collides across the tests tree.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import yaml

from trafficpulse.classifier import RawHelmetPrediction, StubHelmetClassifier
from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector

# --- clip geometry -----------------------------------------------------------
WIDTH = 320
HEIGHT = 240
FPS = 10  # PTS steps of 0.1 s -> the 1.0 s min_persistence is reached in ~10 frames

# Enough frames to pass min_persistence (1.0 s) with margin.
FRAME_COUNT = 30

# The bike drifts right this many px/frame. Small vs the box width so consecutive
# boxes overlap far above the tracker's 0.3 IoU default and one id is held.
_STEP_X = 1.5
_START_X = 40.0

_BIKE_W, _BIKE_H = 60.0, 50.0
_BIKE_BOTTOM = 200.0

# The rider sits astride the bike: horizontally inset, extending well above it and
# overlapping its upper part. The overlap clears the association min_overlap (0.30
# IoMin) comfortably; the extension above is the region the head crop is cut from.
_RIDER_INSET = 10.0
_RIDER_W = _BIKE_W - 2 * _RIDER_INSET
_RIDER_TOP = 60.0
_RIDER_BOTTOM = 180.0

# The scripted helmet labels. Named constants (not inline literals) so a test reads
# as a scenario rather than a magic string.
NO_HELMET = RawHelmetPrediction(label="no_helmet", score=0.88)
HELMET = RawHelmetPrediction(label="helmet", score=0.91)
TURBAN = RawHelmetPrediction(label="turban", score=0.79)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SCENE_PATH = _REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"


def _bike_box(frame_index: int) -> tuple[float, float, float, float]:
    x1 = _START_X + _STEP_X * frame_index
    return (x1, _BIKE_BOTTOM - _BIKE_H, x1 + _BIKE_W, _BIKE_BOTTOM)


def _rider_box(frame_index: int) -> tuple[float, float, float, float]:
    x1 = _START_X + _STEP_X * frame_index + _RIDER_INSET
    return (x1, _RIDER_TOP, x1 + _RIDER_W, _RIDER_BOTTOM)


def write_no_helmet_clip(path: Path, *, frames: int = FRAME_COUNT) -> Path:
    """Encode a tiny mpeg4/mp4 clip of a rider astride a motorcycle. Returns ``path``.

    Deterministic pixels; portable in PyAV's bundled FFmpeg (the P1-U5 test codec).
    The rider is drawn on top of the bike so the two rectangles are visually
    distinguishable in a viewer preview.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for index in range(frames):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bx1, by1, bx2, by2 = (int(round(v)) for v in _bike_box(index))
        image[by1:by2, bx1:bx2] = (40, 60, 220)  # bike
        rx1, ry1, rx2, ry2 = (int(round(v)) for v in _rider_box(index))
        image[ry1:ry2, rx1:rx2] = (220, 160, 40)  # rider
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def scripted_rider_detector(frames: int = FRAME_COUNT) -> StubDetector:
    """A ``StubDetector`` emitting one ``motorbike`` + one ``person`` per frame.

    Keyed by ``frame.frame_index`` so it aligns with the decoded frame order, and
    matched to the rectangles :func:`write_no_helmet_clip` renders. The native
    label is ``"motorbike"`` deliberately -- that is the spelling P4-U1 found the
    real RT-DETR checkpoint emits, so the fixture exercises the same label map the
    real path uses.
    """

    per_frame = {
        i: (
            RawDetection(label="motorbike", score=0.9, box=_bike_box(i)),
            RawDetection(label="person", score=0.9, box=_rider_box(i)),
        )
        for i in range(frames)
    }
    return StubDetector(per_frame=per_frame)


def helmet_detector_config() -> DetectorConfig:
    """Detector config mapping the scripted labels to their frozen ``ObjectClass``."""

    return DetectorConfig(
        label_map={
            "motorbike": ObjectClass.MOTORCYCLE,
            "motorcycle": ObjectClass.MOTORCYCLE,
            "person": ObjectClass.PERSON,
        }
    )


def scripted_helmet_classifier(
    prediction: RawHelmetPrediction = NO_HELMET,
) -> StubHelmetClassifier:
    """A ``StubHelmetClassifier`` replaying one scripted label for every crop.

    The default is used for every crop because the rider's track id is assigned by
    the real tracker at run time and is not known to the fixture. This replays a
    caller-authored script; it performs no inference and never reads the pixels.
    """

    return StubHelmetClassifier(prediction)


def helmet_example_scene() -> SceneConfig:
    """The committed example scene, unmodified (no-helmet needs no zone patching)."""

    return SceneConfig.model_validate(
        yaml.safe_load(_EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    )

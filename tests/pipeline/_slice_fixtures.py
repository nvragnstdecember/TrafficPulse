"""Repository-safe synthetic-clip fixtures for the P1-U12 slice tests.

Generates a tiny real video **at test time** with PyAV (the same no-download,
no-network, no-committed-binary pattern the P1-U5 ingestion tests use) plus the
matching scripted ``StubDetector``. The clip shows one bright rectangle moving
**down** (+y in image space), which contradicts the example scene's legal
``dir-north`` (0, -1) -- so the full slice confirms exactly one wrong-way event.

Uniquely named (``_slice_fixtures`` -- not ``_pipeline_helpers`` or ``_builders``)
so pytest's prepend import mode never collides across the tests tree.

Two honesty levels this module supports:

* the **real-ingestion + stub-detector** level (default suite): a real encoded clip
  is decoded through P1-U5, and :func:`scripted_down_detector` replays detections
  matched to the rectangle's known motion so the real ``IouTracker`` + real rules
  produce the event -- because a COCO RT-DETR does not fire the *vehicle* class on
  these synthetic pixels;
* the **real RT-DETR** level (opt-in test) reuses the *same* generated clip but
  injects the real backend instead, to prove real inference integrates end to end.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import yaml

from trafficpulse.contracts import SceneConfig
from trafficpulse.detector import RawDetection, StubDetector

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SCENE_PATH = _REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"

# Clip geometry. FPS * FRAME_COUNT well exceeds the example scene's 1.0 s wrong_way
# min_persistence (10 fps * 30 frames = 3.0 s of media time).
WIDTH = 320
HEIGHT = 240
FPS = 10
FRAME_COUNT = 30

# The moving rectangle: fixed x-band, top edge stepping DOWN every frame.
_BOX_X1 = 140.0
_BOX_X2 = 180.0
_BOX_H = 30.0
_Y0 = 20.0
_STEP = 6.0


def _box_top(frame_index: int) -> float:
    return _Y0 + frame_index * _STEP


# The lane the rectangle drives down, in the clip's own 320x240 pixel space.
# Wrong-way reasoning is contained to the polygon of the lane its legal direction
# governs, so the clip's rectangle has to actually be *in* that lane -- and the
# example scene's ``zone-lane-north`` sits at 1920x1080 coordinates a 320x240 clip
# cannot reach. Chosen so the box center (160, 35..209) keeps >= 25 px of clearance
# from every edge for the whole run, comfortably outside the boundary-abstain band.
LANE_POLYGON: tuple[tuple[float, float], ...] = (
    (100.0, 0.0),
    (220.0, 0.0),
    (220.0, 260.0),
    (100.0, 260.0),
)


def wrong_way_test_scene() -> SceneConfig:
    """The example scene with ``zone-lane-north`` patched into the clip's pixel space.

    The wrong-way analogue of ``_stopping_fixtures.illegal_stopping_test_scene``,
    and it exists for the same reason: the rule is geometry-gated, and the example
    scene's geometry is authored for 1920x1080 footage rather than a tiny synthetic
    clip. Only that one polygon changes -- frame reference, other zones, directions,
    calibration and every rule parameter stay the example scene verbatim, so the
    scene stays valid and the change is auditable.

    Note the polygon's points remain inside the example scene's declared 1920x1080
    frame, so ``SceneConfig``'s bounds validator still passes.

    This is a **test fixture only**: the committed example scene's
    ``zone-lane-north`` is still the analogue for the real/demo path.
    """

    raw = yaml.safe_load(_EXAMPLE_SCENE_PATH.read_text(encoding="utf-8"))
    scene = SceneConfig.model_validate(raw).model_dump(mode="json")
    for zone in scene["zones"]:
        if zone["zone_id"] == "zone-lane-north":
            zone["polygon"] = [list(pt) for pt in LANE_POLYGON]
    return SceneConfig.model_validate(scene)


def write_wrong_way_scene_file(path: Path) -> Path:
    """Dump :func:`wrong_way_test_scene` to YAML for callers that need a scene *path*.

    The application layer is configured with a scene **file**, not a
    ``SceneConfig``, so its tests need the clip-space scene on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(wrong_way_test_scene().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_wrong_way_clip(
    path: Path, *, frames: int = FRAME_COUNT, fps: int = FPS
) -> Path:
    """Write a tiny mpeg4/mp4 clip of a red rectangle moving down. Returns ``path``.

    Deterministic pixels; portable in PyAV's bundled FFmpeg (the P1-U5 test codec).
    """

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for i in range(frames):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        top = int(_box_top(i))
        image[top : top + int(_BOX_H), int(_BOX_X1) : int(_BOX_X2)] = (220, 40, 40)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def scripted_down_detector(frames: int = FRAME_COUNT) -> StubDetector:
    """A ``StubDetector`` scripted to the clip's known rectangle motion (per frame).

    Emits one ``car`` detection per frame whose box matches the rendered rectangle,
    keyed by ``frame.frame_index`` -- so it aligns with the decoded frame order. It
    replays a known script; it performs no inference and never reads the pixels.
    """

    per_frame = {
        i: (
            RawDetection(
                label="car",
                score=0.9,
                box=(_BOX_X1, _box_top(i), _BOX_X2, _box_top(i) + _BOX_H),
            ),
        )
        for i in range(frames)
    }
    return StubDetector(per_frame=per_frame)

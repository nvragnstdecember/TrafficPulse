"""Shared, model-free builders for the pipeline orchestration tests (P1-U10).

Deterministic construction of the inputs the ``WrongWayPipeline`` composes:
synthetic ``FrameRecord``s, a scripted ``StubDetector`` emitting ``RawDetection``s,
and the example ``SceneConfig``. Kept in a **uniquely-named** helper module
(``_pipeline_helpers`` -- never a second ``_builders``) so pytest's prepend import
mode does not collide with the tracking tests' ``_builders``.

Timestamps are anchored at the pipeline's fixed media-time epoch and ``camera_id``
is the example scene's camera, so ``Detection``s built directly here for the
equivalence reference carry the *same* identity as the ones the pipeline adapts
from frames -- making observation/event ids line up exactly.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from _slice_fixtures import wrong_way_test_scene  # noqa: E402  (tests-tree sibling)

from trafficpulse.contracts import (
    BoundingBox,
    Detection,
    ObjectClass,
    SceneConfig,
)
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector
from trafficpulse.ingestion.video import FrameRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
#: The scene these fixtures reason against: the example scene with its northbound
#: lane patched into the synthetic clip's pixel space. Wrong-way reasoning is
#: contained to that lane's polygon, and the example scene's own ``zone-lane-north``
#: is authored for 1920x1080 footage that no fixture here produces -- so reasoning
#: against it unpatched would (correctly) confirm nothing at all. Every other field
#: is the example scene verbatim; see ``_slice_fixtures.wrong_way_test_scene``.
SCENE: SceneConfig = wrong_way_test_scene()
CAMERA = SCENE.scene.camera_id  # "cam-synthetic-01"
NORTH_DIRECTION_ID = "dir-north"  # legal direction "north" = (0, -1); moving down is wrong-way

# The pipeline anchors media-relative PTS at this fixed UTC epoch; direct-reference
# Detections use the same base so their timestamps (and derived ids) match.
BASE = datetime(1970, 1, 1, tzinfo=UTC)
FRAME_INTERVAL_S = 1.0 / 30.0

# Enough frames to exceed the example scene's 1.0 s wrong_way min_persistence.
DEFAULT_FRAME_COUNT = 45
STEP_PX = 5.0  # small step: consecutive boxes overlap (IoU ~0.6 > 0.3 default)

# Wrong-way reasoning is contained to the polygon of the lane whose legal direction
# governs it, so these synthetic boxes must travel *inside* that lane -- the same
# requirement ``synth.scenarios`` already states for its own trajectories. ``SCENE``
# below is the clip-space wrong-way test scene, so the lane is
# ``_slice_fixtures.LANE_POLYGON`` (x 100..220, y 0..260) and these defaults share
# it with the real clip's rectangle. Every default keeps the 20x20 box's center
# >= 18 px clear of the lane's edges for the whole 45-frame run, well outside the
# boundary-abstain band. Boxes outside the lane produce no heading facts at all,
# which is the point: a track on another carriageway is not this lane's traffic.
# (Before containment these sat at x=50, y=50 -- outside any declared lane.)
LANE_X = 150.0  # box x1; center x = 160, the lane's mid-line
LANE_X_LEFT = 120.0  # a second, distinguishable in-lane column (center 130)
LANE_X_RIGHT = 180.0  # ...and a third (center 190)
LANE_Y0_DOWN = 8.0  # start high, step DOWN the lane = against ``dir-north``
LANE_Y0_UP = 228.0  # start low, step UP the lane = with ``dir-north``

DETECTOR_CONFIG = DetectorConfig(label_map={"car": ObjectClass.CAR})

_PIXEL = np.zeros((1, 1, 3), dtype=np.uint8)  # opaque, never read by the stub/adapter


def make_frame_record(
    frame_index: int,
    *,
    camera_id: str | None = CAMERA,
    timestamp_seconds: float | None = None,
) -> FrameRecord:
    """Build one synthetic ``FrameRecord`` (PTS = ``frame_index / 30`` by default)."""

    ts = frame_index * FRAME_INTERVAL_S if timestamp_seconds is None else timestamp_seconds
    return FrameRecord(
        source_id="vsrc-test",
        camera_id=camera_id,
        frame_id=f"vfrm-{frame_index}",
        frame_index=frame_index,
        timestamp_seconds=ts,
        width=1,
        height=1,
        image=_PIXEL,
    )


def _box(
    frame_index: int, *, x: float, step: float, direction: int, y0: float
) -> tuple[float, float, float, float]:
    top = y0 + frame_index * step * direction
    return (x, top, x + 20.0, top + 20.0)


def moving_raw(
    frame_index: int,
    *,
    x: float = LANE_X,
    step: float = STEP_PX,
    direction: int = 1,
    y0: float = LANE_Y0_DOWN,
) -> RawDetection:
    """A ``RawDetection`` for a car moving ``direction`` * ``step`` px/frame in y.

    ``direction=+1`` moves **down** (wrong-way vs legal north); ``-1`` moves up
    (legal). ``y0`` is the starting top -- use :data:`LANE_Y0_UP` for upward
    motion so the box stays inside the lane. ``label='car'`` maps to
    ``ObjectClass.CAR``.
    """

    return RawDetection(
        label="car", score=0.9, box=_box(frame_index, x=x, step=step, direction=direction, y0=y0)
    )


def moving_detection(
    frame_index: int,
    *,
    x: float = LANE_X,
    step: float = STEP_PX,
    direction: int = 1,
    y0: float = LANE_Y0_DOWN,
    camera_id: str = CAMERA,
) -> Detection:
    """A ``Detection`` equivalent to what the pipeline adapts from ``moving_raw``."""

    x1, y1, x2, y2 = _box(frame_index, x=x, step=step, direction=direction, y0=y0)
    return Detection(
        detection_id=f"det-{camera_id}-{frame_index}-{int(x)}",
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp=BASE + timedelta(seconds=frame_index * FRAME_INTERVAL_S),
        object_class=ObjectClass.CAR,
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def moving_down_detector(
    frame_count: int = DEFAULT_FRAME_COUNT,
    *,
    x: float = LANE_X,
    direction: int = 1,
    y0: float = LANE_Y0_DOWN,
) -> StubDetector:
    """A ``StubDetector`` scripted to emit one moving car per frame."""

    per_frame = {
        i: (moving_raw(i, x=x, direction=direction, y0=y0),) for i in range(frame_count)
    }
    return StubDetector(per_frame=per_frame)

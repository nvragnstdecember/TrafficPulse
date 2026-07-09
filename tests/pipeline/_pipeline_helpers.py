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
import yaml

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
SCENE: SceneConfig = SceneConfig.model_validate(
    yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
)
CAMERA = SCENE.scene.camera_id  # "cam-synthetic-01"
NORTH_DIRECTION_ID = "dir-north"  # legal direction "north" = (0, -1); moving down is wrong-way

# The pipeline anchors media-relative PTS at this fixed UTC epoch; direct-reference
# Detections use the same base so their timestamps (and derived ids) match.
BASE = datetime(1970, 1, 1, tzinfo=UTC)
FRAME_INTERVAL_S = 1.0 / 30.0

# Enough frames to exceed the example scene's 1.0 s wrong_way min_persistence.
DEFAULT_FRAME_COUNT = 45
STEP_PX = 5.0  # small step: consecutive boxes overlap (IoU ~0.6 > 0.3 default)

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
    x: float = 50.0,
    step: float = STEP_PX,
    direction: int = 1,
    y0: float = 50.0,
) -> RawDetection:
    """A ``RawDetection`` for a car moving ``direction`` * ``step`` px/frame in y.

    ``direction=+1`` moves **down** (wrong-way vs legal north); ``-1`` moves up
    (legal). ``y0`` is the starting top (raise it for upward motion so the box
    stays in frame). ``label='car'`` maps to ``ObjectClass.CAR``.
    """

    return RawDetection(
        label="car", score=0.9, box=_box(frame_index, x=x, step=step, direction=direction, y0=y0)
    )


def moving_detection(
    frame_index: int,
    *,
    x: float = 50.0,
    step: float = STEP_PX,
    direction: int = 1,
    y0: float = 50.0,
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
    frame_count: int = DEFAULT_FRAME_COUNT, *, x: float = 50.0, direction: int = 1, y0: float = 50.0
) -> StubDetector:
    """A ``StubDetector`` scripted to emit one moving car per frame."""

    per_frame = {
        i: (moving_raw(i, x=x, direction=direction, y0=y0),) for i in range(frame_count)
    }
    return StubDetector(per_frame=per_frame)

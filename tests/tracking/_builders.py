"""Shared detection builders for the tracking tests (P1-U8).

Deterministic, model-free ``Detection`` construction. Kept in a plain helper
module (imported by sibling test files under pytest's prepend import mode) so the
frozen-contract detection shape is built one way across the suite.
"""

from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import BoundingBox, Detection, ObjectClass

BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
FRAME_INTERVAL_S = 1.0 / 30.0


def make_detection(
    frame_index: int,
    ordinal: int = 0,
    *,
    camera_id: str = "cam1",
    object_class: ObjectClass = ObjectClass.CAR,
    confidence: float = 0.9,
    box: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
    timestamp: datetime | None = None,
) -> Detection:
    """Build one deterministic ``Detection`` at ``frame_index`` (unique per ordinal)."""

    ts = timestamp
    if ts is None:
        ts = BASE + timedelta(seconds=frame_index * FRAME_INTERVAL_S)
    x1, y1, x2, y2 = box
    return Detection(
        detection_id=f"det-{frame_index}-{ordinal}",
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp=ts,
        object_class=object_class,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )

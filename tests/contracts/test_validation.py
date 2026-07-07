"""Validation-failure tests for the U2 contracts."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from trafficpulse.contracts import (
    BoundingBox,
    Detection,
    ObjectClass,
    TimeInterval,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
NAIVE_TS = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo
BBOX = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)


def _detection(**overrides: Any) -> Detection:
    kwargs: dict[str, Any] = {
        "detection_id": "d1",
        "camera_id": "cam1",
        "frame_index": 0,
        "timestamp": TS,
        "object_class": ObjectClass.CAR,
        "confidence": 0.5,
        "bbox": BBOX,
    }
    kwargs.update(overrides)
    return Detection(**kwargs)


def test_confidence_below_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        _detection(confidence=-0.1)


def test_confidence_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _detection(confidence=1.1)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        _detection(timestamp=NAIVE_TS)


def test_bbox_zero_width_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=10.0, y1=0.0, x2=10.0, y2=20.0)


def test_bbox_inverted_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=10.0, y1=0.0, x2=5.0, y2=20.0)


def test_bbox_negative_coordinate_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=-1.0, y1=0.0, x2=10.0, y2=20.0)


def test_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _detection(detection_id="")


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _detection(unexpected="nope")


def test_time_interval_end_before_start_rejected() -> None:
    earlier = datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        TimeInterval(start=TS, end=earlier)

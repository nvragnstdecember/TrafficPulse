"""Adapter conversion, filtering, validation, and error-handling tests (P1-U6).

Exercises the ``DetectionAdapter`` seam: raw detector output -> frozen U2
``Detection``. Covers happy-path conversion and contract compatibility, label
mapping (including dropping unmodeled classes), confidence gating, malformed-
output rejection (score and box), frame-identity validation, ordinal-stable
identity under filtering, and provenance stamping.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from trafficpulse.contracts import Detection, ModelRef, ObjectClass
from trafficpulse.detector import (
    DetectionAdapter,
    DetectorConfig,
    Frame,
    InvalidFrameError,
    MalformedDetectorOutputError,
    RawDetection,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
NAIVE_TS = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo

LABEL_MAP = {
    "motorcycle": ObjectClass.MOTORCYCLE,
    "car": ObjectClass.CAR,
    "person": ObjectClass.PERSON,
}


def _config(**overrides: Any) -> DetectorConfig:
    kwargs: dict[str, Any] = {"label_map": dict(LABEL_MAP)}
    kwargs.update(overrides)
    return DetectorConfig(**kwargs)


def _frame(index: int = 0, camera_id: str = "cam1", timestamp: datetime = TS) -> Frame:
    return Frame(camera_id=camera_id, frame_index=index, timestamp=timestamp)


# --- happy path & contract compatibility ------------------------------------
def test_adapts_single_detection_to_frozen_contract() -> None:
    adapter = DetectionAdapter(_config())
    (det,) = adapter.adapt(_frame(), [RawDetection("car", 0.9, (10.0, 20.0, 30.0, 40.0))])
    assert isinstance(det, Detection)
    assert det.object_class is ObjectClass.CAR
    assert det.confidence == 0.9
    assert (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2) == (10.0, 20.0, 30.0, 40.0)
    assert det.camera_id == "cam1"
    assert det.frame_index == 0
    assert det.timestamp == TS
    assert det.source_model is None


def test_produced_detection_is_frozen_and_serializable() -> None:
    adapter = DetectionAdapter(_config())
    (det,) = adapter.adapt(_frame(), [RawDetection("car", 0.5, (1.0, 1.0, 2.0, 2.0))])
    # Frozen U2 contract: round-trips through JSON and rejects mutation.
    assert Detection.model_validate_json(det.model_dump_json()) == det
    with pytest.raises(ValidationError):
        det.confidence = 0.1  # type: ignore[misc]


def test_preserves_order_and_count() -> None:
    adapter = DetectionAdapter(_config())
    raws = [
        RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),
        RawDetection("motorcycle", 0.8, (3.0, 3.0, 4.0, 4.0)),
        RawDetection("person", 0.7, (5.0, 5.0, 6.0, 6.0)),
    ]
    dets = adapter.adapt(_frame(), raws)
    assert [d.object_class for d in dets] == [
        ObjectClass.CAR,
        ObjectClass.MOTORCYCLE,
        ObjectClass.PERSON,
    ]


def test_multiple_detections_get_distinct_ids() -> None:
    adapter = DetectionAdapter(_config())
    raws = [
        RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),
        RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),  # identical content
    ]
    dets = adapter.adapt(_frame(), raws)
    assert dets[0].detection_id != dets[1].detection_id  # ordinal disambiguates duplicates


def test_source_model_is_stamped_from_config() -> None:
    model = ModelRef(name="rt-detr", version="r50")
    adapter = DetectionAdapter(_config(source_model=model))
    (det,) = adapter.adapt(_frame(), [RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))])
    assert det.source_model == model


def test_empty_input_yields_empty_tuple() -> None:
    adapter = DetectionAdapter(_config())
    assert adapter.adapt(_frame(), []) == ()


# --- label mapping / filtering ----------------------------------------------
def test_unmodeled_label_is_dropped_not_error() -> None:
    adapter = DetectionAdapter(_config())
    raws = [
        RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),
        RawDetection("traffic_light", 0.9, (3.0, 3.0, 4.0, 4.0)),  # unmodeled -> dropped
    ]
    dets = adapter.adapt(_frame(), raws)
    assert len(dets) == 1
    assert dets[0].object_class is ObjectClass.CAR


def test_score_below_threshold_is_dropped() -> None:
    adapter = DetectionAdapter(_config(score_threshold=0.5))
    raws = [
        RawDetection("car", 0.49, (1.0, 1.0, 2.0, 2.0)),  # below -> dropped
        RawDetection("person", 0.5, (3.0, 3.0, 4.0, 4.0)),  # at threshold -> kept
    ]
    dets = adapter.adapt(_frame(), raws)
    assert len(dets) == 1
    assert dets[0].object_class is ObjectClass.PERSON


def test_default_threshold_keeps_all_valid_scores() -> None:
    adapter = DetectionAdapter(_config())
    (det,) = adapter.adapt(_frame(), [RawDetection("car", 0.0, (1.0, 1.0, 2.0, 2.0))])
    assert det.confidence == 0.0


def test_dropping_does_not_shift_kept_ids() -> None:
    """A kept detection's id depends on its emission ordinal, not the kept order."""

    adapter = DetectionAdapter(_config())
    frame = _frame()
    with_drop = adapter.adapt(
        frame,
        [
            RawDetection("bus", 0.9, (1.0, 1.0, 2.0, 2.0)),  # unmodeled, ordinal 0 -> dropped
            RawDetection("car", 0.9, (3.0, 3.0, 4.0, 4.0)),  # ordinal 1 -> kept
        ],
    )
    only_kept_at_same_ordinal = adapter.adapt(
        frame,
        [
            RawDetection("truck", 0.9, (9.0, 9.0, 9.5, 9.5)),  # unmodeled placeholder at 0
            RawDetection("car", 0.9, (3.0, 3.0, 4.0, 4.0)),  # ordinal 1 -> kept
        ],
    )
    assert with_drop[0].detection_id == only_kept_at_same_ordinal[0].detection_id


# --- malformed detector output ----------------------------------------------
@pytest.mark.parametrize("bad_score", [1.5, -0.1, float("nan"), float("inf"), float("-inf")])
def test_out_of_range_or_non_finite_score_rejected(bad_score: float) -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(MalformedDetectorOutputError):
        adapter.adapt(_frame(), [RawDetection("car", bad_score, (1.0, 1.0, 2.0, 2.0))])


@pytest.mark.parametrize(
    "bad_box",
    [
        (5.0, 1.0, 3.0, 4.0),  # x2 <= x1
        (1.0, 5.0, 3.0, 4.0),  # y2 <= y1
        (-1.0, 1.0, 3.0, 4.0),  # negative coordinate
        (1.0, 1.0, float("inf"), 4.0),  # non-finite coordinate
        (1.0, 1.0, float("nan"), 4.0),  # NaN coordinate
    ],
)
def test_malformed_box_rejected(bad_box: tuple[float, float, float, float]) -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(MalformedDetectorOutputError):
        adapter.adapt(_frame(), [RawDetection("car", 0.9, bad_box)])


def test_wrong_arity_box_rejected() -> None:
    adapter = DetectionAdapter(_config())
    bad = RawDetection("car", 0.9, (1.0, 2.0, 3.0))  # type: ignore[arg-type]
    with pytest.raises(MalformedDetectorOutputError):
        adapter.adapt(_frame(), [bad])


def test_malformed_output_rejects_batch_after_valid_ones() -> None:
    """A malformed output raises rather than being silently dropped mid-batch."""

    adapter = DetectionAdapter(_config())
    raws = [
        RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),  # valid
        RawDetection("person", 2.0, (3.0, 3.0, 4.0, 4.0)),  # malformed score
    ]
    with pytest.raises(MalformedDetectorOutputError):
        adapter.adapt(_frame(), raws)


def test_malformed_error_message_carries_context() -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(MalformedDetectorOutputError) as exc_info:
        adapter.adapt(_frame(index=7), [RawDetection("car", 3.0, (1.0, 1.0, 2.0, 2.0))])
    message = str(exc_info.value)
    assert "frame_index=7" in message
    assert "ordinal=0" in message
    assert "car" in message


# --- invalid frame identity -------------------------------------------------
def test_empty_camera_id_rejected() -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(InvalidFrameError):
        adapter.adapt(_frame(camera_id=""), [RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))])


def test_negative_frame_index_rejected() -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(InvalidFrameError):
        adapter.adapt(_frame(index=-1), [RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))])


def test_naive_timestamp_rejected() -> None:
    adapter = DetectionAdapter(_config())
    with pytest.raises(InvalidFrameError):
        adapter.adapt(_frame(timestamp=NAIVE_TS), [RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0))])


def test_frame_is_validated_before_detections() -> None:
    """An invalid frame is rejected even when detections are also malformed."""

    adapter = DetectionAdapter(_config())
    with pytest.raises(InvalidFrameError):
        adapter.adapt(_frame(camera_id=""), [RawDetection("car", 9.9, (1.0, 1.0, 2.0, 2.0))])


def test_config_is_exposed() -> None:
    cfg = _config()
    assert DetectionAdapter(cfg).config is cfg

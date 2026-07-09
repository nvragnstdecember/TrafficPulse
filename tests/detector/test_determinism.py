"""Determinism tests for the adapter conversion (P1-U6).

The reasoning layer's defensibility rests on deterministic, replayable data
(architecture-review §15). These tests pin the adapter's determinism guarantees:
identical inputs produce byte-identical ``Detection`` values (ids included),
identity is a pure function of frame identity + emission ordinal, and no
wall-clock or randomness leaks in.
"""

from datetime import UTC, datetime

from trafficpulse.contracts import ObjectClass
from trafficpulse.detector import DetectionAdapter, DetectorConfig, Frame, RawDetection

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
CONFIG = DetectorConfig(label_map={"car": ObjectClass.CAR, "person": ObjectClass.PERSON})
RAWS = [
    RawDetection("car", 0.9, (1.0, 1.0, 2.0, 2.0)),
    RawDetection("person", 0.8, (3.0, 3.0, 4.0, 4.0)),
]


def _frame(index: int = 0, camera_id: str = "cam1") -> Frame:
    return Frame(camera_id=camera_id, frame_index=index, timestamp=TS)


def test_repeated_adapt_is_byte_identical() -> None:
    adapter = DetectionAdapter(CONFIG)
    first = adapter.adapt(_frame(), RAWS)
    second = adapter.adapt(_frame(), RAWS)
    assert first == second
    assert [d.detection_id for d in first] == [d.detection_id for d in second]


def test_fresh_adapter_instances_agree() -> None:
    """Identity carries no per-instance state (no counters, no wall-clock)."""

    a = DetectionAdapter(CONFIG).adapt(_frame(), RAWS)
    b = DetectionAdapter(CONFIG).adapt(_frame(), RAWS)
    assert [d.detection_id for d in a] == [d.detection_id for d in b]


def test_detection_id_depends_on_frame_index() -> None:
    adapter = DetectionAdapter(CONFIG)
    (a,) = adapter.adapt(_frame(index=0), [RAWS[0]])
    (b,) = adapter.adapt(_frame(index=1), [RAWS[0]])
    assert a.detection_id != b.detection_id


def test_detection_id_depends_on_camera_id() -> None:
    adapter = DetectionAdapter(CONFIG)
    (a,) = adapter.adapt(_frame(camera_id="camA"), [RAWS[0]])
    (b,) = adapter.adapt(_frame(camera_id="camB"), [RAWS[0]])
    assert a.detection_id != b.detection_id


def test_detection_id_depends_on_ordinal() -> None:
    adapter = DetectionAdapter(CONFIG)
    dets = adapter.adapt(_frame(), RAWS)
    assert dets[0].detection_id != dets[1].detection_id


def test_detection_id_is_stable_string_form() -> None:
    adapter = DetectionAdapter(CONFIG)
    (det,) = adapter.adapt(_frame(), [RAWS[0]])
    # Deterministic "det-" + 16 lowercase hex chars from SHA-256 (a source label).
    assert det.detection_id.startswith("det-")
    body = det.detection_id.removeprefix("det-")
    assert len(body) == 16
    assert all(c in "0123456789abcdef" for c in body)

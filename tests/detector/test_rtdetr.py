"""RT-DETR backend tests (P1-U7): real backend, fake-engine driven.

These exercise the ``RTDetrDetector`` against a framework-neutral fake inference
engine, so the whole suite needs **no** network, **no** GPU, **no** checkpoint,
and **no** torch/transformers install. They pin the backend's boundary contract:
config validation, RGB-pixel handling, engine-output -> ``RawDetection``
conversion, native-label preservation, coordinate/scalar hygiene (no framework
object escapes), error translation, determinism, and end-to-end adaptation into
frozen U2 ``Detection`` contracts via the existing ``DetectionAdapter``.

The real transformers engine (``_TransformersRTDetrEngine``) is *not* exercised
here -- it requires the optional ``rtdetr`` deps and a local checkpoint. Its real
inference is covered separately and opt-in by ``test_rtdetr_smoke.py``.
"""

import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from trafficpulse.contracts import Detection, ModelRef, ObjectClass
from trafficpulse.detector import (
    DetectionAdapter,
    Detector,
    DetectorConfig,
    Frame,
    MalformedBackendOutputError,
    MissingFrameImageError,
    RTDetrBackendError,
    RTDetrConfig,
    RTDetrDetector,
)
from trafficpulse.detector.rtdetr import (
    BackendDependencyError,
    EngineDetection,
    RTDetrInferenceEngine,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# COCO-style native labels an RT-DETR checkpoint might emit (integer id -> string).
ID2LABEL = {0: "car", 1: "person", 2: "traffic_light"}


class FakeEngine:
    """A deterministic, framework-neutral ``RTDetrInferenceEngine`` for tests."""

    def __init__(
        self,
        detections: Sequence[EngineDetection],
        *,
        id2label: dict[int, str] | None = None,
    ) -> None:
        self._detections = tuple(detections)
        self._id2label = dict(ID2LABEL if id2label is None else id2label)
        self.seen_thresholds: list[float] = []
        self.seen_shapes: list[tuple[int, ...]] = []

    def label_name(self, label_id: int) -> str | None:
        return self._id2label.get(label_id)

    def infer(
        self, image: np.ndarray, *, threshold: float
    ) -> Sequence[EngineDetection]:
        self.seen_thresholds.append(threshold)
        self.seen_shapes.append(tuple(image.shape))
        return self._detections


def _image(height: int = 480, width: int = 640) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _frame(index: int = 0, *, with_image: bool = True) -> Frame:
    image = _image() if with_image else None
    return Frame(camera_id="cam1", frame_index=index, timestamp=TS, image=image)


def _detector(
    detections: Sequence[EngineDetection],
    *,
    id2label: dict[int, str] | None = None,
    **config_overrides: object,
) -> tuple[RTDetrDetector, FakeEngine]:
    engine = FakeEngine(detections, id2label=id2label)
    kwargs: dict[str, object] = {"checkpoint": "local/rtdetr-test"}
    kwargs.update(config_overrides)
    detector = RTDetrDetector(RTDetrConfig(**kwargs), engine=engine)  # type: ignore[arg-type]
    return detector, engine


# --- interface & seam --------------------------------------------------------
def test_rtdetr_detector_is_a_detector() -> None:
    detector, _ = _detector([])
    assert isinstance(detector, Detector)


def test_fake_engine_satisfies_the_protocol() -> None:
    # Structural typing sanity: the fake is a valid RTDetrInferenceEngine.
    engine: RTDetrInferenceEngine = FakeEngine([])
    assert engine.label_name(0) == "car"


def test_config_is_exposed() -> None:
    cfg = RTDetrConfig(checkpoint="local/rtdetr-test")
    assert RTDetrDetector(cfg, engine=FakeEngine([])).config is cfg


# --- import / construction do not pull ML or download ------------------------
def test_importing_rtdetr_module_pulls_in_no_ml_framework() -> None:
    # Hermetic check in a fresh interpreter: importing the backend module must pull
    # in NO ML framework (torch/transformers are lazy). A subprocess avoids mutating
    # this process's already-imported modules (an in-process importlib.reload would
    # rebind the module's classes and break other tests' exception identities).
    code = (
        "import sys, importlib;"
        "importlib.import_module('trafficpulse.detector.rtdetr');"
        "bad=sorted(set(sys.modules) & {'torch','torchvision','transformers','onnxruntime','cv2'});"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"import pulled in ML modules: {result.stdout.strip()!r}"


def test_constructing_config_imports_no_ml_framework() -> None:
    RTDetrConfig(checkpoint="local/rtdetr-test", device="cuda", threshold=0.7)
    forbidden = {"torch", "torchvision", "transformers", "onnxruntime"}
    assert not (set(sys.modules) & forbidden)


# --- configuration validation ------------------------------------------------
def test_config_defaults() -> None:
    cfg = RTDetrConfig(checkpoint="local/rtdetr-test")
    assert cfg.device == "cpu"
    assert cfg.local_files_only is True
    assert cfg.threshold == 0.5


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1"])
def test_valid_device_strings_accepted(device: str) -> None:
    assert RTDetrConfig(checkpoint="c", device=device).device == device


@pytest.mark.parametrize("device", ["gpu", "gpu:0", "gpu:", "CUDA", "cuda:x", "", "mps"])
def test_invalid_device_rejected(device: str) -> None:
    with pytest.raises(ValidationError):
        RTDetrConfig(checkpoint="c", device=device)


def test_empty_checkpoint_rejected() -> None:
    with pytest.raises(ValidationError):
        RTDetrConfig(checkpoint="")


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_threshold_out_of_range_rejected(threshold: float) -> None:
    with pytest.raises(ValidationError):
        RTDetrConfig(checkpoint="c", threshold=threshold)


def test_config_is_frozen() -> None:
    cfg = RTDetrConfig(checkpoint="c")
    with pytest.raises(ValidationError):
        cfg.threshold = 0.9  # type: ignore[misc]


def test_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RTDetrConfig(checkpoint="c", weights_path="model.pt")  # type: ignore[call-arg]


# --- engine output -> RawDetection conversion --------------------------------
def test_engine_detection_converts_to_raw() -> None:
    detector, _ = _detector([EngineDetection(label_id=0, score=0.9, box=(10.0, 20.0, 30.0, 40.0))])
    (raw,) = detector.detect(_frame())
    assert raw.label == "car"
    assert raw.score == 0.9
    assert raw.box == (10.0, 20.0, 30.0, 40.0)


def test_boxes_are_preserved_as_original_pixel_coordinates() -> None:
    # The engine already returns original-frame xyxy pixels; the detector must not
    # alter them, and they must survive into the frozen BoundingBox contract.
    box = (12.5, 34.0, 200.0, 400.0)
    detector, _ = _detector([EngineDetection(label_id=0, score=0.8, box=box)])
    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    (det,) = adapter.adapt_from(detector, _frame())
    assert (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2) == box


def test_scores_are_preserved() -> None:
    detector, _ = _detector(
        [
            EngineDetection(label_id=0, score=0.11, box=(1.0, 1.0, 2.0, 2.0)),
            EngineDetection(label_id=1, score=0.99, box=(3.0, 3.0, 4.0, 4.0)),
        ]
    )
    assert [r.score for r in detector.detect(_frame())] == [0.11, 0.99]


def test_native_labels_are_preserved_for_adapter_mapping() -> None:
    detector, _ = _detector(
        [
            EngineDetection(label_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0)),
            EngineDetection(label_id=1, score=0.9, box=(3.0, 3.0, 4.0, 4.0)),
            EngineDetection(label_id=2, score=0.9, box=(5.0, 5.0, 6.0, 6.0)),
        ]
    )
    assert [r.label for r in detector.detect(_frame())] == ["car", "person", "traffic_light"]


def test_multiple_detections_preserve_deterministic_order() -> None:
    detector, _ = _detector(
        [
            EngineDetection(label_id=1, score=0.7, box=(5.0, 5.0, 6.0, 6.0)),
            EngineDetection(label_id=0, score=0.8, box=(1.0, 1.0, 2.0, 2.0)),
        ]
    )
    labels = [r.label for r in detector.detect(_frame())]
    assert labels == ["person", "car"]


def test_empty_detections_return_empty_tuple() -> None:
    detector, _ = _detector([])
    result = detector.detect(_frame())
    assert result == ()
    assert isinstance(result, tuple)


def test_detect_returns_a_tuple() -> None:
    detector, _ = _detector([EngineDetection(label_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0))])
    assert isinstance(detector.detect(_frame()), tuple)


# --- box clipping to the frame (coordinate contract) -------------------------
def test_in_bounds_box_is_unchanged() -> None:
    # image is 480x640; this box is fully inside -> passes through unclipped.
    detector, _ = _detector([EngineDetection(label_id=0, score=0.9, box=(10.0, 20.0, 30.0, 40.0))])
    (raw,) = detector.detect(_frame())
    assert raw.box == (10.0, 20.0, 30.0, 40.0)


def test_marginally_out_of_bounds_box_is_clipped_to_frame() -> None:
    # RT-DETR can predict fractionally outside the frame; the backend clips to
    # [0,width] x [0,height] so the frozen BoundingBox contract accepts it.
    detector, _ = _detector(
        [EngineDetection(label_id=0, score=0.9, box=(-0.6, -3.0, 641.0, 481.0))]
    )
    (raw,) = detector.detect(_frame())  # frame image is (480, 640, 3)
    assert raw.box == (0.0, 0.0, 640.0, 480.0)


def test_box_fully_outside_frame_is_dropped() -> None:
    # A box entirely past the right/bottom edge has no in-frame area after clipping.
    detector, _ = _detector(
        [
            EngineDetection(label_id=0, score=0.9, box=(700.0, 10.0, 720.0, 50.0)),  # x>640
            EngineDetection(label_id=1, score=0.9, box=(10.0, 20.0, 30.0, 40.0)),  # kept
        ]
    )
    dets = detector.detect(_frame())
    assert len(dets) == 1
    assert dets[0].label == "person"


def test_clipped_boxes_are_accepted_by_the_adapter() -> None:
    # End-to-end: a marginally-out-of-bounds prediction survives into a frozen
    # Detection because the backend clipped it (the adapter would reject a raw
    # negative coordinate as malformed).
    detector, _ = _detector(
        [EngineDetection(label_id=0, score=0.9, box=(-0.4, 5.0, 200.0, 480.9))]
    )
    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    (det,) = adapter.adapt_from(detector, _frame())
    assert (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2) == (0.0, 5.0, 200.0, 480.0)


# --- threshold plumbing & image handling -------------------------------------
def test_backend_threshold_is_forwarded_to_engine() -> None:
    detector, engine = _detector([], threshold=0.42)
    detector.detect(_frame())
    assert engine.seen_thresholds == [0.42]


def test_frame_image_is_passed_through_without_copy_or_reshape() -> None:
    detector, engine = _detector([])
    detector.detect(_frame(with_image=True))
    assert engine.seen_shapes == [(480, 640, 3)]  # RGB HxWx3, unaltered


def test_missing_image_raises_missing_frame_image_error() -> None:
    detector, _ = _detector([EngineDetection(label_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0))])
    with pytest.raises(MissingFrameImageError):
        detector.detect(_frame(with_image=False))


# --- malformed backend output ------------------------------------------------
def test_unmapped_class_id_is_malformed_backend_output() -> None:
    detector, _ = _detector([EngineDetection(label_id=99, score=0.9, box=(1.0, 1.0, 2.0, 2.0))])
    with pytest.raises(MalformedBackendOutputError):
        detector.detect(_frame())


def test_wrong_arity_box_is_malformed_backend_output() -> None:
    bad = EngineDetection(label_id=0, score=0.9, box=(1.0, 2.0, 3.0))  # type: ignore[arg-type]
    detector, _ = _detector([bad])
    with pytest.raises(MalformedBackendOutputError):
        detector.detect(_frame())


def test_malformed_backend_output_is_an_rtdetr_backend_error() -> None:
    detector, _ = _detector([EngineDetection(label_id=99, score=0.9, box=(1.0, 1.0, 2.0, 2.0))])
    with pytest.raises(RTDetrBackendError):
        detector.detect(_frame())


# --- no framework-native object escapes --------------------------------------
def test_no_framework_native_scalar_escapes_the_backend() -> None:
    # Engine returns numpy scalars (as a real torch/numpy path would); the backend
    # must coerce them to builtin float/str so nothing framework-native leaks out.
    det = EngineDetection(
        label_id=0,
        score=np.float32(0.5),  # type: ignore[arg-type]
        box=(np.float64(1.0), np.float32(2.0), np.float64(3.0), np.float32(4.0)),  # type: ignore[arg-type]
    )
    detector, _ = _detector([det])
    (raw,) = detector.detect(_frame())
    assert type(raw.score) is float
    assert type(raw.label) is str
    assert all(type(v) is float for v in raw.box)


# --- DetectionAdapter integration (frozen contracts) -------------------------
def test_adapter_integration_produces_frozen_detection() -> None:
    detector, _ = _detector([EngineDetection(label_id=0, score=0.9, box=(10.0, 20.0, 30.0, 40.0))])
    adapter = DetectionAdapter(
        DetectorConfig(
            label_map={"car": ObjectClass.CAR},
            source_model=ModelRef(name="rt-detr", version="test"),
        )
    )
    (det,) = adapter.adapt_from(detector, _frame())
    assert type(det) is Detection
    assert det.object_class is ObjectClass.CAR
    assert det.source_model is not None and det.source_model.name == "rt-detr"
    # Frozen: round-trips through JSON, rejects mutation.
    assert Detection.model_validate_json(det.model_dump_json()) == det
    with pytest.raises(ValidationError):
        det.confidence = 0.1  # type: ignore[misc]


def test_unmapped_native_label_retains_p1u6_drop_behavior() -> None:
    # "traffic_light" is a valid native label but is not in the adapter's label_map:
    # the adapter drops it (P1-U6 behavior), it is NOT an error.
    detector, _ = _detector(
        [
            EngineDetection(label_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0)),
            EngineDetection(label_id=2, score=0.9, box=(3.0, 3.0, 4.0, 4.0)),  # traffic_light
        ]
    )
    adapter = DetectionAdapter(DetectorConfig(label_map={"car": ObjectClass.CAR}))
    dets = adapter.adapt_from(detector, _frame())
    assert len(dets) == 1
    assert dets[0].object_class is ObjectClass.CAR


def test_adapter_score_threshold_gates_after_backend() -> None:
    # The adapter remains the authoritative confidence gate: a valid score below its
    # threshold is dropped even though the backend emitted it.
    detector, _ = _detector(
        [
            EngineDetection(label_id=0, score=0.4, box=(1.0, 1.0, 2.0, 2.0)),
            EngineDetection(label_id=1, score=0.6, box=(3.0, 3.0, 4.0, 4.0)),
        ]
    )
    adapter = DetectionAdapter(
        DetectorConfig(
            label_map={"car": ObjectClass.CAR, "person": ObjectClass.PERSON},
            score_threshold=0.5,
        )
    )
    dets = adapter.adapt_from(detector, _frame())
    assert [d.object_class for d in dets] == [ObjectClass.PERSON]


# --- determinism -------------------------------------------------------------
def test_repeated_inference_gives_deterministic_raw_and_detection() -> None:
    raws_in = [
        EngineDetection(label_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0)),
        EngineDetection(label_id=1, score=0.8, box=(3.0, 3.0, 4.0, 4.0)),
    ]
    detector, _ = _detector(raws_in)
    first = detector.detect(_frame())
    second = detector.detect(_frame())
    assert first == second

    adapter = DetectionAdapter(
        DetectorConfig(label_map={"car": ObjectClass.CAR, "person": ObjectClass.PERSON})
    )
    d1 = adapter.adapt(_frame(), first)
    d2 = adapter.adapt(_frame(), second)
    assert d1 == d2
    assert [d.detection_id for d in d1] == [d.detection_id for d in d2]


# --- backend-dependency error path (no torch/transformers installed) ---------
def test_missing_backend_dependencies_raise_backend_dependency_error() -> None:
    have_torch = importlib.util.find_spec("torch") is not None
    have_tf = importlib.util.find_spec("transformers") is not None
    if have_torch and have_tf:
        pytest.skip("torch+transformers installed; dependency-absence path not exercised here")
    # No engine injected -> the real engine load is attempted -> deps are missing.
    with pytest.raises(BackendDependencyError):
        RTDetrDetector(RTDetrConfig(checkpoint="local/does-not-matter"))

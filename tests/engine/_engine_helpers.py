"""Shared builders for the H6 engine tests.

Reuses the pipeline fixture helpers (path-shimmed by this directory's conftest)
and adds only engine-specific conveniences: full-size synthetic frame records
(the helmet observer crops pixels, so 1x1 images are not enough), a scripted
perf counter, and a stub engine factory. Uniquely named (``_engine_helpers``)
for pytest's prepend import mode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
from _pipeline_helpers import CAMERA, DETECTOR_CONFIG, NORTH_DIRECTION_ID, SCENE

from trafficpulse.classifier.interface import HelmetClassifier
from trafficpulse.contracts import ConfirmedEvent, SceneConfig
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector import DetectorConfig
from trafficpulse.detector.interface import Detector
from trafficpulse.engine import EngineConfig, InferenceEngine, MemoryLogSink, RuleConfig
from trafficpulse.ingestion.video import FrameRecord
from trafficpulse.tracking import IouTracker

__all__ = [
    "CAMERA",
    "DETECTOR_CONFIG",
    "NORTH_DIRECTION_ID",
    "SCENE",
    "sized_frame_record",
    "frame_records",
    "scripted_perf",
    "stub_engine",
    "event_at",
    "MEDIA_EPOCH",
]

MEDIA_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def sized_frame_record(
    frame_index: int,
    *,
    timestamp_seconds: float,
    camera_id: str | None = CAMERA,
    width: int = 320,
    height: int = 240,
) -> FrameRecord:
    """A synthetic ``FrameRecord`` with a real-size (zeroed) RGB image."""

    return FrameRecord(
        source_id="vsrc-test",
        camera_id=camera_id,
        frame_id=f"vfrm-{frame_index}",
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        width=width,
        height=height,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def frame_records(
    count: int, *, interval_seconds: float = 1.0 / 30.0, camera_id: str | None = CAMERA
) -> list[FrameRecord]:
    """``count`` synthetic records at a fixed media-time interval."""

    return [
        sized_frame_record(
            index, timestamp_seconds=index * interval_seconds, camera_id=camera_id
        )
        for index in range(count)
    ]


def scripted_perf(step_seconds: float = 1.0) -> Callable[[], float]:
    """A deterministic fake ``perf`` counter advancing ``step_seconds`` per call."""

    state = {"now": 0.0}

    def perf() -> float:
        state["now"] += step_seconds
        return state["now"]

    return perf


def stub_engine(
    *,
    detector: Detector,
    rules: Sequence[RuleConfig],
    scene: SceneConfig = SCENE,
    detector_config: DetectorConfig = DETECTOR_CONFIG,
    classifier: HelmetClassifier | None = None,
    **kwargs: object,
) -> tuple[InferenceEngine, MemoryLogSink]:
    """An engine over injected stubs + a fresh IoU tracker + a memory sink."""

    sink = MemoryLogSink()
    engine = InferenceEngine(
        scene=scene,
        detector=detector,
        tracker=IouTracker(),
        detector_config=detector_config,
        config=EngineConfig(rules=tuple(rules), **kwargs),  # type: ignore[arg-type]
        classifier=classifier,
        sink=sink,
    )
    return engine, sink


def event_at(
    trigger_seconds: float,
    *,
    start_seconds: float | None = None,
    camera_id: str = CAMERA,
    event_id: str = "evt-test",
) -> ConfirmedEvent:
    """A minimal frozen event whose timestamps are media-epoch anchored."""

    trigger_at = MEDIA_EPOCH + timedelta(seconds=trigger_seconds)
    start_at = (
        trigger_at
        if start_seconds is None
        else MEDIA_EPOCH + timedelta(seconds=start_seconds)
    )
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id=camera_id,
        start_at=start_at,
        trigger_at=trigger_at,
        rule_id="wrong_way",
        created_at=trigger_at,
    )

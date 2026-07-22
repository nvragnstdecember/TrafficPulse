"""Triple-riding vertical slice, end to end offline (v1.1 U3).

Runs the real detect → track → rider-count-observe → reason chain through the
``TripleRidingPipeline`` over a scripted motorcycle-with-N-riders stream (no pixels
are read for counting, so in-memory ``FrameRecord``s suffice, matching the
no-helmet e2e). Asserts the confirmed event's evidence-bearing fields and the
2-rider false-positive suppression; persistence + the HTTP surface are covered by
``tests/app/test_app_triple_riding.py`` (no duplication).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from _triple_fixtures import scripted_rider_count_detector

from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector import DetectorConfig
from trafficpulse.ingestion.video import FrameRecord
from trafficpulse.pipeline.triple_riding import TripleRidingPipeline
from trafficpulse.tracking import IouTracker

SCENE_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"
SCENE: SceneConfig = SceneConfig.model_validate(
    yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
)
CAMERA = SCENE.scene.camera_id
FPS = 10.0
FRAME_COUNT = 30
WIDTH, HEIGHT = 320, 240
DETECTOR_CONFIG = DetectorConfig(
    label_map={"motorbike": ObjectClass.MOTORCYCLE, "person": ObjectClass.PERSON}
)


def _frames(count: int = FRAME_COUNT) -> list[FrameRecord]:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return [
        FrameRecord(
            source_id="vsrc-test",
            camera_id=CAMERA,
            frame_id=f"vfrm-{i}",
            frame_index=i,
            timestamp_seconds=i / FPS,
            width=WIDTH,
            height=HEIGHT,
            image=image,
        )
        for i in range(count)
    ]


def _pipeline(riders: int) -> TripleRidingPipeline:
    return TripleRidingPipeline(
        detector=scripted_rider_count_detector(riders=riders),
        tracker=IouTracker(),
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
    )


def test_three_riders_confirm_one_triple_riding_event() -> None:
    events = _pipeline(riders=3).process(_frames())
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.TRIPLE_RIDING


def test_two_riders_confirm_nothing() -> None:
    assert _pipeline(riders=2).process(_frames()) == ()


def test_event_carries_the_motorcycle_riders_count_and_confidence() -> None:
    event = _pipeline(riders=3).process(_frames())[0]
    # The event names the motorcycle and its three rider tracks.
    assert len(event.track_ids) == 4
    # Rider count travels as an evidence measurement.
    max_riders = next(m.value for m in event.measurements if m.name == "max_rider_count")
    assert max_riders == 3.0
    # Timing + threshold + a temporal-consistency confidence are populated.
    assert event.trigger_at is not None
    assert event.confidence.temporal_consistency == 1.0
    assert {"min_persistence", "rider_count_threshold"} <= {t.name for t in event.thresholds}


def test_replays_identically() -> None:
    first = [e.model_dump_json() for e in _pipeline(riders=3).process(_frames())]
    second = [e.model_dump_json() for e in _pipeline(riders=3).process(_frames())]
    assert first == second

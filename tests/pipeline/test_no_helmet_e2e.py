"""No-helmet slice end to end: video -> observations -> reasoning -> EventStore (P4-U5).

Runs the whole chain through the real ingestion-shaped inputs, the real tracker,
the real observer, the real reasoner, and the **unmodified** ``EventStore`` --
proving the reasoning integrates with the existing persistence + evidence
architecture without redesigning either.

Detection is a scripted ``StubDetector`` and classification a scripted
``StubHelmetClassifier``: no checkpoint, no dataset, no ML. That is the point --
the reasoning path is fully verifiable offline while the real classifier remains
gated on dataset/licence resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from trafficpulse.classifier import RawHelmetPrediction, StubHelmetClassifier
from trafficpulse.contracts import ObjectClass, SceneConfig
from trafficpulse.contracts.enums import ViolationType
from trafficpulse.detector import DetectorConfig, RawDetection, StubDetector
from trafficpulse.ingestion.video import FrameRecord
from trafficpulse.persistence import EventStore
from trafficpulse.pipeline.no_helmet import NoHelmetPipeline
from trafficpulse.tracking import IouTracker

SCENE_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"
SCENE: SceneConfig = SceneConfig.model_validate(
    yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))
)
CAMERA = SCENE.scene.camera_id

FPS = 10.0
FRAME_COUNT = 30  # 3.0s at 10fps -- comfortably past the 1.0s min_persistence
WIDTH, HEIGHT = 320, 240

DETECTOR_CONFIG = DetectorConfig(
    label_map={"motorbike": ObjectClass.MOTORCYCLE, "person": ObjectClass.PERSON}
)
NO_HELMET = RawHelmetPrediction("no_helmet", 0.88)
HELMET = RawHelmetPrediction("helmet", 0.91)
TURBAN = RawHelmetPrediction("turban", 0.79)


def _image() -> np.ndarray:
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    pattern = (((ys // 2) + (xs // 2)) % 2 * 255).astype(np.uint8)
    return np.stack([pattern] * 3, axis=-1)


def frames(count: int = FRAME_COUNT) -> list[FrameRecord]:
    return [
        FrameRecord(
            source_id="vsrc-test",
            camera_id=CAMERA,
            frame_id=f"vfrm-{i}",
            frame_index=i,
            timestamp_seconds=i / FPS,
            width=WIDTH,
            height=HEIGHT,
            image=_image(),
        )
        for i in range(count)
    ]


def _boxes(i: int) -> tuple[RawDetection, ...]:
    """A rider riding a motorcycle, drifting slowly so IoU association holds."""

    x = 40.0 + i * 1.5
    return (
        RawDetection(label="motorbike", score=0.9, box=(x, 120.0, x + 60.0, 200.0)),
        RawDetection(label="person", score=0.9, box=(x + 10.0, 60.0, x + 50.0, 180.0)),
    )


def detector(count: int = FRAME_COUNT) -> StubDetector:
    return StubDetector(per_frame={i: _boxes(i) for i in range(count)})


def pipeline(classifier: StubHelmetClassifier) -> NoHelmetPipeline:
    return NoHelmetPipeline(
        detector=detector(),
        tracker=IouTracker(),
        classifier=classifier,
        scene=SCENE,
        detector_config=DETECTOR_CONFIG,
    )


def _rider_scripted(prediction: RawHelmetPrediction) -> StubHelmetClassifier:
    """Script every rider track the tracker may mint (ids are tracker-assigned)."""

    return StubHelmetClassifier(prediction)


# --- the full chain ----------------------------------------------------------
def test_bare_headed_rider_confirms_end_to_end() -> None:
    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())

    assert len(events) == 1
    assert events[0].violation_type is ViolationType.NO_HELMET
    assert events[0].camera_id == CAMERA


def test_helmeted_rider_confirms_nothing() -> None:
    assert pipeline(_rider_scripted(HELMET)).process(frames()) == ()


def test_turban_rider_is_exempt_end_to_end() -> None:
    assert pipeline(_rider_scripted(TURBAN)).process(frames()) == ()


def test_unscripted_classifier_abstains_rather_than_confirming() -> None:
    """The stub's default is 'uncertain'; abstention must never confirm."""

    assert pipeline(StubHelmetClassifier()).process(frames()) == ()


def test_event_names_both_the_rider_and_the_motorcycle() -> None:
    pipe = pipeline(_rider_scripted(NO_HELMET))
    events = pipe.process(frames())

    associations = pipe.observer.associations()
    assert associations
    rider = associations[0].subject_track_id
    bike = associations[0].object_track_id
    assert set(events[0].track_ids) == {rider, bike}


def test_perception_and_reasoning_are_separable() -> None:
    """A run that confirms nothing is distinguishable from one that saw nothing."""

    pipe = pipeline(_rider_scripted(HELMET))
    events = pipe.process(frames())

    assert events == ()
    assert pipe.observer.derivation().observations  # it observed plenty


def test_confidence_breakdown_is_populated() -> None:
    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())
    breakdown = events[0].confidence

    assert breakdown.classifier == pytest.approx(0.88)
    assert breakdown.temporal_consistency == pytest.approx(1.0)
    assert breakdown.association is not None
    assert breakdown.aggregate is None  # never collapsed (§13)


def test_scene_hash_and_provenance_travel_onto_the_event() -> None:
    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())

    assert events[0].scene_config_hash is not None
    assert events[0].source_hypothesis_id is not None


# --- EventStore integration (unmodified) -------------------------------------
def test_events_persist_through_the_unmodified_event_store(tmp_path: Path) -> None:
    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())
    store = EventStore(tmp_path)

    stored = store.persist("run-1", events)

    assert len(stored) == 1
    assert stored[0].manifest.event_id == stored[0].event.event_id


def test_persisted_events_reload_without_semantic_loss(tmp_path: Path) -> None:
    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())
    store = EventStore(tmp_path)
    store.persist("run-1", events)

    reloaded = store.load("run-1")

    assert [s.event.model_dump_json() for s in reloaded] == [e.model_dump_json() for e in events]


def test_evidence_manifest_is_built_by_the_existing_stub(tmp_path: Path) -> None:
    """Evidence is reused, not redesigned."""

    events = pipeline(_rider_scripted(NO_HELMET)).process(frames())
    stored = EventStore(tmp_path).persist("run-1", events)

    assert stored[0].manifest.scene_config_hash == events[0].scene_config_hash


def test_replay_persists_byte_identical_files(tmp_path: Path) -> None:
    """Write-once idempotency: an identical replay must not conflict."""

    store = EventStore(tmp_path)
    store.persist("run-1", pipeline(_rider_scripted(NO_HELMET)).process(frames()))
    store.persist("run-1", pipeline(_rider_scripted(NO_HELMET)).process(frames()))

    assert len(store.load("run-1")) == 1


# --- determinism -------------------------------------------------------------
def test_replay_yields_identical_events() -> None:
    def run() -> list[str]:
        return [e.model_dump_json() for e in pipeline(_rider_scripted(NO_HELMET)).process(frames())]

    assert run() == run()


def test_reset_clears_the_observation_stream_between_runs() -> None:
    pipe = pipeline(_rider_scripted(NO_HELMET))
    pipe.process(frames())
    first = len(pipe.observer.derivation().observations)
    pipe.process(frames())

    assert len(pipe.observer.derivation().observations) == first


# --- fail-fast ---------------------------------------------------------------
def test_scene_without_a_no_helmet_block_fails_fast() -> None:
    stripped = SCENE.model_copy(
        update={
            "rule_parameters": tuple(
                b
                for b in SCENE.rule_parameters
                if b.violation_type is not ViolationType.NO_HELMET
            )
        }
    )
    with pytest.raises(ValueError, match="no no_helmet rule-parameter block"):
        NoHelmetPipeline(
            detector=detector(),
            tracker=IouTracker(),
            classifier=StubHelmetClassifier(),
            scene=stripped,
            detector_config=DETECTOR_CONFIG,
        )


# --- boundary ----------------------------------------------------------------
def test_reasoning_imports_no_ml_framework() -> None:
    """No ML framework may leak into reasoning."""

    import trafficpulse.rules.no_helmet as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "transformers" not in source
    assert "from ..classifier" not in source

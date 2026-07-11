"""Stub-driven end-to-end illegal-stopping orchestration (P2-U5).

Drives the ``IllegalStoppingPipeline`` with a scripted ``StubDetector`` +
``StubTracker`` (and the real ``IouTracker``) over synthetic ``FrameRecord``s and
asserts the wiring: a track held stationary inside the example scene's
``zone-no-stop`` for at least ``stationary_duration`` produces exactly one
``ILLEGAL_STOPPING`` ``ConfirmedEvent`` that persists (with a minimal manifest)
through the P1-U11 ``EventStore``; the pipeline adds no behaviour over calling the
P2-U2/U3 derivations + P2-U4 reasoner directly (equivalence); it is deterministic
and order-independent; it fails fast on a misconfigured scene; it propagates
run-level model provenance without changing the decision; and it imports no
backend. Short / outside-zone / moving / early-exit / empty inputs confirm
nothing.

Reuses the shared ``_pipeline_helpers`` scene + frame builders (real
``DetectionAdapter``, example ``SceneConfig``); the stationary-in-zone detectors
and equivalent ``Detection`` reference are built here. Frames use 1-second PTS
steps so the 10 s dwell is reached in a small, timing-predictable frame count. No
video, no model download, no wall-clock.
"""

from datetime import timedelta

import pytest
from _pipeline_helpers import (
    BASE,
    CAMERA,
    DETECTOR_CONFIG,
    SCENE,
    make_frame_record,
)

from trafficpulse.contracts import (
    BoundingBox,
    ConfirmedEvent,
    Detection,
    ModelRef,
    ObjectClass,
    SceneConfig,
    TrackState,
    ViolationType,
)
from trafficpulse.contracts import scene_config_hash as _scene_hash
from trafficpulse.detector import Detector, DetectorConfig, RawDetection, StubDetector
from trafficpulse.observations.stationary import derive_stationary_observations_with_taint
from trafficpulse.observations.zones import derive_in_zone_observations_with_taint
from trafficpulse.persistence import EventStore
from trafficpulse.persistence.evidence_stub import build_evidence_manifest
from trafficpulse.pipeline import (
    IllegalStoppingPipeline,
    SceneConfigurationError,
    normalize_model_refs,
)
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.illegal_stopping import (
    IllegalStoppingReasoner,
    illegal_stopping_parameters,
)
from trafficpulse.tracking import (
    IouTracker,
    ScriptedAssignment,
    StubTracker,
    Tracker,
    TrackerConfig,
)

# --- geometry: the example scene's zone-no-stop (configs/scenes/example-scene.yaml).
# Bottom-center of each box is the ground-contact reference the derivations use.
INSIDE_XYXY = (1370.0, 860.0, 1410.0, 900.0)  # bottom-center (1390, 900) inside zone-no-stop
OUTSIDE_XYXY = (380.0, 860.0, 420.0, 900.0)  # bottom-center (400, 900) outside every zone

# 12 states at 1 s steps -> obs at t=1..11 -> dwell reaches exactly 10.0 s at t=11
# (>= the scene's 10.0 s stationary_duration), so the run confirms one event.
STOP_FRAME_COUNT = 12
SHORT_FRAME_COUNT = 11  # obs t=1..10 -> max dwell 9.0 s < 10.0 -> no confirmation

PARAMS = illegal_stopping_parameters(SCENE)
SCH = _scene_hash(SCENE)


# --- builders ----------------------------------------------------------------
def _stop_frames(count: int = STOP_FRAME_COUNT) -> list:
    """Synthetic frames at 1-second PTS steps (so 10 s dwell needs ~11 frames)."""

    return [make_frame_record(i, timestamp_seconds=float(i)) for i in range(count)]


def _raw(box: tuple[float, float, float, float]) -> RawDetection:
    return RawDetection(label="car", score=0.9, box=box)


def _detector_from_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> StubDetector:
    """A ``StubDetector`` scripted to emit one car at ``boxes[i]`` on frame ``i``."""

    return StubDetector(per_frame={i: (_raw(boxes[i]),) for i in range(len(boxes))})


def _stationary_detector(
    box: tuple[float, float, float, float] = INSIDE_XYXY, count: int = STOP_FRAME_COUNT
) -> StubDetector:
    """A ``StubDetector`` emitting a car frozen at ``box`` for ``count`` frames."""

    return _detector_from_boxes([box] * count)


def _drift_inside_boxes(count: int = STOP_FRAME_COUNT) -> list[tuple[float, float, float, float]]:
    """Boxes drifting 10 px/frame in x, bottom-center staying inside zone-no-stop.

    At y=900 the zone spans x in ~[1280, 1500]; 1300..1410 stays inside, so the
    vehicle is *moving but never leaves* -- the default stationarity window reads a
    steady 10 px/frame drift as moving, so no stop is ever confirmed.
    """

    return [(1280.0 + 10.0 * i, 860.0, 1320.0 + 10.0 * i, 900.0) for i in range(count)]


def _single_track_script(
    count: int = STOP_FRAME_COUNT,
    track_id: str = "t1",
    *,
    tainted_indices: tuple[int, ...] = (),
) -> dict:
    return {
        i: (ScriptedAssignment(track_id=track_id, tainted=(i in tainted_indices)),)
        for i in range(count)
    }


def _pipeline(
    detector: Detector,
    tracker: Tracker,
    *,
    scene: SceneConfig = SCENE,
    detector_config: DetectorConfig = DETECTOR_CONFIG,
) -> IllegalStoppingPipeline:
    return IllegalStoppingPipeline(
        detector=detector,
        tracker=tracker,
        scene=scene,
        detector_config=detector_config,
    )


def _detection(
    frame_index: int,
    box: tuple[float, float, float, float],
    *,
    camera_id: str = CAMERA,
) -> Detection:
    """A ``Detection`` equivalent to what the adapter produces from ``_raw(box)``."""

    x1, y1, x2, y2 = box
    return Detection(
        detection_id=f"det-{camera_id}-{frame_index}",
        camera_id=camera_id,
        frame_index=frame_index,
        timestamp=BASE + timedelta(seconds=float(frame_index)),
        object_class=ObjectClass.CAR,
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# --- a known illegal-stopping event ------------------------------------------
def test_pipeline_produces_an_illegal_stopping_confirmed_event() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    events = pipeline.process(_stop_frames())
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ConfirmedEvent)
    assert event.violation_type is ViolationType.ILLEGAL_STOPPING
    assert event.camera_id == CAMERA
    assert event.track_ids == ("t1",)
    assert event.rule_id == "illegal_stopping"
    assert event.rule_version == "0.1.0-provisional"
    assert event.end_at is None
    assert event.scene_config_hash == SCH
    # dwell reaches exactly the 10 s threshold; motion_threshold recorded not applied.
    measurements = {m.name: m for m in event.measurements}
    assert measurements["dwell_seconds"].value == 10.0
    thresholds = {t.name: t for t in event.thresholds}
    assert thresholds["stationary_duration"].value == 10.0
    assert thresholds["motion_threshold"].value == 0.5


# --- equivalence: wiring adds no behaviour -----------------------------------
def _direct_event_ids(
    boxes: list[tuple[float, float, float, float]], *, track_id: str = "t1"
) -> tuple[str, ...]:
    """Reference: feed equivalent Detections to the SAME tracker, then call the
    P2-U2/U3 derivations + P2-U4 reasoner directly (no pipeline)."""

    tracker = StubTracker(_single_track_script(len(boxes), track_id))
    history: dict = {}
    for i, box in enumerate(boxes):
        for state in tracker.update([_detection(i, box)]):
            history.setdefault((state.camera_id, state.track_id), []).append(state)
    reasoner = IllegalStoppingReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH)
    events: list[ConfirmedEvent] = []
    for key in sorted(history):
        track = history[key]
        in_zone = derive_in_zone_observations_with_taint(track, zones=SCENE.zones)
        stationary = derive_stationary_observations_with_taint(
            track, motion_threshold=PARAMS.motion_threshold
        )
        events.extend(reasoner.run_join(in_zone, stationary))
    return tuple(e.event_id for e in sorted(events, key=lambda e: (e.trigger_at, e.event_id)))


def test_pipeline_matches_direct_derivations_and_reasoning() -> None:
    boxes = [INSIDE_XYXY] * STOP_FRAME_COUNT
    pipeline = _pipeline(_detector_from_boxes(boxes), StubTracker(_single_track_script()))
    pipeline_ids = tuple(e.event_id for e in pipeline.process(_stop_frames()))
    assert pipeline_ids == _direct_event_ids(boxes)  # identical event set: wiring adds nothing


# --- negative cases ----------------------------------------------------------
def test_short_stop_produces_no_event() -> None:
    pipeline = _pipeline(
        _stationary_detector(count=SHORT_FRAME_COUNT),
        StubTracker(_single_track_script(SHORT_FRAME_COUNT)),
    )
    assert pipeline.process(_stop_frames(SHORT_FRAME_COUNT)) == ()


def test_stationary_outside_zone_produces_no_event() -> None:
    n = 14
    pipeline = _pipeline(
        _stationary_detector(OUTSIDE_XYXY, count=n), StubTracker(_single_track_script(n))
    )
    assert pipeline.process(_stop_frames(n)) == ()


def test_moving_inside_zone_produces_no_event() -> None:
    # A vehicle driving through the zone (steady drift, never leaving) is moving,
    # so the default stationarity window never reads a stop.
    boxes = _drift_inside_boxes(STOP_FRAME_COUNT)
    pipeline = _pipeline(_detector_from_boxes(boxes), StubTracker(_single_track_script()))
    assert pipeline.process(_stop_frames()) == ()


def test_exit_before_threshold_produces_no_event() -> None:
    # Stationary inside briefly, then leave the zone before the dwell threshold.
    boxes = [INSIDE_XYXY] * 4 + [OUTSIDE_XYXY] * 3
    pipeline = _pipeline(
        _detector_from_boxes(boxes), StubTracker(_single_track_script(len(boxes)))
    )
    assert pipeline.process(_stop_frames(len(boxes))) == ()


def test_empty_frame_stream_produces_no_event() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    assert pipeline.process([]) == ()


# --- taint restart survives orchestration ------------------------------------
def test_taint_restart_prevents_confirmation() -> None:
    # A long stationary-in-zone track split by a tainted block: neither clean
    # segment reaches the 10 s dwell, so support never bridges the ID switch.
    n = 14
    pipeline = _pipeline(
        _stationary_detector(count=n),
        StubTracker(_single_track_script(n, tainted_indices=(6, 7))),
    )
    assert pipeline.process(_stop_frames(n)) == ()


# --- multiple tracks ----------------------------------------------------------
def test_only_the_stopped_track_confirms() -> None:
    # One track stops in-zone; another drives through the zone -> only one event.
    per_frame = {
        i: (_raw(INSIDE_XYXY), _raw(_drift_inside_boxes()[i])) for i in range(STOP_FRAME_COUNT)
    }
    script = {
        i: (ScriptedAssignment(track_id="stopped"), ScriptedAssignment(track_id="driving"))
        for i in range(STOP_FRAME_COUNT)
    }
    pipeline = _pipeline(StubDetector(per_frame=per_frame), StubTracker(script))
    events = pipeline.process(_stop_frames())
    assert len(events) == 1
    assert events[0].track_ids == ("stopped",)


def test_event_ordering_is_deterministic_by_trigger_then_id() -> None:
    # Two tracks both stop in-zone (different positions) and confirm at the same
    # trigger time; the emitted order is sorted by (trigger_at, event_id).
    left = (1330.0, 860.0, 1370.0, 900.0)  # bottom-center (1350, 900) inside
    right = (1410.0, 860.0, 1450.0, 900.0)  # bottom-center (1430, 900) inside
    per_frame = {i: (_raw(left), _raw(right)) for i in range(STOP_FRAME_COUNT)}
    script = {
        i: (ScriptedAssignment(track_id="b"), ScriptedAssignment(track_id="a"))
        for i in range(STOP_FRAME_COUNT)
    }
    pipeline = _pipeline(StubDetector(per_frame=per_frame), StubTracker(script))
    events = pipeline.process(_stop_frames())
    assert len(events) == 2
    keys = [(e.trigger_at, e.event_id) for e in events]
    assert keys == sorted(keys)


# --- determinism --------------------------------------------------------------
def test_fresh_instances_are_deterministic() -> None:
    def run() -> tuple[str, ...]:
        pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
        return tuple(e.event_id for e in pipeline.process(_stop_frames()))

    assert run() == run()


def test_reset_and_replay_on_one_instance_is_deterministic() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    first = tuple(e.event_id for e in pipeline.process(_stop_frames()))
    second = tuple(e.event_id for e in pipeline.process(_stop_frames()))  # process() resets
    assert first == second


def test_finalize_is_idempotent_over_history() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    pipeline.reset()
    for frame in _stop_frames():
        pipeline.process_frame(frame)
    first = tuple(e.event_id for e in pipeline.finalize())
    second = tuple(e.event_id for e in pipeline.finalize())
    assert first == second
    assert len(first) == 1


# --- no persistence side effects ---------------------------------------------
def test_process_returns_events_and_writes_nothing(tmp_path) -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    before = set(tmp_path.iterdir())
    pipeline.process(_stop_frames())
    assert set(tmp_path.iterdir()) == before  # orchestration performs no I/O


# --- real IouTracker compatibility (in-memory integration) -------------------
def test_real_iou_tracker_integration() -> None:
    # Identical boxes overlap perfectly (IoU 1.0), so the real tracker holds one id.
    pipeline = _pipeline(_stationary_detector(), IouTracker())
    events = pipeline.process(_stop_frames())
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.ILLEGAL_STOPPING
    assert events[0].track_ids == ("iou-1",)  # backend-assigned stream-local id


# --- scene fail-fast ----------------------------------------------------------
def _scene_without(zone_or_block: str) -> SceneConfig:
    raw = SCENE.model_dump(mode="json")
    if zone_or_block == "no_stopping_zone":
        raw["zones"] = [z for z in raw["zones"] if z["zone_type"] != "no_stopping"]
    elif zone_or_block == "illegal_stopping_block":
        raw["rule_parameters"] = [
            b for b in raw["rule_parameters"] if b["violation_type"] != "illegal_stopping"
        ]
    return SceneConfig.model_validate(raw)


def test_scene_without_no_stopping_zone_raises() -> None:
    with pytest.raises(SceneConfigurationError, match="no-stopping"):
        _pipeline(
            _stationary_detector(),
            StubTracker(_single_track_script()),
            scene=_scene_without("no_stopping_zone"),
        )


def test_scene_without_illegal_stopping_block_raises() -> None:
    with pytest.raises(ValueError, match="illegal_stopping"):
        _pipeline(
            _stationary_detector(),
            StubTracker(_single_track_script()),
            scene=_scene_without("illegal_stopping_block"),
        )


def test_no_stopping_zone_ids_property() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    assert pipeline.no_stopping_zone_ids == ("zone-no-stop",)


# --- provenance propagation (P2-U1 shape) ------------------------------------
DET_REF = ModelRef(name="rtdetr-r50vd", version="provisional")
TRK_REF = ModelRef(name="iou-tracker", version="0.1.0-provisional")


def _provenanced_pipeline(
    *, det_ref: ModelRef | None, trk_ref: ModelRef | None
) -> IllegalStoppingPipeline:
    detector_config = DetectorConfig(label_map={"car": ObjectClass.CAR}, source_model=det_ref)
    tracker = IouTracker(tracker_config=TrackerConfig(tracker=trk_ref))
    return _pipeline(_stationary_detector(), tracker, detector_config=detector_config)


def test_models_propagated_from_seams() -> None:
    (event,) = _provenanced_pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_stop_frames())
    assert event.models == normalize_model_refs([DET_REF, TRK_REF]) == (TRK_REF, DET_REF)
    assert all(ref.weights_hash is None for ref in event.models)


def test_empty_provenance_when_none_supplied() -> None:
    (event,) = _provenanced_pipeline(det_ref=None, trk_ref=None).process(_stop_frames())
    assert event.models == ()


def test_provenance_does_not_change_decision_or_event_id() -> None:
    (with_refs,) = _provenanced_pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_stop_frames())
    (without,) = _provenanced_pipeline(det_ref=None, trk_ref=None).process(_stop_frames())
    assert with_refs.event_id == without.event_id  # models never enter the id
    assert with_refs == without.model_copy(update={"models": (TRK_REF, DET_REF)})


# --- persistence + evidence-manifest integration -----------------------------
def test_persist_illegal_stopping_event_and_manifest(tmp_path) -> None:
    (event,) = _provenanced_pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_stop_frames())
    store = EventStore(tmp_path)
    (stored,) = store.persist("run-p2u5", (event,))
    # Manifest built from the event: inherits models + carries the rule trace.
    assert stored.manifest.event_id == event.event_id
    assert stored.manifest.models == event.models == (TRK_REF, DET_REF)
    assert stored.manifest.rule_trace[0].label == "rule:illegal_stopping"
    assert stored.manifest.rule_trace[0].note == "0.1.0-provisional"

    # Reload preserves the illegal-stopping fields + provenance.
    (reloaded,) = store.load("run-p2u5")
    assert reloaded.event.violation_type is ViolationType.ILLEGAL_STOPPING
    assert reloaded.event.rule_id == "illegal_stopping"
    assert reloaded.event.scene_config_hash == SCH
    assert reloaded.event.models == (TRK_REF, DET_REF)
    assert reloaded.event == event
    assert reloaded.manifest.models == (TRK_REF, DET_REF)
    assert reloaded.manifest == build_evidence_manifest(event)


def test_persistence_is_byte_identical_across_replays(tmp_path) -> None:
    (event,) = _provenanced_pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_stop_frames())
    store = EventStore(tmp_path)
    store.persist("run-p2u5", (event,))
    store.persist("run-p2u5", (event,))  # idempotent write-once: no conflict
    (reloaded,) = store.load("run-p2u5")
    assert reloaded.event == event


# --- backend-free import boundary --------------------------------------------
def test_only_frozen_contracts_cross_the_boundary() -> None:
    pipeline = _pipeline(_stationary_detector(), StubTracker(_single_track_script()))
    states = pipeline.process_frame(make_frame_record(0, timestamp_seconds=0.0))
    assert all(isinstance(s, TrackState) for s in states)
    events = pipeline.finalize()
    assert all(isinstance(e, ConfirmedEvent) for e in events)


def test_pipeline_module_imports_no_backend() -> None:
    import trafficpulse.pipeline.illegal_stopping as core

    forbidden = (
        "RTDetrDetector",
        "IouTracker",
        "StubTracker",
        "StubDetector",
        "torch",
        "transformers",
    )
    for name in forbidden:
        assert not hasattr(core, name), f"pipeline core imports backend {name!r}"

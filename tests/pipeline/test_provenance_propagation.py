"""Composition-boundary model-provenance propagation (P2-U1).

Proves the full provenance path on the wrong-way slice: the pipeline collects the
truthful ``ModelRef``s the detector/tracker adapters stamp onto
``Detection.source_model`` / ``TrackState.tracker``, de-duplicates and orders them
deterministically, and stamps them onto every ``ConfirmedEvent.models`` -- which
the ``EvidenceManifest`` inherits and the ``EventStore`` round-trips unchanged.
Provenance never changes the decision (which events, ids, timing), and a stub run
that supplies no refs yields an honestly empty tuple.

Uses the shared ``_pipeline_helpers`` builders (real ``DetectionAdapter`` + real
``IouTracker``, scripted ``StubDetector``); no video, no model download.
"""

from _pipeline_helpers import (
    DEFAULT_FRAME_COUNT,
    NORTH_DIRECTION_ID,
    SCENE,
    make_frame_record,
    moving_down_detector,
)

from trafficpulse.contracts import ModelRef, ObjectClass
from trafficpulse.detector import DetectorConfig
from trafficpulse.persistence import EventStore
from trafficpulse.persistence.evidence_stub import build_evidence_manifest
from trafficpulse.pipeline import WrongWayPipeline, normalize_model_refs
from trafficpulse.tracking import IouTracker, TrackerConfig

DET_REF = ModelRef(name="rtdetr-r50vd", version="provisional")
TRK_REF = ModelRef(name="iou-tracker", version="0.1.0-provisional")
# A second, distinct ref to exercise ordering by version.
DET_REF_V2 = ModelRef(name="rtdetr-r50vd", version="0.2.0-provisional")


def _frames(frame_count: int = DEFAULT_FRAME_COUNT) -> list:
    return [make_frame_record(i) for i in range(frame_count)]


def _pipeline(*, det_ref: ModelRef | None, trk_ref: ModelRef | None) -> WrongWayPipeline:
    """A wrong-way pipeline whose adapters stamp the given (optional) refs."""

    detector_config = DetectorConfig(label_map={"car": ObjectClass.CAR}, source_model=det_ref)
    tracker = IouTracker(tracker_config=TrackerConfig(tracker=trk_ref))
    return WrongWayPipeline(
        detector=moving_down_detector(),
        tracker=tracker,
        scene=SCENE,
        detector_config=detector_config,
        direction_id=NORTH_DIRECTION_ID,
    )


# --- normalize_model_refs (pure helper) --------------------------------------
def test_normalize_empty_is_empty() -> None:
    assert normalize_model_refs([]) == ()


def test_normalize_sorts_by_name_then_version() -> None:
    # Input deliberately out of order; DET_REF ("rtdetr...") sorts after TRK_REF.
    assert normalize_model_refs([DET_REF, TRK_REF]) == (TRK_REF, DET_REF)
    # Same name, different version: ordered by version string.
    assert normalize_model_refs([DET_REF, DET_REF_V2]) == (DET_REF_V2, DET_REF)


def test_normalize_dedups_by_full_identity() -> None:
    # The same ref repeated (as a per-frame detector emits) collapses to one entry.
    assert normalize_model_refs([DET_REF, DET_REF, DET_REF]) == (DET_REF,)
    assert normalize_model_refs([TRK_REF, DET_REF, TRK_REF, DET_REF]) == (TRK_REF, DET_REF)


def test_normalize_is_order_independent() -> None:
    a = normalize_model_refs([DET_REF, TRK_REF, DET_REF_V2])
    b = normalize_model_refs([DET_REF_V2, DET_REF, TRK_REF, TRK_REF])
    assert a == b == (TRK_REF, DET_REF_V2, DET_REF)


# --- pipeline collection -----------------------------------------------------
def test_pipeline_collects_detector_and_tracker_refs() -> None:
    pipeline = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF)
    (event,) = pipeline.process(_frames())
    assert event.models == normalize_model_refs([DET_REF, TRK_REF])
    assert event.models == (TRK_REF, DET_REF)


def test_pipeline_dedups_per_frame_repeated_refs() -> None:
    # The detector stamps DET_REF on every frame and the tracker TRK_REF on every
    # state, yet each contributes exactly one entry.
    pipeline = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF)
    (event,) = pipeline.process(_frames())
    assert len(event.models) == 2


def test_pipeline_empty_provenance_when_none_supplied() -> None:
    pipeline = _pipeline(det_ref=None, trk_ref=None)
    (event,) = pipeline.process(_frames())
    assert event.models == ()


def test_pipeline_collects_only_the_supplied_side() -> None:
    # Detector ref only -> just that; tracker ref only -> just that.
    (det_only,) = _pipeline(det_ref=DET_REF, trk_ref=None).process(_frames())
    assert det_only.models == (DET_REF,)
    (trk_only,) = _pipeline(det_ref=None, trk_ref=TRK_REF).process(_frames())
    assert trk_only.models == (TRK_REF,)


def test_weights_hash_is_none_everywhere() -> None:
    (event,) = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
    assert event.models  # non-empty
    assert all(ref.weights_hash is None for ref in event.models)


# --- decision independence ----------------------------------------------------
def test_provenance_does_not_change_decision_or_event_id() -> None:
    with_refs = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
    without = _pipeline(det_ref=None, trk_ref=None).process(_frames())
    assert tuple(e.event_id for e in with_refs) == tuple(e.event_id for e in without)
    # Everything but the inert models field is identical.
    assert len(with_refs) == len(without) == 1
    assert with_refs[0] == without[0].model_copy(
        update={"models": normalize_model_refs([DET_REF, TRK_REF])}
    )


# --- determinism --------------------------------------------------------------
def test_fresh_instances_produce_identical_models() -> None:
    def run() -> tuple[ModelRef, ...]:
        (event,) = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
        return event.models

    assert run() == run()


def test_reset_and_replay_produce_identical_models() -> None:
    pipeline = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF)
    (first,) = pipeline.process(_frames())  # process() resets internally
    (second,) = pipeline.process(_frames())
    assert first.models == second.models == (TRK_REF, DET_REF)


def test_reset_clears_accumulated_provenance() -> None:
    # A ref-bearing run followed by reset + a ref-free stream must not leak the
    # first run's refs into the second.
    pipeline = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF)
    pipeline.process(_frames())
    pipeline.reset()
    for frame in _frames():
        pipeline.process_frame(frame)
    # Refs were accumulated afresh from the (still ref-bearing) second stream.
    (event,) = pipeline.finalize()
    assert event.models == (TRK_REF, DET_REF)


# --- manifest inheritance + persistence round-trip ---------------------------
def test_manifest_inherits_event_models() -> None:
    (event,) = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
    manifest = build_evidence_manifest(event)
    assert manifest.models == event.models == (TRK_REF, DET_REF)


def test_persistence_round_trip_preserves_event_and_manifest_models(tmp_path) -> None:
    (event,) = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
    store = EventStore(tmp_path)
    store.persist("run-p2u1", (event,))
    (reloaded,) = store.load("run-p2u1")
    assert reloaded.event.models == (TRK_REF, DET_REF)
    assert reloaded.manifest.models == (TRK_REF, DET_REF)
    assert reloaded.event.models == event.models


def test_persistence_is_byte_identical_across_replays(tmp_path) -> None:
    (event,) = _pipeline(det_ref=DET_REF, trk_ref=TRK_REF).process(_frames())
    store = EventStore(tmp_path)
    store.persist("run-p2u1", (event,))
    # Idempotent write-once: re-persisting the same event (models included) is a
    # no-op, never an EventConflictError.
    store.persist("run-p2u1", (event,))
    (reloaded,) = store.load("run-p2u1")
    assert reloaded.event.models == (TRK_REF, DET_REF)

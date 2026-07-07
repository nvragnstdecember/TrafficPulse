"""JSON serialization/deserialization round-trip tests for public contracts."""

from datetime import UTC, datetime

from trafficpulse.contracts import (
    Association,
    AssociationType,
    BoundingBox,
    ConfidenceBreakdown,
    ConfirmedEvent,
    Detection,
    EvidenceManifest,
    LifecycleState,
    MeasuredValue,
    ModelRef,
    ObjectClass,
    ReviewCase,
    ReviewStatus,
    SimulatedAmount,
    SimulatedPenalty,
    SimulatedPenaltyStatus,
    TemporalState,
    TimeInterval,
    TrackState,
    TrackStatus,
    ViolationHypothesis,
    ViolationType,
)
from trafficpulse.contracts.enums import ArtifactKind
from trafficpulse.contracts.evidence import ArtifactReference, OcrResult, RuleTraceStep

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TS2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)
BBOX = BoundingBox(x1=1.0, y1=2.0, x2=11.0, y2=22.0)


def test_detection_roundtrip() -> None:
    d = Detection(
        detection_id="d1",
        camera_id="cam1",
        frame_index=3,
        timestamp=TS,
        object_class=ObjectClass.MOTORCYCLE,
        confidence=0.91,
        bbox=BBOX,
        source_model=ModelRef(name="yolo", version="11s"),
    )
    assert Detection.model_validate_json(d.model_dump_json()) == d


def test_track_state_roundtrip() -> None:
    t = TrackState(
        track_id="t1",
        camera_id="cam1",
        timestamp=TS,
        object_class=ObjectClass.CAR,
        bbox=BBOX,
        confidence=0.8,
        status=TrackStatus.ACTIVE,
        tainted=True,
    )
    assert TrackState.model_validate_json(t.model_dump_json()) == t


def test_association_roundtrip() -> None:
    a = Association(
        association_id="a1",
        camera_id="cam1",
        subject_track_id="head1",
        object_track_id="rider1",
        association_type=AssociationType.HEAD_OF_RIDER,
        confidence=0.75,
        timestamp=TS,
        interval=TimeInterval(start=TS, end=TS2),
    )
    assert Association.model_validate_json(a.model_dump_json()) == a


def test_temporal_state_roundtrip() -> None:
    s = TemporalState(
        state_id="s1",
        camera_id="cam1",
        track_id="t1",
        rule_id="wrong_way",
        lifecycle_state=LifecycleState.CONFIRMED,
        accumulated_score=2.5,
        observation_count=12,
        first_observation_at=TS,
        last_observation_at=TS2,
        updated_at=TS2,
    )
    assert TemporalState.model_validate_json(s.model_dump_json()) == s


def test_violation_hypothesis_roundtrip() -> None:
    h = ViolationHypothesis(
        hypothesis_id="h1",
        violation_type=ViolationType.RED_LIGHT_JUMPING,
        camera_id="cam1",
        track_ids=("t1", "t2"),
        interval=TimeInterval(start=TS, end=TS2),
        state=LifecycleState.CANDIDATE,
        rule_id="red_light",
        rule_version="0.1.0",
        confidence=ConfidenceBreakdown(detector=0.9, temporal_consistency=0.6),
        reasons=("crossed_stop_line_on_red",),
        measurements=(MeasuredValue(name="crossing_speed", value=8.0, unit="m_per_s"),),
        thresholds=(MeasuredValue(name="grace_s", value=0.3, unit="s"),),
    )
    assert ViolationHypothesis.model_validate_json(h.model_dump_json()) == h


def test_confirmed_event_roundtrip() -> None:
    e = ConfirmedEvent(
        event_id="e1",
        violation_type=ViolationType.ILLEGAL_STOPPING,
        camera_id="cam1",
        track_ids=("t1",),
        start_at=TS,
        trigger_at=TS2,
        end_at=TS2,
        rule_id="stopping",
        rule_version="1.2.3",
        confidence=ConfidenceBreakdown(aggregate=0.88),
        measurements=(MeasuredValue(name="dwell_s", value=42.0, unit="s"),),
        thresholds=(MeasuredValue(name="dwell_limit_s", value=30.0, unit="s"),),
        scene_config_hash="a" * 64,
        models=(ModelRef(name="yolo", version="11m", weights_hash="b" * 64),),
        code_version="deadbeef",
        source_hypothesis_id="h1",
        created_at=TS2,
    )
    assert ConfirmedEvent.model_validate_json(e.model_dump_json()) == e


def test_evidence_manifest_roundtrip() -> None:
    m = EvidenceManifest(
        evidence_package_id="ep1",
        event_id="e1",
        before_frame=ArtifactReference(
            kind=ArtifactKind.BEFORE_FRAME,
            locator="frames/before.jpg",
            sha256="c" * 64,
        ),
        trigger_frame=ArtifactReference(
            kind=ArtifactKind.TRIGGER_FRAME,
            locator="frames/trigger.jpg",
        ),
        clip=ArtifactReference(kind=ArtifactKind.CLIP, locator="clips/e1.mp4"),
        plate_crop=ArtifactReference(kind=ArtifactKind.PLATE_CROP, locator="crops/e1.jpg"),
        ocr=OcrResult(text="KA01AB1234", confidence=0.7, per_char_confidence=(0.9, 0.8)),
        rule_trace=(
            RuleTraceStep(index=0, label="entered_zone"),
            RuleTraceStep(
                index=1,
                label="dwell_exceeded",
                measurements=(MeasuredValue(name="dwell_s", value=42.0),),
            ),
        ),
        models=(ModelRef(name="paddleocr", version="pp-ocrv4"),),
        code_version="deadbeef",
        scene_config_hash="d" * 64,
        created_at=TS,
    )
    assert EvidenceManifest.model_validate_json(m.model_dump_json()) == m


def test_review_case_roundtrip() -> None:
    r = ReviewCase(
        review_case_id="rc1",
        evidence_package_id="ep1",
        status=ReviewStatus.APPROVED,
        reviewer_id="reviewer-opaque-42",
        decided_at=TS2,
        note="clear violation",
        audit_ref="audit/rc1",
        created_at=TS,
    )
    assert ReviewCase.model_validate_json(r.model_dump_json()) == r


def test_simulated_penalty_roundtrip() -> None:
    p = SimulatedPenalty(
        penalty_id="p1",
        review_case_id="rc1",
        status=SimulatedPenaltyStatus.ISSUED,
        amount=SimulatedAmount(value=1000.0, currency="INR"),
        issued_at=TS,
        updated_at=TS2,
    )
    restored = SimulatedPenalty.model_validate_json(p.model_dump_json())
    assert restored == p
    assert restored.simulated is True
    assert restored.disclaimer == "SIMULATION - NOT A LEGAL NOTICE."

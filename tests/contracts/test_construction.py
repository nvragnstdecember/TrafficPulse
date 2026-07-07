"""Successful construction of representative U2 contracts."""

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
BBOX = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)


def test_detection_constructs() -> None:
    d = Detection(
        detection_id="d1",
        camera_id="cam1",
        frame_index=0,
        timestamp=TS,
        object_class=ObjectClass.MOTORCYCLE,
        confidence=0.9,
        bbox=BBOX,
    )
    assert d.object_class is ObjectClass.MOTORCYCLE
    assert d.bbox.x2 == 10.0


def test_track_state_constructs() -> None:
    t = TrackState(
        track_id="t1",
        camera_id="cam1",
        timestamp=TS,
        object_class=ObjectClass.CAR,
        bbox=BBOX,
        status=TrackStatus.ACTIVE,
    )
    assert t.tainted is False
    assert t.velocity is None


def test_association_constructs() -> None:
    a = Association(
        association_id="a1",
        camera_id="cam1",
        subject_track_id="rider1",
        object_track_id="moto1",
        association_type=AssociationType.RIDER_OF_MOTORCYCLE,
        confidence=0.8,
        timestamp=TS,
    )
    assert a.association_type is AssociationType.RIDER_OF_MOTORCYCLE


def test_temporal_state_constructs() -> None:
    s = TemporalState(
        state_id="s1",
        camera_id="cam1",
        track_id="t1",
        rule_id="wrong_way",
        lifecycle_state=LifecycleState.CANDIDATE,
        updated_at=TS,
    )
    assert s.observation_count == 0
    assert s.lifecycle_state is LifecycleState.CANDIDATE


def test_violation_hypothesis_constructs() -> None:
    h = ViolationHypothesis(
        hypothesis_id="h1",
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam1",
        track_ids=("t1",),
        interval=TimeInterval(start=TS, end=TS2),
        state=LifecycleState.CANDIDATE,
        rule_id="wrong_way",
        confidence=ConfidenceBreakdown(detector=0.9, geometric_margin=0.7),
        reasons=("sustained_contradiction",),
        measurements=(MeasuredValue(name="deviation_deg", value=170.0, unit="deg"),),
        thresholds=(MeasuredValue(name="theta_max", value=90.0, unit="deg"),),
    )
    assert h.confidence.detector == 0.9
    assert h.measurements[0].name == "deviation_deg"


def test_confirmed_event_constructs() -> None:
    e = ConfirmedEvent(
        event_id="e1",
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam1",
        track_ids=("t1",),
        start_at=TS,
        trigger_at=TS2,
        rule_id="wrong_way",
        rule_version="1.0.0",
        created_at=TS2,
    )
    assert e.violation_type is ViolationType.WRONG_WAY
    assert e.end_at is None


def test_evidence_manifest_constructs() -> None:
    m = EvidenceManifest(
        evidence_package_id="ep1",
        event_id="e1",
        trigger_frame=ArtifactReference(
            kind=ArtifactKind.TRIGGER_FRAME,
            locator="frames/e1_trigger.jpg",
        ),
        ocr=OcrResult(text="KA01AB1234", confidence=0.7),
        rule_trace=(RuleTraceStep(index=0, label="entered_lane"),),
        created_at=TS,
    )
    assert m.trigger_frame is not None
    assert m.ocr is not None
    assert m.ocr.text == "KA01AB1234"


def test_review_case_constructs() -> None:
    r = ReviewCase(
        review_case_id="rc1",
        evidence_package_id="ep1",
        status=ReviewStatus.PENDING,
        created_at=TS,
    )
    assert r.status is ReviewStatus.PENDING


def test_simulated_penalty_constructs() -> None:
    p = SimulatedPenalty(
        penalty_id="p1",
        review_case_id="rc1",
        status=SimulatedPenaltyStatus.ISSUED,
        amount=SimulatedAmount(value=500.0),
        issued_at=TS,
    )
    assert p.simulated is True
    assert p.disclaimer == "SIMULATION - NOT A LEGAL NOTICE."
    assert p.amount is not None
    assert p.amount.currency == "INR"

"""P1-U11 evidence-stub tests: minimal manifest creation, linkage, provenance.

Verifies :func:`build_evidence_manifest` builds a *valid* minimal
``EvidenceManifest`` that is a pure, deterministic function of the event; links to
it correctly; carries only real provenance (fabricating nothing); and references
the trigger frame by a relative locator with no rendered-artifact hash.
"""

from datetime import UTC, datetime

from trafficpulse.contracts import (
    ConfidenceBreakdown,
    ConfirmedEvent,
    EvidenceManifest,
    MeasuredValue,
    ModelRef,
    ViolationType,
)
from trafficpulse.contracts.enums import ArtifactKind
from trafficpulse.persistence import (
    build_evidence_manifest,
    evidence_package_id_for,
    trigger_frame_locator_for,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TS2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)


def _event(**overrides: object) -> ConfirmedEvent:
    base: dict[str, object] = dict(
        event_id="evt-abc123",
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-1",
        track_ids=("t1",),
        start_at=TS,
        trigger_at=TS2,
        rule_id="wrong_way",
        rule_version="0.1.0-provisional",
        scene_config_hash="a" * 64,
        source_hypothesis_id="h1",
        created_at=TS2,
        measurements=(MeasuredValue(name="persistence_seconds", value=5.0, unit="seconds"),),
        thresholds=(MeasuredValue(name="min_persistence", value=1.0, unit="seconds"),),
    )
    base.update(overrides)
    return ConfirmedEvent(**base)  # type: ignore[arg-type]


def test_manifest_is_a_valid_evidence_manifest() -> None:
    manifest = build_evidence_manifest(_event())
    assert isinstance(manifest, EvidenceManifest)
    # Round-trips through its own contract (validation is real, not bypassed).
    assert EvidenceManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_manifest_links_to_the_event() -> None:
    event = _event()
    manifest = build_evidence_manifest(event)
    assert manifest.event_id == event.event_id
    assert manifest.evidence_package_id == f"evp-{event.event_id}"
    assert manifest.evidence_package_id == evidence_package_id_for(event)


def test_trigger_frame_is_a_relative_locator_without_hash() -> None:
    event = _event()
    manifest = build_evidence_manifest(event)
    ref = manifest.trigger_frame
    assert ref is not None
    assert ref.kind is ArtifactKind.TRIGGER_FRAME
    assert ref.locator == trigger_frame_locator_for(event)
    assert event.event_id in ref.locator and event.camera_id in ref.locator
    # No artifact rendered -> no integrity hash fabricated.
    assert ref.sha256 is None


def test_no_rendered_artifacts_present() -> None:
    # The stub is metadata only: no clip/crop/overlay/before/after artifacts.
    manifest = build_evidence_manifest(_event())
    assert manifest.clip is None
    assert manifest.before_frame is None
    assert manifest.after_frame is None
    assert manifest.plate_crop is None
    assert manifest.trajectory is None
    assert manifest.ocr is None
    assert manifest.additional_artifacts == ()


def test_provenance_is_carried_from_the_event() -> None:
    event = _event(
        scene_config_hash="b" * 64,
        code_version="deadbeef",
        models=(ModelRef(name="rtdetr", version="r50"),),
    )
    manifest = build_evidence_manifest(event)
    assert manifest.scene_config_hash == "b" * 64
    assert manifest.code_version == "deadbeef"
    assert manifest.models == event.models
    assert manifest.created_at == event.created_at  # deterministic, never wall-clock


def test_models_not_fabricated_when_event_has_none() -> None:
    # The current reasoner does not stamp model/tracker refs onto the event; the
    # stub must reflect that honestly rather than invent provenance.
    event = _event()
    assert event.models == ()
    assert build_evidence_manifest(event).models == ()


def test_rule_trace_is_derived_from_event_fields() -> None:
    event = _event()
    manifest = build_evidence_manifest(event)
    assert len(manifest.rule_trace) == 2
    first, second = manifest.rule_trace
    assert first.label == f"rule:{event.rule_id}"
    assert first.note == event.rule_version
    assert first.measurements == event.thresholds
    assert second.label == "confirmed"
    assert second.measurements == event.measurements
    # Ordered trace.
    assert (first.index, second.index) == (0, 1)


def test_manifest_is_deterministic_and_pure() -> None:
    # Two equal events yield byte-identical manifests (no randomness, no clock).
    a = build_evidence_manifest(_event())
    b = build_evidence_manifest(_event())
    assert a == b
    assert a.model_dump_json() == b.model_dump_json()


def test_manifest_carries_confidence_free_minimalism() -> None:
    # An event with a populated confidence breakdown does not force any manifest
    # field the stub does not model; the stub stays minimal regardless.
    event = _event(confidence=ConfidenceBreakdown(aggregate=0.9))
    manifest = build_evidence_manifest(event)
    assert manifest.event_id == event.event_id

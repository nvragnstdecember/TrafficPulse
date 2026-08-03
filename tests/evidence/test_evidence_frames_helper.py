"""Reading the evidence frame times a manifest already carries (H14)."""

from __future__ import annotations

from datetime import UTC, datetime

from trafficpulse.contracts import ConfirmedEvent, EvidenceManifest, MeasuredValue
from trafficpulse.contracts.enums import ArtifactKind, ViolationType
from trafficpulse.contracts.evidence import ArtifactReference, RuleTraceStep
from trafficpulse.evidence.frames import (
    evidence_frame_references,
    evidence_frame_times,
    media_time_measurement_name,
)
from trafficpulse.persistence.evidence_stub import build_evidence_manifest

_AT = datetime(1970, 1, 1, 0, 0, 9, tzinfo=UTC)


def _event() -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id="evt-1",
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-1",
        start_at=_AT,
        trigger_at=_AT,
        rule_id="rule-wrong-way",
        created_at=_AT,
    )


def _reference(kind: ArtifactKind, frame: str) -> ArtifactReference:
    return ArtifactReference(kind=kind, locator=f"frames/cam-1/{frame}")


def _manifest_with_frames(**times: float) -> EvidenceManifest:
    """A manifest shaped exactly as ``build_engine_manifest`` writes one."""

    slots = {
        "before": ArtifactKind.BEFORE_FRAME,
        "trigger": ArtifactKind.TRIGGER_FRAME,
        "after": ArtifactKind.AFTER_FRAME,
    }
    fields = {
        f"{name}_frame": _reference(kind, f"vfrm-{name}")
        for name, kind in slots.items()
        if name in times
    }
    step = RuleTraceStep(
        index=0,
        label="evidence-frames",
        measurements=tuple(
            MeasuredValue(
                name=media_time_measurement_name(slots[name]), value=seconds, unit="s"
            )
            for name, seconds in times.items()
        ),
    )
    return build_evidence_manifest(_event()).model_copy(
        update={**fields, "rule_trace": (step,)}
    )


def test_reads_every_declared_frame_time() -> None:
    manifest = _manifest_with_frames(before=8.84, trigger=9.84, after=10.01)
    assert evidence_frame_times(manifest) == {
        ArtifactKind.BEFORE_FRAME: 8.84,
        ArtifactKind.TRIGGER_FRAME: 9.84,
        ArtifactKind.AFTER_FRAME: 10.01,
    }


def test_a_missing_frame_slot_is_simply_absent() -> None:
    """A stream that started inside the margin has no before-frame; that is honest."""

    manifest = _manifest_with_frames(trigger=9.84, after=10.01)
    times = evidence_frame_times(manifest)
    assert ArtifactKind.BEFORE_FRAME not in times
    assert set(times) == {ArtifactKind.TRIGGER_FRAME, ArtifactKind.AFTER_FRAME}


def test_a_pre_engine_stub_manifest_declares_no_frame_times() -> None:
    """The P1-U11 shape references a frame nobody processed, so nothing is rendered."""

    stub = build_evidence_manifest(_event())
    assert stub.trigger_frame is not None  # it has a reference ...
    assert evidence_frame_times(stub) == {}  # ... but no recorded media time


def test_references_are_reported_independently_of_times() -> None:
    manifest = _manifest_with_frames(trigger=9.84)
    references = evidence_frame_references(manifest)
    assert set(references) == {ArtifactKind.TRIGGER_FRAME}
    assert references[ArtifactKind.TRIGGER_FRAME].locator == "frames/cam-1/vfrm-trigger"


def test_a_negative_recorded_time_is_rejected() -> None:
    """Media time is non-negative; a negative value is corrupt, not a position."""

    manifest = _manifest_with_frames(trigger=-1.0)
    assert evidence_frame_times(manifest) == {}


def test_measurement_name_matches_the_engine_convention() -> None:
    # Pins the exact names build_engine_manifest writes (and H11-H13 repositories hold).
    assert media_time_measurement_name(ArtifactKind.TRIGGER_FRAME) == "trigger_frame_media_time"
    assert media_time_measurement_name(ArtifactKind.BEFORE_FRAME) == "before_frame_media_time"
    assert media_time_measurement_name(ArtifactKind.AFTER_FRAME) == "after_frame_media_time"

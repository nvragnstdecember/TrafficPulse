"""EventStore.persist_pairs: caller-built manifests behind write-once semantics (H6)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trafficpulse.contracts import ConfirmedEvent, EvidenceManifest
from trafficpulse.contracts.enums import ArtifactKind, ViolationType
from trafficpulse.contracts.evidence import ArtifactReference
from trafficpulse.persistence import EventStore
from trafficpulse.persistence.errors import EventConflictError
from trafficpulse.persistence.evidence_stub import build_evidence_manifest


def _event(event_id: str) -> ConfirmedEvent:
    at = datetime(1970, 1, 1, 0, 0, 5, tzinfo=UTC)
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-a",
        start_at=at,
        trigger_at=at,
        rule_id="wrong_way",
        created_at=at,
    )


def _rich_manifest(event: ConfirmedEvent) -> EvidenceManifest:
    return build_evidence_manifest(event).model_copy(
        update={
            "before_frame": ArtifactReference(
                kind=ArtifactKind.BEFORE_FRAME, locator="frames/cam-a/vfrm-1"
            )
        }
    )


def test_persist_pairs_stores_the_callers_manifest(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event("evt-1")
    manifest = _rich_manifest(event)
    stored = store.persist_pairs("run-1", [(event, manifest)])
    assert stored[0].manifest == manifest
    reloaded = store.load("run-1")
    assert reloaded[0].manifest.before_frame is not None  # richer than the stub


def test_persist_pairs_rejects_a_mismatched_pair(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event("evt-1")
    wrong_manifest = build_evidence_manifest(_event("evt-2"))
    with pytest.raises(ValueError, match="paired event"):
        store.persist_pairs("run-1", [(event, wrong_manifest)])


def test_persist_pairs_is_write_once(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event("evt-1")
    manifest = _rich_manifest(event)
    store.persist_pairs("run-1", [(event, manifest)])
    # identical replay: idempotent no-op
    store.persist_pairs("run-1", [(event, manifest)])
    # a differing manifest under the same id: refused
    with pytest.raises(EventConflictError):
        store.persist_pairs("run-1", [(event, build_evidence_manifest(event))])


def test_persist_still_builds_stub_manifests(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event("evt-1")
    stored = store.persist("run-1", [event])
    assert stored[0].manifest == build_evidence_manifest(event)  # delegation unchanged

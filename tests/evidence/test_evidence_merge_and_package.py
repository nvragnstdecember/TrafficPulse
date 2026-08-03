"""Read-time manifest merging + deterministic package assembly (H14)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from trafficpulse.contracts import ConfirmedEvent, EvidenceManifest
from trafficpulse.contracts.enums import ArtifactKind, ViolationType
from trafficpulse.contracts.evidence import ArtifactReference
from trafficpulse.evidence.artifacts import ArtifactStore, artifact_sha256
from trafficpulse.evidence.merge import merge_rendered_artifacts, rendered_artifact_for
from trafficpulse.evidence.package import build_evidence_package, evidence_package_filename
from trafficpulse.persistence.evidence_stub import build_evidence_manifest

_AT = datetime(1970, 1, 1, 0, 0, 9, tzinfo=UTC)


def _event(event_id: str = "evt-1") -> ConfirmedEvent:
    return ConfirmedEvent(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-1",
        start_at=_AT,
        trigger_at=_AT,
        rule_id="rule-wrong-way",
        created_at=_AT,
    )


def _persisted_manifest() -> EvidenceManifest:
    """A manifest exactly as a run wrote it: a reference, no hash."""

    return build_evidence_manifest(_event())


def _rendered(store: ArtifactStore, kind: ArtifactKind, payload: bytes) -> ArtifactReference:
    return store.put(payload, kind=kind, media_type="image/png")


# --- merge ---------------------------------------------------------------------------
def test_merge_replaces_the_frame_slot_with_a_fetchable_reference(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = _persisted_manifest()
    assert manifest.trigger_frame is not None
    original = manifest.trigger_frame.locator

    trigger = _rendered(store, ArtifactKind.TRIGGER_FRAME, b"trigger-pixels")
    merged = merge_rendered_artifacts(manifest, [trigger])

    assert merged.trigger_frame == trigger
    assert merged.trigger_frame is not None
    assert merged.trigger_frame.sha256 is not None
    # The input manifest is untouched -- the merge returns a copy.
    assert manifest.trigger_frame.locator == original
    assert manifest.trigger_frame.sha256 is None


def test_merge_also_exposes_every_rendered_artifact_in_additional(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    before = _rendered(store, ArtifactKind.BEFORE_FRAME, b"before")
    trigger = _rendered(store, ArtifactKind.TRIGGER_FRAME, b"trigger")
    merged = merge_rendered_artifacts(_persisted_manifest(), [before, trigger])

    assert set(merged.additional_artifacts) == {before, trigger}
    assert merged.before_frame == before
    assert merged.trigger_frame == trigger


def test_merging_nothing_returns_the_manifest_unchanged() -> None:
    """A pre-H14 repository serves byte-identically to how it always did."""

    manifest = _persisted_manifest()
    merged = merge_rendered_artifacts(manifest, [])
    assert merged is manifest
    assert merged.model_dump_json() == manifest.model_dump_json()


def test_merge_is_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    trigger = _rendered(store, ArtifactKind.TRIGGER_FRAME, b"trigger")
    once = merge_rendered_artifacts(_persisted_manifest(), [trigger])
    twice = merge_rendered_artifacts(once, [trigger])
    assert twice.model_dump_json() == once.model_dump_json()


def test_rendered_artifact_lookup_ignores_unhashed_placeholders() -> None:
    """A reference with no hash names a frame nobody rendered; it is not fetchable."""

    manifest = _persisted_manifest()
    assert manifest.trigger_frame is not None
    assert rendered_artifact_for(manifest, ArtifactKind.TRIGGER_FRAME) is None


def test_rendered_artifact_lookup_finds_kinds_without_a_typed_slot(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    extra = store.put(b"clip-bytes", kind=ArtifactKind.CLIP, media_type="video/mp4")
    merged = merge_rendered_artifacts(_persisted_manifest(), [extra])
    assert rendered_artifact_for(merged, ArtifactKind.CLIP) == extra


# --- package -------------------------------------------------------------------------
def _package(tmp_path: Path) -> tuple[bytes, ArtifactReference, EvidenceManifest]:
    store = ArtifactStore(tmp_path)
    trigger = _rendered(store, ArtifactKind.TRIGGER_FRAME, b"trigger-pixels")
    before = _rendered(store, ArtifactKind.BEFORE_FRAME, b"before-pixels")
    manifest = merge_rendered_artifacts(_persisted_manifest(), [before, trigger])
    package = build_evidence_package(event=_event(), manifest=manifest, artifacts=store)
    return package, trigger, manifest


def test_package_contains_event_manifest_and_every_rendered_frame(tmp_path: Path) -> None:
    data, trigger, manifest = _package(tmp_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = sorted(archive.namelist())
        assert names == sorted(
            [
                "evt-1/event.json",
                "evt-1/manifest.json",
                f"evt-1/frames/{Path(trigger.locator).name}",
                f"evt-1/frames/{Path(manifest.before_frame.locator).name}",  # type: ignore[union-attr]
            ]
        )
        assert json.loads(archive.read("evt-1/event.json"))["event_id"] == "evt-1"


def test_every_packaged_reference_resolves_inside_the_archive(tmp_path: Path) -> None:
    """The property that makes a package independently verifiable, offline."""

    data, _, _ = _package(tmp_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("evt-1/manifest.json"))
        for slot in ("before_frame", "trigger_frame"):
            reference = manifest[slot]
            member = f"evt-1/frames/{Path(reference['locator']).name}"
            assert member in archive.namelist()
            assert artifact_sha256(archive.read(member)) == reference["sha256"]


def test_packages_are_byte_identical_across_builds(tmp_path: Path) -> None:
    first, _, _ = _package(tmp_path)
    second, _, _ = _package(tmp_path)
    assert first == second


def test_package_timestamps_are_fixed_not_wall_clock(tmp_path: Path) -> None:
    data, _, _ = _package(tmp_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_a_package_without_rendered_frames_is_still_the_full_record(tmp_path: Path) -> None:
    """An event nobody rendered still packages what the system concluded."""

    store = ArtifactStore(tmp_path)
    data = build_evidence_package(
        event=_event(), manifest=_persisted_manifest(), artifacts=store
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert sorted(archive.namelist()) == ["evt-1/event.json", "evt-1/manifest.json"]


def test_a_reference_whose_bytes_are_gone_is_skipped_not_faked(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    trigger = _rendered(store, ArtifactKind.TRIGGER_FRAME, b"trigger-pixels")
    store.path_for(trigger.locator).unlink()

    manifest = merge_rendered_artifacts(_persisted_manifest(), [trigger])
    data = build_evidence_package(event=_event(), manifest=manifest, artifacts=store)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert not [n for n in archive.namelist() if n.startswith("evt-1/frames/")]


def test_package_filename_is_derived_from_the_event() -> None:
    assert evidence_package_filename("evt-abc") == "evidence-evt-abc.zip"

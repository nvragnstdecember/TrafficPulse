"""Content-addressed artifact storage + the append-only rendered sidecar (H14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trafficpulse.contracts.enums import ArtifactKind
from trafficpulse.contracts.evidence import ArtifactReference
from trafficpulse.evidence.artifacts import ArtifactStore, artifact_sha256
from trafficpulse.persistence.errors import CorruptRecordError
from trafficpulse.persistence.rendered_store import RenderedArtifactStore

_PNG = b"\x89PNG\r\n\x1a\npretend-pixels"


# --- artifact store -----------------------------------------------------------------
def test_put_addresses_an_artifact_by_its_own_hash(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    reference = store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")

    digest = artifact_sha256(_PNG)
    assert reference.sha256 == digest
    assert reference.locator == f"artifacts/{digest[:2]}/{digest}.png"
    assert reference.media_type == "image/png"
    assert store.read(reference.locator) == _PNG


def test_storing_identical_bytes_twice_is_idempotent(tmp_path: Path) -> None:
    """Re-rendering an unchanged event must cost a hash, not a second copy."""

    store = ArtifactStore(tmp_path)
    first = store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    written = store.path_for(first.locator).stat().st_mtime_ns

    second = store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    assert second == first
    assert store.path_for(second.locator).stat().st_mtime_ns == written  # never rewritten
    assert len(list(tmp_path.rglob("*.png"))) == 1


def test_two_kinds_of_the_same_frame_share_one_file(tmp_path: Path) -> None:
    """Content addressing dedups across events whose evidence windows overlap."""

    store = ArtifactStore(tmp_path)
    before = store.put(_PNG, kind=ArtifactKind.BEFORE_FRAME, media_type="image/png")
    trigger = store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    assert before.locator == trigger.locator
    assert before.kind is not trigger.kind
    assert len(list(tmp_path.rglob("*.png"))) == 1


def test_verify_detects_tampering(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    reference = store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    assert store.verify(reference)

    store.path_for(reference.locator).write_bytes(b"different")
    assert not store.verify(reference)


def test_an_unhashed_reference_is_never_reported_as_verified(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    unhashed = ArtifactReference(kind=ArtifactKind.TRIGGER_FRAME, locator="frames/cam/vfrm-1")
    assert not store.verify(unhashed)


def test_missing_artifacts_read_as_none(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    assert store.read("artifacts/ab/abcdef.png") is None
    assert not store.contains("artifacts/ab/abcdef.png")


@pytest.mark.parametrize("locator", ["../escape.png", "/etc/passwd", "artifacts/../../x"])
def test_locators_cannot_escape_the_store_root(tmp_path: Path, locator: str) -> None:
    """Locators reach this method from persisted records and from client requests."""

    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe artifact locator"):
        store.path_for(locator)
    assert store.read(locator) is None
    assert not store.contains(locator)


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.put(_PNG, kind=ArtifactKind.TRIGGER_FRAME, media_type="image/png")
    assert not list(tmp_path.rglob("*.partial"))


# --- rendered-artifact sidecar ------------------------------------------------------
def _reference(kind: ArtifactKind, digest: str) -> ArtifactReference:
    return ArtifactReference(
        kind=kind, locator=f"artifacts/{digest[:2]}/{digest}.png",
        sha256=digest, media_type="image/png",
    )


_A = "a" * 64
_B = "b" * 64


def test_record_then_read_round_trips(tmp_path: Path) -> None:
    store = RenderedArtifactStore(tmp_path)
    trigger = _reference(ArtifactKind.TRIGGER_FRAME, _A)
    assert store.record("evt-1", [trigger]) == (trigger,)
    assert store.artifacts("evt-1") == (trigger,)
    assert store.artifact("evt-1", ArtifactKind.TRIGGER_FRAME) == trigger


def test_recording_the_same_artifact_twice_reads_back_once(tmp_path: Path) -> None:
    """Repeated rendering is idempotent from the reader's point of view."""

    store = RenderedArtifactStore(tmp_path)
    trigger = _reference(ArtifactKind.TRIGGER_FRAME, _A)
    store.record("evt-1", [trigger])
    store.record("evt-1", [trigger])
    assert store.artifacts("evt-1") == (trigger,)


def test_the_journal_is_append_only(tmp_path: Path) -> None:
    """A later render adds to the record; it never replaces what came before."""

    store = RenderedArtifactStore(tmp_path)
    first = _reference(ArtifactKind.TRIGGER_FRAME, _A)
    second = _reference(ArtifactKind.AFTER_FRAME, _B)
    store.record("evt-1", [first])
    store.record("evt-1", [second])
    assert store.artifacts("evt-1") == (first, second)
    assert len(store.journal_path("evt-1").read_text().splitlines()) == 2


def test_an_event_with_no_journal_has_no_artifacts(tmp_path: Path) -> None:
    """The pre-H14 repository shape: nothing rendered, nothing claimed."""

    store = RenderedArtifactStore(tmp_path)
    assert store.artifacts("evt-unknown") == ()
    assert store.artifact("evt-unknown", ArtifactKind.TRIGGER_FRAME) is None
    assert store.rendered_event_ids() == frozenset()


def test_recording_nothing_is_a_no_op(tmp_path: Path) -> None:
    store = RenderedArtifactStore(tmp_path)
    assert store.record("evt-1", []) == ()
    assert not store.journal_path("evt-1").exists()


def test_events_are_isolated_from_each_other(tmp_path: Path) -> None:
    store = RenderedArtifactStore(tmp_path)
    store.record("evt-1", [_reference(ArtifactKind.TRIGGER_FRAME, _A)])
    store.record("evt-2", [_reference(ArtifactKind.TRIGGER_FRAME, _B)])
    assert store.artifacts("evt-1")[0].sha256 == _A
    assert store.artifacts("evt-2")[0].sha256 == _B
    assert store.rendered_event_ids() == {"evt-1", "evt-2"}


def test_a_corrupt_journal_line_is_reported_not_ignored(tmp_path: Path) -> None:
    store = RenderedArtifactStore(tmp_path)
    store.record("evt-1", [_reference(ArtifactKind.TRIGGER_FRAME, _A)])
    with store.journal_path("evt-1").open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    with pytest.raises(CorruptRecordError):
        store.artifacts("evt-1")


def test_the_sidecar_writes_only_under_its_own_subtree(tmp_path: Path) -> None:
    """The invariant that keeps it incapable of touching a write-once record."""

    store = RenderedArtifactStore(tmp_path)
    store.record("evt-1", [_reference(ArtifactKind.TRIGGER_FRAME, _A)])
    written = {
        path.relative_to(tmp_path).parts[0]
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert written == {"rendered"}

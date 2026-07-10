"""P1-U11 event-store tests: persist/reload round-trip, determinism, idempotency.

All persistence targets a pytest ``tmp_path``; nothing is written into a tracked
repository path. Covers round-trip equality, deterministic replay, write-once
idempotency vs conflict, distinct-event separation, provenance survival, the typed
corruption/missing-run boundary, no input mutation, and gitignore coverage of the
default output location.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trafficpulse.contracts import (
    ConfirmedEvent,
    MeasuredValue,
    ModelRef,
    ViolationType,
)
from trafficpulse.persistence import (
    DEFAULT_RUN_ROOT,
    CorruptRecordError,
    EventConflictError,
    EventStore,
    RunNotFoundError,
    StoredEvent,
    build_evidence_manifest,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TS2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _event(event_id: str = "evt-abc123", **overrides: object) -> ConfirmedEvent:
    base: dict[str, object] = dict(
        event_id=event_id,
        violation_type=ViolationType.WRONG_WAY,
        camera_id="cam-1",
        track_ids=("t1", "t2"),
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


# --- persistence + round-trip ------------------------------------------------
def test_persist_returns_stored_pairs(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    stored = store.persist("run-1", [event])
    assert len(stored) == 1
    assert isinstance(stored[0], StoredEvent)
    assert stored[0].event == event
    assert stored[0].manifest == build_evidence_manifest(event)


def test_persist_reload_preserves_event_equality(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    store.persist("run-1", [event])
    reloaded = store.load("run-1")
    assert len(reloaded) == 1
    assert reloaded[0].event == event  # semantic equality on the frozen contract


def test_persist_reload_preserves_manifest_and_linkage(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    store.persist("run-1", [event])
    reloaded = store.load("run-1")[0]
    assert reloaded.manifest == build_evidence_manifest(event)
    assert reloaded.manifest.event_id == reloaded.event.event_id
    assert reloaded.manifest.evidence_package_id == f"evp-{event.event_id}"


def test_files_are_laid_out_by_event_id(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    store.persist("run-1", [event])
    run_dir = tmp_path / "run-1"
    assert (run_dir / "events" / f"{event.event_id}.json").is_file()
    assert (run_dir / "manifests" / f"{event.event_id}.json").is_file()


# --- provenance / field survival ---------------------------------------------
def test_provenance_survives_persist_reload(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event(
        code_version="deadbeef",
        models=(ModelRef(name="rtdetr", version="r50", weights_hash="c" * 64),),
    )
    store.persist("run-1", [event])
    reloaded = store.load("run-1")[0].event
    assert reloaded.code_version == "deadbeef"
    assert reloaded.models == event.models
    assert reloaded.rule_id == "wrong_way"
    assert reloaded.rule_version == "0.1.0-provisional"


def test_scene_config_hash_survives(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event(scene_config_hash="f" * 64)
    store.persist("run-1", [event])
    assert store.load("run-1")[0].event.scene_config_hash == "f" * 64
    assert store.load("run-1")[0].manifest.scene_config_hash == "f" * 64


def test_track_ids_survive(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event(track_ids=("left", "right"))
    store.persist("run-1", [event])
    assert store.load("run-1")[0].event.track_ids == ("left", "right")


def test_violation_type_survives(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event()])
    assert store.load("run-1")[0].event.violation_type is ViolationType.WRONG_WAY


# --- determinism / idempotency -----------------------------------------------
def test_repeated_persist_is_byte_identical(tmp_path: Path) -> None:
    event = _event()
    a = tmp_path / "a"
    b = tmp_path / "b"
    EventStore(a).persist("run-1", [event])
    EventStore(b).persist("run-1", [event])
    event_file = Path("run-1") / "events" / f"{event.event_id}.json"
    manifest_file = Path("run-1") / "manifests" / f"{event.event_id}.json"
    assert (a / event_file).read_bytes() == (b / event_file).read_bytes()
    assert (a / manifest_file).read_bytes() == (b / manifest_file).read_bytes()


def test_reload_is_deterministic(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event()])
    assert store.load("run-1") == store.load("run-1")


def test_identical_replay_into_same_run_is_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    store.persist("run-1", [event])
    # Re-persisting byte-identical content is a no-op, not a duplicate or error.
    store.persist("run-1", [event])
    reloaded = store.load("run-1")
    assert len(reloaded) == 1
    assert reloaded[0].event == event


def test_differing_content_under_same_id_raises_conflict(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event(camera_id="cam-1")])
    # Same event_id, different payload -> refuse silent overwrite (ADR-004).
    with pytest.raises(EventConflictError):
        store.persist("run-1", [_event(camera_id="cam-CHANGED")])


def test_multiple_distinct_events_remain_distinct(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    e1 = _event("evt-001", camera_id="cam-a")
    e2 = _event("evt-002", camera_id="cam-b")
    e3 = _event("evt-003", camera_id="cam-c")
    store.persist("run-1", [e2, e3, e1])  # unordered input
    reloaded = store.load("run-1")
    assert [s.event.event_id for s in reloaded] == ["evt-001", "evt-002", "evt-003"]
    assert {s.event for s in reloaded} == {e1, e2, e3}


def test_distinct_runs_are_independent(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event("evt-001")])
    store.persist("run-2", [_event("evt-999", camera_id="other")])
    assert [s.event.event_id for s in store.load("run-1")] == ["evt-001"]
    assert [s.event.event_id for s in store.load("run-2")] == ["evt-999"]


# --- no input mutation --------------------------------------------------------
def test_persist_does_not_mutate_input_event(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    snapshot = event.model_dump_json()
    store.persist("run-1", [event])
    assert event.model_dump_json() == snapshot  # frozen contract untouched


# --- typed error boundary -----------------------------------------------------
def test_load_missing_run_raises_run_not_found(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    with pytest.raises(RunNotFoundError):
        store.load("never-persisted")


def test_load_corrupt_event_file_raises(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event()])
    corrupt = next((tmp_path / "run-1" / "events").glob("*.json"))
    corrupt.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CorruptRecordError):
        store.load("run-1")


def test_load_event_validation_failure_raises(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.persist("run-1", [_event()])
    corrupt = next((tmp_path / "run-1" / "events").glob("*.json"))
    corrupt.write_text('{"event_id": "x"}', encoding="utf-8")  # valid JSON, invalid contract
    with pytest.raises(CorruptRecordError):
        store.load("run-1")


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = _event()
    store.persist("run-1", [event])
    (tmp_path / "run-1" / "manifests" / f"{event.event_id}.json").unlink()
    with pytest.raises(CorruptRecordError):
        store.load("run-1")


# --- gitignore coverage of the default output location -----------------------
def test_default_run_root_is_gitignored() -> None:
    assert Path("runs") == DEFAULT_RUN_ROOT
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/runs/" in gitignore

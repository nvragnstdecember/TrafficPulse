"""Minimal deterministic event + evidence persistence (P1-U11).

Persists a run's confirmed events and their minimal evidence manifests to
deterministic JSON files, and reloads them without semantic loss. This is the
smallest storage posture the P1-U11 card authorises: ADR-002 names SQLite for the
eventual event store but adds *no dependency now* ("SQLAlchemy/PyArrow become
Phase 1 runtime dependencies **when the storage runtime is built** -- none are
added now"), and the card's stop-condition/fallback both default to deterministic
JSON files for this slice. So this layer adds **no** dependency: it uses
``pydantic``'s ``model_dump_json`` / ``model_validate_json`` (already a base
dependency) and the standard library only.

Layout (per run, all under a gitignored runtime root)
-----------------------------------------------------
```
<root>/<run_id>/events/<event_id>.json      # ConfirmedEvent.model_dump_json()
<root>/<run_id>/manifests/<event_id>.json   # EvidenceManifest.model_dump_json()
```
Both files are keyed by ``event_id`` so the event and its manifest pair
trivially on reload. :data:`DEFAULT_RUN_ROOT` (``runs/``) is the documented
default output root and is covered by ``.gitignore`` (``/runs/``); tests pass an
explicit temporary root, so nothing is written into a tracked path.

Determinism and replay (ADR-004-aware, nothing frozen)
------------------------------------------------------
``event_id`` is minted upstream by the reasoner as a content-derived hash, and
the manifest is a pure function of the event, so an identical replay produces
byte-identical files. Writes are **write-once per ``(run_id, event_id)``**:
re-persisting byte-identical content is an idempotent no-op, while an attempt to
write *differing* content under an existing id raises
:class:`~trafficpulse.persistence.errors.EventConflictError`. That honours
ADR-004's *proposed* "manifests are append-only; no run silently overwrites
another" while deciding **no** cross-run identity or deduplication rule (ADR-004
stays Proposed). Distinct runs live in distinct directories and never interact.

Boundary
--------
Persistence depends only on the frozen U2 contracts and the evidence stub. It
imports **no** detector or tracker backend, and **no** ML framework: an event is
just data by the time it reaches this layer.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from ..contracts import ConfirmedEvent, EvidenceManifest
from ..contracts.primitives import ContractModel
from .errors import CorruptRecordError, EventConflictError, RunNotFoundError
from .evidence_stub import build_evidence_manifest

# Bound to the frozen contract base so the reload helper stays generic over the
# two record kinds while returning the exact type it was asked to parse. Kept a
# classic ``TypeVar`` (not PEP 695 syntax) to preserve the project's Python 3.11
# floor (mypy target py311; requires-python >= 3.11).
_C = TypeVar("_C", bound=ContractModel)

# Documented default output root. Root-anchored ``/runs/`` is gitignored so
# persisted runs never enter version control. Callers may pass an explicit root
# (tests always use a temporary directory).
DEFAULT_RUN_ROOT = Path("runs")

_EVENTS_DIR = "events"
_MANIFESTS_DIR = "manifests"


@dataclass(frozen=True)
class StoredEvent:
    """A persisted event paired with its evidence manifest (a reload grouping).

    Not a domain contract -- just the immutable pairing returned by the store so
    the deterministic ``event_id`` <-> manifest linkage is explicit in the return
    type. ``manifest.event_id == event.event_id`` always holds.
    """

    event: ConfirmedEvent
    manifest: EvidenceManifest


class EventStore:
    """Deterministic JSON-file store for confirmed events + evidence manifests.

    Constructed with a runtime output ``root`` (defaults to :data:`DEFAULT_RUN_ROOT`).
    :meth:`persist` writes a run's events and their built manifests; :meth:`load`
    reloads them into equal contracts. The store holds no mutable per-run state --
    it is a thin, deterministic filesystem adapter.
    """

    def __init__(self, root: Path | str = DEFAULT_RUN_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The runtime output root this store writes under."""

        return self._root

    def run_dir(self, run_id: str) -> Path:
        """The directory holding one run's persisted records."""

        return self._root / run_id

    def persist(
        self, run_id: str, events: Iterable[ConfirmedEvent]
    ) -> tuple[StoredEvent, ...]:
        """Persist a run's events + minimal manifests; return the stored pairs.

        For each event a minimal :class:`EvidenceManifest` is built (via the
        evidence stub) and both are written as deterministic JSON keyed by
        ``event_id``. Events are processed in ``event_id`` order so the operation
        is order-independent. Write-once semantics apply per ``(run_id,
        event_id)`` (see module docstring): identical replay is an idempotent
        no-op; a differing write raises
        :class:`~trafficpulse.persistence.errors.EventConflictError`. The input
        events are never mutated (they are frozen contracts, only read).
        """

        events_dir = self.run_dir(run_id) / _EVENTS_DIR
        manifests_dir = self.run_dir(run_id) / _MANIFESTS_DIR
        stored: list[StoredEvent] = []
        for event in sorted(events, key=lambda e: e.event_id):
            manifest = build_evidence_manifest(event)
            self._write_once(
                events_dir / f"{event.event_id}.json",
                event.model_dump_json(),
            )
            self._write_once(
                manifests_dir / f"{event.event_id}.json",
                manifest.model_dump_json(),
            )
            stored.append(StoredEvent(event=event, manifest=manifest))
        return tuple(stored)

    def load(self, run_id: str) -> tuple[StoredEvent, ...]:
        """Reload a run's stored events + manifests into equal frozen contracts.

        Returns the stored pairs in ``event_id`` order (deterministic). Raises
        :class:`RunNotFoundError` if the run was never persisted, and
        :class:`CorruptRecordError` if a stored file is missing its sibling
        manifest or cannot be validated back into its U2 contract.
        """

        run_dir = self.run_dir(run_id)
        if not run_dir.is_dir():
            raise RunNotFoundError(f"no persisted run {run_id!r} under {self._root}")
        events_dir = run_dir / _EVENTS_DIR
        manifests_dir = run_dir / _MANIFESTS_DIR
        stored: list[StoredEvent] = []
        for event_file in sorted(events_dir.glob("*.json")):
            event = self._load_contract(ConfirmedEvent, event_file)
            manifest_file = manifests_dir / event_file.name
            if not manifest_file.is_file():
                raise CorruptRecordError(
                    f"event {event_file.name!r} in run {run_id!r} has no evidence manifest"
                )
            manifest = self._load_contract(EvidenceManifest, manifest_file)
            stored.append(StoredEvent(event=event, manifest=manifest))
        return tuple(stored)

    @staticmethod
    def _write_once(path: Path, payload: str) -> None:
        data = payload.encode("utf-8")
        if path.exists():
            if path.read_bytes() == data:
                return  # idempotent replay: identical content already persisted
            raise EventConflictError(
                f"{path} already holds differing content; refusing to overwrite "
                "(ADR-004 append-only: no run silently overwrites another)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def _load_contract(contract: type[_C], path: Path) -> _C:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - unreadable file is environmental
            raise CorruptRecordError(f"cannot read persisted record {path}") from exc
        try:
            return contract.model_validate_json(raw)
        except ValidationError as exc:
            raise CorruptRecordError(
                f"persisted record {path} is not a valid {contract.__name__}"
            ) from exc

"""Minimal event persistence + evidence stub for TrafficPulse (Phase 1, P1-U11).

Makes a confirmed event **reviewable**: it persists each frozen U2
:class:`~trafficpulse.contracts.ConfirmedEvent` produced by the P1-U10 pipeline
together with a minimal, provenance-bearing
:class:`~trafficpulse.contracts.EvidenceManifest`, and reloads them without
semantic loss -- deterministically and fully offline.

```
ConfirmedEvent (P1-U10)
  -> build_evidence_manifest        -> minimal EvidenceManifest (id linkage +
                                       trigger-frame reference + rule trace +
                                       carried provenance)
  -> EventStore.persist(run_id, ..) -> deterministic JSON files under runs/<id>/
  -> EventStore.load(run_id)        -> (ConfirmedEvent, EvidenceManifest) pairs
```

This layer is the smallest storage posture the P1-U11 card authorises: it adds
**no** dependency (ADR-002 defers the SQLite runtime; the card defaults to
deterministic JSON for this slice), imports **no** detector/tracker backend and
**no** ML framework, and freezes **no** ADR-004 cross-run identity/dedup rule
(writes are only write-once-per-run, honouring "no silent overwrite"). It does
not render frames or clips, do OCR, or build the review/penalty workflow -- those
are later units.
"""

from .errors import (
    CorruptRecordError,
    EventConflictError,
    PersistenceError,
    RunNotFoundError,
)
from .evidence_stub import (
    build_evidence_manifest,
    evidence_package_id_for,
    trigger_frame_locator_for,
)
from .store import DEFAULT_RUN_ROOT, EventStore, StoredEvent

__all__ = [
    # store
    "EventStore",
    "StoredEvent",
    "DEFAULT_RUN_ROOT",
    # evidence stub
    "build_evidence_manifest",
    "evidence_package_id_for",
    "trigger_frame_locator_for",
    # errors
    "PersistenceError",
    "RunNotFoundError",
    "CorruptRecordError",
    "EventConflictError",
]

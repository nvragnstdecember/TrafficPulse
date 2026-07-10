"""Minimal, provenance-bearing evidence-manifest stub (P1-U11).

Builds a minimal :class:`~trafficpulse.contracts.EvidenceManifest` from a single
frozen :class:`~trafficpulse.contracts.ConfirmedEvent` -- the "evidence stub" the
P1-U11 card calls for: enough to make a confirmed event *reviewable* (id linkage +
a trigger-frame reference + a short rule trace + carried provenance), and nothing
of the future evidence engine (no clip/frame rendering, no crops, no overlays, no
content-addressed media hashing, no OCR).

Pure and deterministic
----------------------
:func:`build_evidence_manifest` is a pure function of the event alone. It reads
**no** wall-clock and generates **no** randomness: ``created_at`` is copied from
the event's own deterministic ``created_at`` (itself the reasoner's data
timestamp, never wall-clock), so re-deriving a manifest for an equal event yields
an equal manifest. This is what lets the whole persist path be byte-deterministic
across replays.

Honest provenance (nothing fabricated)
--------------------------------------
Only provenance already present on the event is carried: ``scene_config_hash``,
``rule_id`` / ``rule_version`` (into the rule trace), ``code_version`` and
``models``. In the current slice the U4 reasoner does not stamp detector/tracker
``ModelRef``s onto ``ConfirmedEvent.models`` (it defaults to ``()``), so the
manifest's ``models`` is honestly empty rather than invented; enriching that is a
future concern for the reasoner/pipeline, not for this stub to fabricate.

The trigger-frame reference is a **relative locator only** -- it points at where a
trigger frame *would* live, deterministically keyed by event identity, and carries
**no** ``sha256`` because nothing has been rendered or hashed. Per the
``ArtifactReference`` contract, a locator without an integrity hash is exactly "a
reference we have not hashed", which is the truthful shape of a stub.
"""

from ..contracts import ConfirmedEvent, EvidenceManifest
from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference, RuleTraceStep


def evidence_package_id_for(event: ConfirmedEvent) -> str:
    """Deterministic evidence-package id for an event (``"evp-" + event_id``).

    A pure function of the event's own id, so the linkage
    ``ConfirmedEvent.event_id`` <-> ``EvidenceManifest.evidence_package_id`` is
    explicit, reproducible, and requires no external state.
    """

    return f"evp-{event.event_id}"


def trigger_frame_locator_for(event: ConfirmedEvent) -> str:
    """Deterministic *relative* locator for the trigger frame reference.

    Encodes only event identity (``camera_id`` + ``event_id``); it is an opaque
    relative path, not a claim that a rendered artifact exists at it.
    """

    return f"frames/{event.camera_id}/{event.event_id}/trigger"


def build_evidence_manifest(event: ConfirmedEvent) -> EvidenceManifest:
    """Build a minimal, provenance-bearing ``EvidenceManifest`` for one event.

    Deterministic and pure (see module docstring). The manifest links back to the
    event by ``event_id`` and a derived ``evidence_package_id``; references the
    trigger frame by a relative locator (no rendered artifact, no hash); and
    carries a short ``rule_trace`` plus the provenance already present on the
    event. It fabricates nothing the event does not provide.
    """

    rule_trace = (
        RuleTraceStep(
            index=0,
            label=f"rule:{event.rule_id}",
            note=event.rule_version,
            measurements=event.thresholds,
        ),
        RuleTraceStep(
            index=1,
            label="confirmed",
            measurements=event.measurements,
        ),
    )
    return EvidenceManifest(
        evidence_package_id=evidence_package_id_for(event),
        event_id=event.event_id,
        trigger_frame=ArtifactReference(
            kind=ArtifactKind.TRIGGER_FRAME,
            locator=trigger_frame_locator_for(event),
        ),
        rule_trace=rule_trace,
        models=event.models,
        code_version=event.code_version,
        scene_config_hash=event.scene_config_hash,
        created_at=event.created_at,
    )

"""Rendering a run's evidence frames into stored artifacts (H14).

The orchestration layer of the evidence rendering engine: it decides *which* frames
an event's evidence needs, renders them through the shared overlay renderer, stores
them by content address, and records the resulting references in the append-only
sidecar. It draws nothing itself and stores no bytes itself -- it composes
:mod:`trafficpulse.overlay.frame`,
:mod:`trafficpulse.evidence.artifacts`, and
:class:`~trafficpulse.persistence.rendered_store.RenderedArtifactStore`.

One decode for a whole run
--------------------------
Every event's evidence frames are collected into a **single** set of media times and
satisfied by one sequential pass over the source. A run confirming twenty events
therefore decodes the clip once, not sixty times, and events whose evidence windows
overlap share both the decode and -- through content addressing -- the stored bytes.

Determinism and replay
----------------------
The media times come from the persisted manifest, the scene comes from metadata
inference already produced, and the artifact address is the SHA-256 of the encoded
bytes. Re-rendering an unchanged run therefore recomputes identical hashes, rewrites
no file, and appends references the sidecar collapses on read: repeated rendering is
observably idempotent.

Failure posture
---------------
Rendering is presentation, and it runs *after* events and manifests are durably
persisted. Every failure mode here -- an absent drawing backend, an unreadable
source, a manifest with no recorded frame times -- degrades to *fewer artifacts*,
never to a failed run and never to a corrupted record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..contracts import ConfirmedEvent, EvidenceManifest
from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference
from ..overlay.frame import render_stills_at
from ..overlay.registry import OverlayCompositor
from ..overlay.renderer import OverlayRenderer
from ..persistence.rendered_store import RenderedArtifactStore
from .artifacts import ArtifactStore
from .frames import evidence_frame_times

_logger = logging.getLogger("trafficpulse.evidence")


@dataclass(frozen=True)
class EvidenceRenderReport:
    """What one run's evidence render produced (metrics only, no bytes)."""

    events_rendered: int
    artifacts_written: int
    frames_decoded: int

    @property
    def is_empty(self) -> bool:
        return self.events_rendered == 0 and self.artifacts_written == 0


def render_run_evidence(
    *,
    pairs: Sequence[tuple[ConfirmedEvent, EvidenceManifest]],
    source_path: Path | str,
    camera_id: str,
    artifacts: ArtifactStore,
    rendered: RenderedArtifactStore,
    compositor: OverlayCompositor | None = None,
    renderer: OverlayRenderer | None = None,
    image_format: str = "png",
) -> EvidenceRenderReport:
    """Render + store the evidence frames for every event of one run.

    ``pairs`` are the persisted ``(event, manifest)`` records, exactly as
    ``EventStore`` returned them. Events whose manifest records no frame times (a
    pre-engine stub manifest) are skipped rather than rendered from a guessed
    position.
    """

    wanted: dict[str, dict[str, float]] = {}
    for event, manifest in pairs:
        times = evidence_frame_times(manifest)
        if times:
            wanted[event.event_id] = {kind.value: seconds for kind, seconds in times.items()}

    if not wanted:
        return EvidenceRenderReport(events_rendered=0, artifacts_written=0, frames_decoded=0)

    every_time = sorted({seconds for slots in wanted.values() for seconds in slots.values()})
    stills = render_stills_at(
        source_path=source_path,
        media_times=every_time,
        camera_id=camera_id,
        compositor=compositor,
        renderer=renderer,
        image_format=image_format,
    )
    if not stills:
        _logger.warning("evidence render decoded no frames from %s", source_path)
        return EvidenceRenderReport(events_rendered=0, artifacts_written=0, frames_decoded=0)

    events_rendered = 0
    written = 0
    for event_id, slots in sorted(wanted.items()):
        references: list[ArtifactReference] = []
        for kind_value, seconds in sorted(slots.items()):
            still = stills.get(seconds)
            if still is None:  # pragma: no cover - every requested time is rendered
                continue
            references.append(
                artifacts.put(
                    still.data,
                    kind=ArtifactKind(kind_value),
                    media_type=still.media_type,
                )
            )
        if references:
            rendered.record(event_id, references)
            events_rendered += 1
            written += len(references)

    return EvidenceRenderReport(
        events_rendered=events_rendered,
        artifacts_written=written,
        frames_decoded=len(stills),
    )

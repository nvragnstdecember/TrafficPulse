"""The production evidence rendering engine (H14).

Turns a confirmed event's **reference-only** evidence manifest into artifacts an
analyst can actually look at, download, and verify -- without altering a single
persisted record.

```
ConfirmedEvent + EvidenceManifest      (immutable, already persisted)
  -> evidence_frame_times              which source frames the manifest declares
  -> overlay.frame.render_stills_at    one decode, drawn by the shared renderer
  -> ArtifactStore.put                 content-addressed bytes + SHA-256
  -> RenderedArtifactStore.record      append-only sidecar of references
  -> merge_rendered_artifacts          served manifest, composed at read time
  -> build_evidence_package            deterministic ZIP for download
```

The three guarantees this layer is built around
-----------------------------------------------
* **Immutability** -- nothing here writes to ``events/`` or ``manifests/``. Rendered
  references live in their own append-only journal and are merged when served, so
  ``EventStore``'s write-once contract is untouched.
* **Determinism** -- frames are addressed by the media times the manifest already
  records (never by storage-dependent frame ids), artifacts are addressed by the
  SHA-256 of their bytes, and packages are built with fixed timestamps. Re-rendering
  an unchanged run rewrites nothing.
* **One renderer** -- stills are drawn by the same
  :class:`~trafficpulse.overlay.renderer.OverlayRenderer` that draws the annotated
  video, from the same :class:`~trafficpulse.overlay.registry.OverlayCompositor`
  metadata. A still and the video can never disagree about what the system concluded.

Degradation is always toward *less evidence*, never toward a failed run or a bad
record: rendering happens after persistence, and a missing drawing backend, an
unreadable source, or a pre-H14 manifest simply yields fewer artifacts.
"""

from .artifacts import ArtifactStore, artifact_locator, artifact_sha256
from .frames import (
    EVIDENCE_FRAME_KINDS,
    evidence_frame_references,
    evidence_frame_times,
)
from .merge import merge_rendered_artifacts, rendered_artifact_for
from .package import build_evidence_package, evidence_package_filename
from .renderer import EvidenceRenderReport, render_run_evidence

__all__ = [
    # frame addressing
    "evidence_frame_times",
    "evidence_frame_references",
    "EVIDENCE_FRAME_KINDS",
    # artifact storage
    "ArtifactStore",
    "artifact_sha256",
    "artifact_locator",
    # rendering
    "render_run_evidence",
    "EvidenceRenderReport",
    # serving
    "merge_rendered_artifacts",
    "rendered_artifact_for",
    "build_evidence_package",
    "evidence_package_filename",
]

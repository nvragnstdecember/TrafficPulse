"""Assembling a downloadable evidence package (H14).

Bundles everything a reviewer needs to inspect one confirmed event -- offline, and
independently of this system -- into a single deterministic ZIP:

```
<event_id>/event.json            the frozen ConfirmedEvent, verbatim
<event_id>/manifest.json         the evidence manifest as served (rendered refs merged)
<event_id>/frames/*.png          the rendered before/trigger/after stills
```

Deterministic bytes
-------------------
Two packages built from the same event and the same artifacts are byte-identical:
entries are written in sorted order, every entry carries a **fixed** timestamp
(ZIP's 1980 epoch) rather than the wall clock, and the JSON comes from pydantic's
deterministic ``model_dump_json``. So a package hash is a property of the evidence,
not of when somebody clicked download -- which is the only way a package hash can
mean anything in a review workflow.

The manifest inside the package is the **served** one -- rendered references merged
in -- so every ``locator`` in it resolves to a file inside the same archive, and
every ``sha256`` can be checked against those bytes with nothing but a standard
unzip and a hash tool.

Nothing is rendered here
------------------------
This module reads already-stored artifacts and serialises already-built contracts.
It decodes no video, draws nothing, and adds no dependency (``zipfile`` is standard
library).
"""

from __future__ import annotations

import posixpath
import zipfile
from io import BytesIO

from ..contracts import ConfirmedEvent, EvidenceManifest
from ..contracts.evidence import ArtifactReference
from .artifacts import ArtifactStore

#: Fixed ZIP entry timestamp (the format's own epoch), so packages are reproducible.
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

#: Directory inside the archive holding rendered frames.
_FRAMES_DIR = "frames"


def evidence_package_filename(event_id: str) -> str:
    """The download filename for one event's package."""

    return f"evidence-{event_id}.zip"


def build_evidence_package(
    *,
    event: ConfirmedEvent,
    manifest: EvidenceManifest,
    artifacts: ArtifactStore,
) -> bytes:
    """Build one event's evidence package as deterministic ZIP bytes.

    ``manifest`` must be the *served* manifest (rendered references already merged
    in), so the archive is internally consistent. Referenced artifacts that are not
    present in the store are skipped rather than faked -- a package containing two
    of three frames is honest about what exists; a package containing a zero-byte
    placeholder is not.
    """

    root = event.event_id
    members: dict[str, bytes] = {
        posixpath.join(root, "event.json"): event.model_dump_json(indent=2).encode("utf-8"),
        posixpath.join(root, "manifest.json"): manifest.model_dump_json(indent=2).encode(
            "utf-8"
        ),
    }

    for reference in _packaged_references(manifest):
        data = artifacts.read(reference.locator)
        if data is None:
            continue
        name = posixpath.basename(reference.locator)
        members[posixpath.join(root, _FRAMES_DIR, name)] = data

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            # A stable external_attr keeps the archive identical across platforms
            # (the default encodes the writing OS's permissions).
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _packaged_references(manifest: EvidenceManifest) -> tuple[ArtifactReference, ...]:
    """Every artifact reference in a manifest that a package should try to include."""

    candidates = (
        manifest.before_frame,
        manifest.trigger_frame,
        manifest.after_frame,
        manifest.clip,
        manifest.trajectory,
        manifest.plate_crop,
        *manifest.additional_artifacts,
    )
    seen: set[str] = set()
    packaged: list[ArtifactReference] = []
    for reference in candidates:
        if reference is None or reference.sha256 is None:
            # Without a hash the reference names a frame nobody rendered (the
            # pre-H14 shape); there are no bytes to package.
            continue
        if reference.locator in seen:
            continue
        seen.add(reference.locator)
        packaged.append(reference)
    return tuple(packaged)

"""Merging rendered artifacts into a manifest at read time (H14).

The join between the two halves of the evidence record: the **immutable** manifest
the reasoner's run persisted, and the **append-only** rendered-artifact journal the
rendering engine writes afterwards. Neither is ever rewritten; this module composes
them into the manifest a client is served.

Why the merge replaces the frame slots
--------------------------------------
A persisted manifest's ``before_frame`` / ``trigger_frame`` / ``after_frame``
locators name frames by ingestion identity -- a truthful record of *which* frames the
engine picked, but not something a client can fetch, and carrying no hash because
nothing had been rendered. Once a still for that slot exists, the served manifest
points at the artifact instead: a content-addressed locator that resolves through
the artifact endpoint and whose ``sha256`` can be verified against the bytes.

That is a substitution of a *reference*, never of a fact. The frame the reference
identifies is unchanged -- the rendered still was decoded from exactly the media time
the persisted manifest recorded -- and the original locator remains in the persisted
file, which is the authority for audit. What changes is only whether the reference
can be dereferenced.

An event with no rendered artifacts (every repository written before H14) merges to
a manifest that is byte-identical to the persisted one, which is what keeps those
repositories serving exactly as they did.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import EvidenceManifest
from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference

#: Manifest field name for each frame slot the merge can fill.
_FRAME_FIELDS: dict[ArtifactKind, str] = {
    ArtifactKind.BEFORE_FRAME: "before_frame",
    ArtifactKind.TRIGGER_FRAME: "trigger_frame",
    ArtifactKind.AFTER_FRAME: "after_frame",
}


def merge_rendered_artifacts(
    manifest: EvidenceManifest, rendered: Sequence[ArtifactReference]
) -> EvidenceManifest:
    """Return ``manifest`` with rendered artifacts merged in (a copy; input untouched).

    Rendered frames replace their typed slot; every rendered artifact also appears in
    ``additional_artifacts``, so a kind with no dedicated field (and any future output
    format) still reaches the client. With no rendered artifacts the manifest is
    returned unchanged.
    """

    if not rendered:
        return manifest

    updates: dict[str, object] = {}
    filled: set[ArtifactKind] = set()
    for reference in rendered:
        field = _FRAME_FIELDS.get(reference.kind)
        if field is not None and reference.kind not in filled:
            updates[field] = reference
            filled.add(reference.kind)

    known = {artifact.locator for artifact in manifest.additional_artifacts}
    extra = tuple(
        reference for reference in rendered if reference.locator not in known
    )
    if extra:
        updates["additional_artifacts"] = (*manifest.additional_artifacts, *extra)

    return manifest.model_copy(update=updates) if updates else manifest


def rendered_artifact_for(
    manifest: EvidenceManifest, kind: ArtifactKind
) -> ArtifactReference | None:
    """The fetchable artifact of ``kind`` on a served manifest, or ``None``.

    "Fetchable" means content-addressed: a reference without a ``sha256`` is a
    pre-render placeholder, and reporting it here would send a client after bytes
    that were never produced.
    """

    field = _FRAME_FIELDS.get(kind)
    if field is not None:
        slot = getattr(manifest, field, None)
        if isinstance(slot, ArtifactReference) and slot.sha256 is not None:
            return slot
    for reference in manifest.additional_artifacts:
        if reference.kind is kind and reference.sha256 is not None:
            return reference
    return None

"""Reading the evidence frames a manifest already declares (H14).

The **single** place in the codebase that knows how an ``EvidenceManifest``
records which source frames its evidence references point at. Everything that
needs to render, re-render, or serve an evidence frame reads it through here.

Why a reader at all (and why not a contract change)
---------------------------------------------------
``build_engine_manifest`` picks the before/trigger/after frames from the frames a
run actually processed and records their media times in a ``rule_trace`` step
labelled ``evidence-frames``. That step is already present in **every** manifest
this system has ever persisted, and its measurement names are pinned by
``tests/engine/test_engine_evidence.py``. So the information a renderer needs is
persisted and tested -- it is just written in a string-keyed side channel.

The alternative addressing route is the reference's own ``locator``, which embeds
``frame_id = sha256(source_id || frame_index)``. That route is rejected: ``source_id``
derives from the video's *resolved absolute path*, so moving the storage directory
(or launching the server from another working directory) silently changes every
frame id and permanently orphans already-persisted manifests. Media time is
independent of where the bytes live, which is why it is the addressing key.

Concentrating the convention here is what keeps it a convention with one reader
instead of a format parsed in three places: if the manifest shape ever changes,
exactly one function moves.
"""

from __future__ import annotations

from ..contracts import EvidenceManifest
from ..contracts.enums import ArtifactKind
from ..contracts.evidence import ArtifactReference

#: The ``rule_trace`` step label under which frame media times are recorded.
EVIDENCE_FRAMES_LABEL = "evidence-frames"

#: Suffix appended to an :class:`ArtifactKind` value to name its media-time measurement
#: (e.g. ``trigger_frame`` -> ``trigger_frame_media_time``).
MEDIA_TIME_SUFFIX = "_media_time"

#: The frame slots an evidence manifest can carry, in presentation order.
EVIDENCE_FRAME_KINDS: tuple[ArtifactKind, ...] = (
    ArtifactKind.BEFORE_FRAME,
    ArtifactKind.TRIGGER_FRAME,
    ArtifactKind.AFTER_FRAME,
)


def media_time_measurement_name(kind: ArtifactKind) -> str:
    """The measurement name carrying ``kind``'s media time in the rule trace."""

    return f"{kind.value}{MEDIA_TIME_SUFFIX}"


def evidence_frame_references(
    manifest: EvidenceManifest,
) -> dict[ArtifactKind, ArtifactReference]:
    """The frame references a manifest declares, keyed by kind (absent ones omitted).

    A manifest legitimately has no before/after frame when the stream started or
    ended inside the evidence margin, so a missing slot is an honest absence rather
    than an error.
    """

    slots = (
        (ArtifactKind.BEFORE_FRAME, manifest.before_frame),
        (ArtifactKind.TRIGGER_FRAME, manifest.trigger_frame),
        (ArtifactKind.AFTER_FRAME, manifest.after_frame),
    )
    return {kind: reference for kind, reference in slots if reference is not None}


def evidence_frame_times(manifest: EvidenceManifest) -> dict[ArtifactKind, float]:
    """The media time (seconds) of each evidence frame the manifest declares.

    Returns only the kinds for which the manifest carries **both** a reference and a
    recorded media time. A manifest that predates the engine's frame picking (the
    P1-U11 stub shape, whose trigger reference is a synthetic locator with no
    processed frame behind it) therefore yields an empty mapping, and the caller
    renders nothing rather than seeking to a fabricated position.
    """

    references = evidence_frame_references(manifest)
    if not references:
        return {}

    measured: dict[str, float] = {}
    for step in manifest.rule_trace:
        if step.label != EVIDENCE_FRAMES_LABEL:
            continue
        for measurement in step.measurements:
            measured[measurement.name] = measurement.value

    times: dict[ArtifactKind, float] = {}
    for kind in references:
        value = measured.get(media_time_measurement_name(kind))
        if value is not None and value >= 0.0:
            times[kind] = value
    return times

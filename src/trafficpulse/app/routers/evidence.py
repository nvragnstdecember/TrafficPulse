"""Evidence endpoints: the manifest, its rendered artifacts, and its package.

The manifest endpoint (H7A) now serves the **composed** record: the write-once
manifest the run persisted, with any rendered-artifact references merged in at read
time (H14). The persisted file is never modified, so a repository written before
H14 is served exactly as it always was.

Two endpoints join it. ``/artifacts/{kind}`` serves the bytes of one rendered
evidence frame -- the only way a client should obtain evidence pixels, because it is
the only source whose provenance the backend can state. ``/package`` serves the
deterministic ZIP an analyst downloads.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from ...contracts import EvidenceManifest
from ...contracts.enums import ArtifactKind
from ..dependencies import EvidenceServiceDep
from ..models import ErrorResponse

router = APIRouter(tags=["evidence"])


@router.get(
    "/api/evidence/{event_id}",
    response_model=EvidenceManifest,
    summary="Evidence manifest for an event",
    description="Return the evidence manifest for a confirmed event: the record the "
    "run persisted, with references to any rendered artifacts merged in. A frame "
    "reference carrying a `sha256` is fetchable through the artifacts endpoint; one "
    "without is a pre-render reference to a frame that was never materialised.",
    responses={404: {"model": ErrorResponse, "description": "Unknown event id"}},
)
def get_evidence(event_id: str, evidence: EvidenceServiceDep) -> EvidenceManifest:
    return evidence.get(event_id)


@router.get(
    "/api/evidence/{event_id}/artifacts/{kind}",
    summary="Rendered evidence artifact",
    description="Serve one rendered evidence frame (`before_frame`, `trigger_frame`, "
    "or `after_frame`) as an image. These are the frames the engine actually picked, "
    "drawn with the same overlay renderer as the annotated video, and verified "
    "against the SHA-256 the manifest records before they are served. 404 when "
    "nothing was rendered for that slot.",
    response_class=Response,
    responses={
        200: {"content": {"image/*": {}}, "description": "The rendered frame."},
        404: {
            "model": ErrorResponse,
            "description": "Unknown event, or no rendered artifact of that kind",
        },
    },
)
def get_evidence_artifact(
    event_id: str, kind: ArtifactKind, evidence: EvidenceServiceDep
) -> Response:
    data, media_type = evidence.artifact(event_id, kind)
    # Artifacts are content-addressed and write-once, so their bytes can never change
    # under a locator -- they are safe to cache immutably.
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get(
    "/api/evidence/{event_id}/package",
    summary="Evidence package (ZIP)",
    description="Download the complete evidence package for one event: the frozen "
    "confirmed event, the served manifest, and every rendered frame. Deterministic "
    "-- the same event and artifacts always produce byte-identical bytes -- so the "
    "archive's own hash is a property of the evidence rather than of the download.",
    response_class=Response,
    responses={
        200: {"content": {"application/zip": {}}, "description": "The evidence package."},
        404: {"model": ErrorResponse, "description": "Unknown event id"},
    },
)
def get_evidence_package(event_id: str, evidence: EvidenceServiceDep) -> Response:
    data, filename = evidence.package(event_id)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

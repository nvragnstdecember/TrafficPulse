"""Evidence endpoint: the manifest for a confirmed event (H7A).

Returns the frozen ``EvidenceManifest`` -- id linkage, provenance, and relative
frame/artifact **references** only. No media is rendered or served: the manifest
points at where artifacts live, exactly as H6/P1-U11 built it.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...contracts import EvidenceManifest
from ..dependencies import EvidenceServiceDep
from ..models import ErrorResponse

router = APIRouter(tags=["evidence"])


@router.get(
    "/api/evidence/{event_id}",
    response_model=EvidenceManifest,
    summary="Evidence manifest for an event",
    description="Return the evidence manifest (frame references + metadata, no "
    "rendered media) for a confirmed event.",
    responses={404: {"model": ErrorResponse, "description": "Unknown event id"}},
)
def get_evidence(event_id: str, evidence: EvidenceServiceDep) -> EvidenceManifest:
    return evidence.get(event_id)

"""Helmet-analysis and deployment-posture endpoints.

Two reads, both about the same thing from different ends: what a run actually saw, and
what this deployment is entitled to claim about it. They are kept out of the events
router deliberately -- an analysis produces no ``ConfirmedEvent``, and serving it from
the same resource as confirmed violations would be the first step towards a client
treating the two as interchangeable.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import posture as posture_module
from ..dependencies import ConfigDep, ProcessingServiceDep
from ..models import ErrorResponse, HelmetAnalysisResponse, SystemPostureResponse

router = APIRouter(tags=["analysis"])


@router.get(
    "/api/system/posture",
    response_model=SystemPostureResponse,
    summary="What this deployment can honestly claim",
    description="The capability status strip: detection, helmet classification, driver "
    "attribution, turban exemption, and helmet violation enforcement -- each with a "
    "state and a plain-language reason. Distinct from /api/health, which reports "
    "whether the service is working rather than what its configuration entitles anyone "
    "to say. Computed from configuration only: no model is loaded and no checkpoint "
    "is read.",
)
def get_posture(config: ConfigDep) -> SystemPostureResponse:
    return SystemPostureResponse.from_posture(posture_module.describe(config))


@router.get(
    "/api/process/{job_id}/helmet-analysis",
    response_model=HelmetAnalysisResponse,
    summary="Helmet analysis for a finished run",
    description="Per-rider helmet classification for a run that declared a helmet "
    "analysis: the stabilized label, its supporting agreement, the observed per-frame "
    "instability, and -- separately -- whether anything about that rider could be acted "
    "on. **Nothing here is a violation.** An analysis mints no event; /api/events "
    "remains the only source of confirmed violations. Available only for a run that "
    "declared an analysis and finished in this process: the fold is derived, not "
    "persisted, so a run recovered after a restart reports 404.",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Unknown job, or the job has no helmet analysis",
        }
    },
)
def get_helmet_analysis(
    job_id: str, processing: ProcessingServiceDep, config: ConfigDep
) -> HelmetAnalysisResponse:
    report = processing.helmet_analysis(job_id)
    return HelmetAnalysisResponse.from_report(
        job_id,
        report,
        # Read at request time rather than stamped at run time: the posture describes
        # the deployment, and a client must be told the posture it is reading *now*.
        enforcement=posture_module.describe(config).helmet_enforcement,
    )

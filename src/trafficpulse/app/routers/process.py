"""Processing-job endpoints: create a job and poll its status (H7A)."""

from __future__ import annotations

from fastapi import APIRouter, status

from ..dependencies import ProcessingServiceDep
from ..models import (
    ErrorResponse,
    JobStatusResponse,
    ProcessRequest,
    ProcessResponse,
)

router = APIRouter(tags=["processing"])


@router.post(
    "/api/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start processing a video",
    description="Create a processing job for a previously uploaded video and "
    "return its id. Inference runs asynchronously; poll GET /api/process/{job_id}.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid processing configuration"},
        404: {"model": ErrorResponse, "description": "Unknown video id"},
        503: {"model": ErrorResponse, "description": "Inference backend unavailable"},
    },
)
def create_job(request: ProcessRequest, processing: ProcessingServiceDep) -> ProcessResponse:
    job = processing.submit(video_id=request.video_id, rules=request.rules)
    return ProcessResponse(job_id=job.job_id, video_id=job.video_id, status=job.status)


@router.get(
    "/api/process/{job_id}",
    response_model=JobStatusResponse,
    summary="Processing-job status",
    description="Return a job's status, progress, frames processed, FPS, and "
    "estimated time remaining. Unknown values are null rather than fabricated.",
    responses={404: {"model": ErrorResponse, "description": "Unknown job id"}},
)
def get_job(job_id: str, processing: ProcessingServiceDep) -> JobStatusResponse:
    return processing.status(job_id)


@router.post(
    "/api/process/{job_id}/cancel",
    response_model=JobStatusResponse,
    summary="Cancel a processing job",
    description="Request cooperative cancellation of a job and return its current "
    "status. Cancellation is asynchronous: a running job stops at the next frame "
    "and transitions to 'cancelled'; poll GET /api/process/{job_id} until it does. "
    "Cancelling an already-finished job is a no-op that returns its existing status.",
    responses={404: {"model": ErrorResponse, "description": "Unknown job id"}},
)
def cancel_job(job_id: str, processing: ProcessingServiceDep) -> JobStatusResponse:
    return processing.cancel(job_id)

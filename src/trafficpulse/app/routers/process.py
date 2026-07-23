"""Processing-job endpoints: create a job and poll its status (H7A)."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import FileResponse

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


@router.get(
    "/api/process/{job_id}/overlay",
    summary="Annotated (overlay) video for a job",
    description="Stream the rendered overlay video -- the source clip with detection "
    "boxes, association lines, observation state, and confirmed-violation banners "
    "drawn on every frame. The original upload is served separately and never "
    "modified. Available only after a successful run that produced overlay metadata "
    "(poll GET /api/process/{job_id} for `overlay_available`).",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "H.264/MP4 overlay video."},
        404: {"model": ErrorResponse, "description": "No overlay video for this job"},
    },
)
def get_overlay(job_id: str, processing: ProcessingServiceDep) -> FileResponse:
    # Range requests (206) are handled by Starlette's FileResponse, so the <video>
    # element can seek. Served inline (no download disposition).
    return FileResponse(processing.overlay_video_path(job_id), media_type="video/mp4")

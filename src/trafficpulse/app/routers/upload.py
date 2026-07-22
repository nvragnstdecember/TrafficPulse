"""Video upload endpoint (H7A).

Reads the multipart body under a hard size cap (streaming, so an over-limit
upload is rejected without buffering the whole payload) and delegates storage +
validation to :class:`~trafficpulse.app.services.VideoService`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from ..dependencies import ConfigDep, VideoServiceDep
from ..errors import PayloadTooLargeError
from ..models import ErrorResponse, VideoUploadResponse

router = APIRouter(tags=["video"])

_READ_CHUNK = 1024 * 1024  # 1 MiB


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload, aborting as soon as it exceeds ``max_bytes``."""

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(
                f"upload exceeds the {max_bytes}-byte limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/api/video/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a source video",
    description="Accept a multipart video upload, validate its extension, size, "
    "and readability, store it, and return a content-addressed video id.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported or unreadable media"},
        409: {"model": ErrorResponse, "description": "Identical video already uploaded"},
        413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
    },
)
async def upload_video(
    videos: VideoServiceDep,
    config: ConfigDep,
    file: Annotated[UploadFile, File(description="The video file to upload.")],
) -> VideoUploadResponse:
    filename = file.filename or "upload"
    videos.assert_supported_extension(filename)  # fast-fail before reading the body
    data = await _read_capped(file, config.max_upload_bytes)
    record = videos.store_upload(filename, data)
    return VideoUploadResponse.from_record(record)

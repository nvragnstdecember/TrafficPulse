"""Historical video library endpoints: browse, describe, and play back (H11).

The read side of the video resource. Upload keeps its established path
(``POST /api/video/upload``, unchanged so no client breaks); everything that
addresses the *collection* lives under ``/api/videos``, which is the only shape a
list can take.

These three endpoints are what make persisted work discoverable. Before them a
video could only be reached by an id the client happened to be holding, which in
practice meant the one upload the browser had in local storage -- so a restarted
backend served events nobody could navigate to. The list answers "what is in this
repository", the detail answers "what is the state of this one", and the media
endpoint answers "let me see it", which no endpoint could answer for a video the
current browser session did not upload.

Nothing here loads a video's analysis. Events, evidence, review, and the annotated
overlay keep their own endpoints and are fetched after a selection.
"""

from __future__ import annotations

import mimetypes
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ..dependencies import VideoLibraryServiceDep, VideoServiceDep
from ..models import ErrorResponse, VideoListResponse, VideoSort, VideoSummary

router = APIRouter(tags=["video"])


@router.get(
    "/api/videos",
    response_model=VideoListResponse,
    summary="List stored videos",
    description="Browse every video in the repository -- including those recovered "
    "from disk after a restart, which are indistinguishable from ones uploaded in "
    "this session. Returns browsing metadata only (identity, processing state, "
    "event and review counts); no event, evidence, or overlay payload is loaded, so "
    "a listing costs no deserialisation regardless of repository size. Fetch a "
    "video's analysis through the existing event/evidence endpoints once selected.",
)
def list_videos(
    library: VideoLibraryServiceDep,
    limit: Annotated[int, Query(ge=1, le=200, description="Page size.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    sort: Annotated[
        VideoSort, Query(description="Ordering; '-' prefixes descending.")
    ] = VideoSort.UPLOADED_AT_DESC,
) -> VideoListResponse:
    return library.list(limit=limit, offset=offset, sort=sort)


@router.get(
    "/api/videos/{video_id}",
    response_model=VideoSummary,
    summary="Stored-video summary",
    description="Return one video's browsing metadata -- the same row the list "
    "returns, for a client that holds an id (e.g. restoring a remembered selection) "
    "and must not fetch the whole library to resolve it.",
    responses={404: {"model": ErrorResponse, "description": "Unknown video id"}},
)
def get_video(video_id: str, library: VideoLibraryServiceDep) -> VideoSummary:
    return library.get(video_id)


@router.get(
    "/api/videos/{video_id}/media",
    summary="Stored source video",
    description="Stream the original uploaded video, exactly as stored and never "
    "modified. This is what makes a previously-processed video playable again: the "
    "workspace otherwise plays a browser object URL for the file the analyst picked, "
    "which does not survive the session. Prefer the annotated overlay at "
    "GET /api/process/{job_id}/overlay when one is available; this is the source it "
    "was rendered from, and the fallback when a run produced none.",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/*": {}}, "description": "The stored source video."},
        404: {
            "model": ErrorResponse,
            "description": "Unknown video id, or its stored file is gone",
        },
    },
)
def get_video_media(video_id: str, videos: VideoServiceDep) -> FileResponse:
    # Range requests (206) are handled by Starlette's FileResponse, so the <video>
    # element can seek. The container is whatever was uploaded, so the media type is
    # read from the extension rather than assumed to be MP4.
    path = videos.media_path(video_id)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)

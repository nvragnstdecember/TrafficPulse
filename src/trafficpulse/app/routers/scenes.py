"""Scene resources: calibrate a video, read the scene it was reasoned under (H12).

Scenes become addressable here. Before this, the governing ``SceneConfig`` was a
file path in the server's configuration -- not a resource, not fetchable, and the
same one for every upload. These endpoints make a scene something a client can
create, inspect, and bind, per video.

Two path shapes, one resource
------------------------------
``/api/scenes/{scene_hash}`` addresses a stored scene **revision** by its content
hash -- which is exactly the value every ``ConfirmedEvent`` carries in
``scene_config_hash``. So an analyst looking at a two-month-old violation can
fetch the precise geometry and thresholds it was confirmed under, rather than
whatever the scene has since been edited into.

``/api/videos/{video_id}/scene`` addresses the **binding**: which revision this
video is calibrated against. Calibrating is a ``PUT`` because it is idempotent
replacement -- re-sending the same drawing is a no-op, and a changed drawing
supersedes the previous binding without destroying the revision it pointed at.

Both live in one router because they are one resource seen from two directions;
splitting them would put the scene's read path and its write path in different
modules for no gain.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from ...contracts import SceneConfig
from ...scenes import SceneDraft
from ..dependencies import SceneServiceDep
from ..models import ErrorResponse, SceneSummary, SceneValidationResponse

router = APIRouter(tags=["scenes"])


@router.get(
    "/api/scenes/{scene_hash}",
    response_model=SceneConfig,
    summary="Stored scene revision",
    description="Return one stored scene, verbatim as the frozen SceneConfig "
    "contract -- the complete geometry, thresholds, provenance, and calibration "
    "state. Addressed by the scene's own content hash, so any event's "
    "`scene_config_hash` resolves here to the exact revision that produced it. "
    "Revisions are immutable: editing a video's calibration stores a new one and "
    "leaves this untouched.",
    responses={404: {"model": ErrorResponse, "description": "No such stored scene"}},
)
def get_scene(scene_hash: str, scenes: SceneServiceDep) -> SceneConfig:
    return scenes.get(scene_hash)


@router.get(
    "/api/videos/{video_id}/scene",
    response_model=SceneSummary,
    summary="A video's calibrated scene",
    description="Return the summary of the scene this video is calibrated against, "
    "including which violations that scene can actually reason about. A video that "
    "has not been calibrated returns 404 -- that is a state to render (offer "
    "calibration), not an error to report.",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Unknown video, or the video has no calibrated scene",
        }
    },
)
def get_video_scene(video_id: str, scenes: SceneServiceDep) -> SceneSummary:
    return scenes.summary_for_video(video_id)


@router.put(
    "/api/videos/{video_id}/scene",
    response_model=SceneSummary,
    status_code=status.HTTP_200_OK,
    summary="Calibrate a video",
    description="Author this video's scene from a drawn draft, store it, and bind "
    "it to the video. Idempotent: an unchanged drawing rebuilds identical content, "
    "hashes to the same address, and changes nothing. The draft's frame size must "
    "match the video's decoded dimensions -- geometry drawn against a different "
    "frame would land in the wrong place. Processing this video afterwards resolves "
    "this scene, which is what enables the geometry-dependent rules (wrong way, "
    "illegal stopping) without any configuration file.",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The drawing does not match the video's frame",
        },
        404: {"model": ErrorResponse, "description": "Unknown video id"},
        422: {"model": ErrorResponse, "description": "The draft is not a valid scene"},
    },
)
def put_video_scene(
    video_id: str, draft: SceneDraft, scenes: SceneServiceDep
) -> SceneSummary:
    return scenes.calibrate(video_id, draft)


@router.post(
    "/api/videos/{video_id}/scene/validate",
    response_model=SceneValidationResponse,
    summary="Check a draft without saving it",
    description="Validate a drawn draft against the frozen scene contract and "
    "report which violations it would unlock, without storing anything. Returns 200 "
    "with `valid: false` and the contract's own messages for an unusable drawing "
    "rather than a request error, so a half-finished calibration is a state the UI "
    "can render while the analyst is still drawing.",
    responses={404: {"model": ErrorResponse, "description": "Unknown video id"}},
)
def validate_video_scene(
    video_id: str, draft: SceneDraft, scenes: SceneServiceDep
) -> SceneValidationResponse:
    return scenes.validate(video_id, draft)

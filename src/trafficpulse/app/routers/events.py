"""Confirmed-event endpoints: list, detail, and analyst review (H7A; review H9).

The list returns compact summaries; the detail returns the frozen
``ConfirmedEvent`` contract verbatim (full fidelity, no re-modelling).

The review endpoints live here rather than in a router of their own because a
review case has no identity apart from its event -- it is addressed by
``event_id``, it 404s when the event is unknown, and giving it a separate resource
tree would imply a lifecycle it does not have.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ...contracts import ConfirmedEvent
from ..dependencies import EventServiceDep, ReviewServiceDep
from ..models import (
    ErrorResponse,
    EventListResponse,
    EventSort,
    ReviewDecisionRequest,
    ReviewResponse,
)

router = APIRouter(tags=["events"])


@router.get(
    "/api/events",
    response_model=EventListResponse,
    summary="List confirmed events",
    description="List event summaries, optionally filtered by video and by "
    "processing run, with deterministic sorting and pagination.\n\n"
    "Without `job_id` the response covers **every succeeded run** of the matching "
    "videos, deduplicated by event id -- reprocessing re-confirms byte-identical "
    "events, so this is the repository view: what these videos have ever been found "
    "to contain. With `job_id` it covers exactly that one run, which is what a "
    "review surface wants once a video has been processed more than once.",
)
def list_events(
    events: EventServiceDep,
    video_id: Annotated[
        str | None, Query(description="Restrict to one video's events.")
    ] = None,
    job_id: Annotated[
        str | None,
        Query(
            description="Restrict to one processing run's events. Omit for the "
            "video's whole history. A job that does not exist, has not succeeded, "
            "or belongs to a different video selects no run and returns an empty "
            "page."
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Page size.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    sort: Annotated[
        EventSort, Query(description="Ordering; '-' prefixes descending.")
    ] = EventSort.TRIGGER_AT_ASC,
) -> EventListResponse:
    return events.list(
        video_id=video_id, job_id=job_id, limit=limit, offset=offset, sort=sort
    )


@router.get(
    "/api/events/{event_id}",
    response_model=ConfirmedEvent,
    summary="Confirmed-event detail",
    description="Return the complete confirmed-event record.",
    responses={404: {"model": ErrorResponse, "description": "Unknown event id"}},
)
def get_event(event_id: str, events: EventServiceDep) -> ConfirmedEvent:
    return events.get(event_id)


@router.get(
    "/api/events/{event_id}/review",
    response_model=ReviewResponse,
    summary="Analyst review case + audit history",
    description="Return the event's current review case and every recorded analyst "
    "action, oldest first. The case is derived from the history, so the two can "
    "never disagree. An event nobody has reviewed returns status 'pending' with an "
    "empty history rather than a 404.",
    responses={404: {"model": ErrorResponse, "description": "Unknown event id"}},
)
def get_review(event_id: str, reviews: ReviewServiceDep) -> ReviewResponse:
    return reviews.get(event_id)


@router.post(
    "/api/events/{event_id}/review",
    response_model=ReviewResponse,
    summary="Record an analyst decision",
    description="Append one analyst action (open / note / approve / reject / "
    "false_positive / needs_more_evidence / reopen / export) to the event's "
    "append-only review journal, and return the resulting case + history. Actions "
    "are never overwritten: correcting a decision means reopening and deciding "
    "again, and both remain in the history.",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown event id"},
        409: {
            "model": ErrorResponse,
            "description": "The action is not legal from the event's current status",
        },
    },
)
def post_review(
    event_id: str, request: ReviewDecisionRequest, reviews: ReviewServiceDep
) -> ReviewResponse:
    return reviews.decide(event_id, request)

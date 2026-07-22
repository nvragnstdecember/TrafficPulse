"""Confirmed-event endpoints: paginated list + full detail (H7A).

The list returns compact summaries; the detail returns the frozen
``ConfirmedEvent`` contract verbatim (full fidelity, no re-modelling).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ...contracts import ConfirmedEvent
from ..dependencies import EventServiceDep
from ..models import ErrorResponse, EventListResponse, EventSort

router = APIRouter(tags=["events"])


@router.get(
    "/api/events",
    response_model=EventListResponse,
    summary="List confirmed events",
    description="List event summaries, optionally filtered by video, with "
    "deterministic sorting and pagination.",
)
def list_events(
    events: EventServiceDep,
    video_id: Annotated[
        str | None, Query(description="Restrict to one video's events.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Page size.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    sort: Annotated[
        EventSort, Query(description="Ordering; '-' prefixes descending.")
    ] = EventSort.TRIGGER_AT_ASC,
) -> EventListResponse:
    return events.list(video_id=video_id, limit=limit, offset=offset, sort=sort)


@router.get(
    "/api/events/{event_id}",
    response_model=ConfirmedEvent,
    summary="Confirmed-event detail",
    description="Return the complete confirmed-event record.",
    responses={404: {"model": ErrorResponse, "description": "Unknown event id"}},
)
def get_event(event_id: str, events: EventServiceDep) -> ConfirmedEvent:
    return events.get(event_id)

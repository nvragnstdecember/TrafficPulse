"""Controlled demonstration: declare what a clip contains, compare with what ran.

The **expectation** resource. A controlled demonstration has two halves that must
never be confused -- what a clip was built to contain, and what the reasoners
independently confirmed -- and these endpoints own the first half plus the one
place the two are put side by side.

Why the declaration is a resource and not a request field
----------------------------------------------------------
It is deliberately *not* part of ``ProcessRequest``. A processing request is the
input to the engine, and anything on it is something the engine could in principle
read; putting expectations there would make "the demo told the system the answer"
a question a reviewer has to take on trust rather than one the type system settles.
As a separate resource, written to a separate store that no rule, reasoner or
engine ever opens, the separation is structural.

Why the comparison never 404s on a missing declaration
--------------------------------------------------------
``GET .../expectation`` 404s when nothing was declared -- the client asked for the
declaration itself. ``GET .../expectation/comparison`` does not: "nothing was
claimed about this clip, and here is what was nonetheless confirmed" is a real
answer with a real table, and it is what an *uncalibrated* or ordinary video
legitimately returns.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Response, status

from ..dependencies import ExpectationServiceDep
from ..models import (
    ErrorResponse,
    ExpectationComparison,
    ExpectationDeclaration,
    ExpectationRecord,
)

router = APIRouter(tags=["controlled demo"])


@router.get(
    "/api/videos/{video_id}/expectation",
    response_model=ExpectationRecord,
    summary="A video's declared expectation",
    description="Return what this clip was declared to have been built to contain, "
    "and who declared it when. An expectation is ground truth for a **controlled "
    "demonstration** only -- it is never a detection, never reaches the engine, and "
    "never appears in any event listing. A video with no declaration returns 404: "
    "that is a state to render (offer to declare one), not an error.",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Unknown video, or the video has no declared expectation",
        }
    },
)
def get_expectation(video_id: str, expectations: ExpectationServiceDep) -> ExpectationRecord:
    return expectations.get(video_id)


@router.put(
    "/api/videos/{video_id}/expectation",
    response_model=ExpectationRecord,
    status_code=status.HTTP_200_OK,
    summary="Declare what a controlled clip contains",
    description="Record the violation families this clip was constructed to contain, "
    "plus the operator's written declaration of what the controlled scenario is. "
    "Replacement, not merge: the stored declaration becomes exactly what was sent, "
    "with a fresh `declared_at`, because when somebody last asserted this is part of "
    "what makes it auditable. Declaring changes nothing about how the video is "
    "processed -- the same rules run over the same scene either way.",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown video id"},
        422: {"model": ErrorResponse, "description": "The declaration is malformed"},
    },
)
def put_expectation(
    video_id: str,
    declaration: ExpectationDeclaration,
    expectations: ExpectationServiceDep,
) -> ExpectationRecord:
    return expectations.declare(video_id, declaration)


@router.delete(
    "/api/videos/{video_id}/expectation",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Withdraw a declaration",
    description="Remove this video's declared expectation. Idempotent: withdrawing "
    "one that does not exist succeeds, because the caller's intent -- that this video "
    "carries no claim -- is satisfied either way. Confirmed events are untouched.",
    responses={404: {"model": ErrorResponse, "description": "Unknown video id"}},
)
def delete_expectation(video_id: str, expectations: ExpectationServiceDep) -> Response:
    expectations.clear(video_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/api/videos/{video_id}/expectation/comparison",
    response_model=ExpectationComparison,
    summary="Expected beside detected",
    description="Put the declared expectation next to the events the reasoners "
    "actually confirmed, as counts of real event ids that can each be opened and "
    "inspected. Reports matched / missing / unexpected and **no accuracy metric**: "
    "precision or recall over one hand-authored clip would be arithmetic against "
    "ground truth the same person wrote. Pass `job_id` to scope the detected side to "
    "one run, which is what a demonstration wants; omit it to compare against every "
    "succeeded run of the video. A video with no declaration is not an error -- every "
    "detected family is reported as unexpected.",
    responses={404: {"model": ErrorResponse, "description": "Unknown video id"}},
)
def compare_expectation(
    video_id: str,
    expectations: ExpectationServiceDep,
    job_id: str | None = Query(
        default=None,
        description="Scope the detected side to one run. Omit for every succeeded run.",
    ),
) -> ExpectationComparison:
    return expectations.compare(video_id, job_id=job_id)

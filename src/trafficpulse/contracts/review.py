"""Review-case contract (human-review state around an evidence package).

Data only: represents the review decision state and the legal transitions between
its states. It implements no authentication, authorization, UI, or audit-log
persistence (architecture-review §21) -- the journal itself lives in
:mod:`trafficpulse.persistence.review_store`, which is what ``audit_ref`` points at.

Two records, one source of truth (H9)
--------------------------------------
:class:`ReviewEntry` is one immutable thing an analyst did. :class:`ReviewCase` is
the current state, and it is a **fold over the entries** -- never stored
independently. That is deliberate: a status field written alongside a history log
is two representations of the same fact, and they drift the moment a write is lost
or replayed. Here the history *is* the record and the status is derived from it, so
"the case says approved but the log has no approval" is unrepresentable.

The transition table is part of this contract because which transitions are legal
is part of what the states *mean*. It is pure data plus pure functions: no I/O, no
policy about who may act, no wall-clock.
"""

from pydantic import AwareDatetime

from .enums import ReviewAction, ReviewStatus
from .primitives import ContractModel, NonEmptyStr

#: Which action may be taken from which status.
#:
#: A decided case can be reopened or re-decided -- an analyst may correct a
#: mistake, and the journal preserves that they did. Notes and exports are legal
#: from every state: refusing to let a reviewer annotate a case would be a workflow
#: defect, not a safeguard.
REVIEW_TRANSITIONS: dict[ReviewStatus, frozenset[ReviewAction]] = {
    ReviewStatus.PENDING: frozenset(
        {ReviewAction.OPEN, ReviewAction.NOTE, ReviewAction.EXPORT}
    ),
    ReviewStatus.IN_REVIEW: frozenset(
        {
            ReviewAction.NOTE,
            ReviewAction.APPROVE,
            ReviewAction.REJECT,
            ReviewAction.FALSE_POSITIVE,
            ReviewAction.NEEDS_MORE_EVIDENCE,
            ReviewAction.EXPORT,
        }
    ),
    ReviewStatus.NEEDS_MORE_EVIDENCE: frozenset(
        {
            ReviewAction.NOTE,
            ReviewAction.REOPEN,
            ReviewAction.APPROVE,
            ReviewAction.REJECT,
            ReviewAction.FALSE_POSITIVE,
            ReviewAction.EXPORT,
        }
    ),
    ReviewStatus.APPROVED: frozenset(
        {ReviewAction.NOTE, ReviewAction.REOPEN, ReviewAction.EXPORT}
    ),
    ReviewStatus.REJECTED: frozenset(
        {ReviewAction.NOTE, ReviewAction.REOPEN, ReviewAction.EXPORT}
    ),
    ReviewStatus.FALSE_POSITIVE: frozenset(
        {ReviewAction.NOTE, ReviewAction.REOPEN, ReviewAction.EXPORT}
    ),
}

#: The status an action produces. Actions absent from this map (``NOTE``,
#: ``EXPORT``) are *activity*, not decisions: they are journalled but leave the
#: status exactly where it was.
_ACTION_RESULT: dict[ReviewAction, ReviewStatus] = {
    ReviewAction.OPEN: ReviewStatus.IN_REVIEW,
    ReviewAction.REOPEN: ReviewStatus.IN_REVIEW,
    ReviewAction.APPROVE: ReviewStatus.APPROVED,
    ReviewAction.REJECT: ReviewStatus.REJECTED,
    ReviewAction.FALSE_POSITIVE: ReviewStatus.FALSE_POSITIVE,
    ReviewAction.NEEDS_MORE_EVIDENCE: ReviewStatus.NEEDS_MORE_EVIDENCE,
}


def can_transition(status: ReviewStatus, action: ReviewAction) -> bool:
    """Whether ``action`` is legal from ``status``."""

    return action in REVIEW_TRANSITIONS.get(status, frozenset())


def next_status(status: ReviewStatus, action: ReviewAction) -> ReviewStatus:
    """The status ``action`` produces from ``status``.

    Pure; the caller is responsible for having checked :func:`can_transition`.
    An action that records activity rather than a decision returns ``status``
    unchanged, which is what keeps a note from silently deciding a case.
    """

    return _ACTION_RESULT.get(action, status)


class ReviewEntry(ContractModel):
    """One immutable analyst action against a review case (H9).

    Append-only by construction: entries are never updated or deleted, so the
    journal is the audit trail rather than a summary of one. ``status_before`` and
    ``status_after`` are recorded on the entry so a reader can verify the fold
    without re-deriving it, and so a past transition stays legible even if the
    transition table later changes.

    ``at`` is the wall-clock instant of the action. This is one of the few places
    in TrafficPulse where wall-clock is correct: it timestamps a *human* act, not a
    media observation, and must not be confused with the media-time timestamps
    events carry.
    """

    entry_id: NonEmptyStr
    event_id: NonEmptyStr
    action: ReviewAction
    status_before: ReviewStatus
    status_after: ReviewStatus
    reviewer: NonEmptyStr
    at: AwareDatetime
    note: str | None = None
    reason: str | None = None


class ReviewCase(ContractModel):
    """The human-review state attached to one evidence package.

    ``reviewer_id`` is an opaque identifier only. ``audit_ref`` is a pointer to
    an append-only audit record maintained elsewhere; the log itself is not
    part of this contract.

    Fields added in H9 (``event_id``, ``reason``, ``updated_at``) are optional so
    records written before the analyst workflow existed still validate.
    ``event_id`` addresses the case by the id every other layer already uses.
    """

    review_case_id: NonEmptyStr
    evidence_package_id: NonEmptyStr
    event_id: NonEmptyStr | None = None
    status: ReviewStatus
    reviewer_id: NonEmptyStr | None = None
    decided_at: AwareDatetime | None = None
    note: str | None = None
    reason: str | None = None
    updated_at: AwareDatetime | None = None
    audit_ref: NonEmptyStr | None = None
    created_at: AwareDatetime

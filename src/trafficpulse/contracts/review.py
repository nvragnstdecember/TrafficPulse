"""Review-case contract (human-review state around an evidence package).

Data only: represents the review decision state. It implements no
authentication, authorization, UI, or audit-log persistence
(architecture-review §21).
"""

from pydantic import AwareDatetime

from .enums import ReviewStatus
from .primitives import ContractModel, NonEmptyStr


class ReviewCase(ContractModel):
    """The human-review state attached to one evidence package.

    ``reviewer_id`` is an opaque identifier only. ``audit_ref`` is a pointer to
    an append-only audit record maintained elsewhere; the log itself is not
    part of this contract.
    """

    review_case_id: NonEmptyStr
    evidence_package_id: NonEmptyStr
    status: ReviewStatus
    reviewer_id: NonEmptyStr | None = None
    decided_at: AwareDatetime | None = None
    note: str | None = None
    audit_ref: NonEmptyStr | None = None
    created_at: AwareDatetime

"""The review state machine: which analyst actions are legal, and what they produce."""

from __future__ import annotations

import pytest

from trafficpulse.contracts import REVIEW_TRANSITIONS, can_transition, next_status
from trafficpulse.contracts.enums import ReviewAction, ReviewStatus


def test_the_happy_path_runs_pending_to_decided() -> None:
    assert can_transition(ReviewStatus.PENDING, ReviewAction.OPEN)
    opened = next_status(ReviewStatus.PENDING, ReviewAction.OPEN)
    assert opened is ReviewStatus.IN_REVIEW

    for action, expected in (
        (ReviewAction.APPROVE, ReviewStatus.APPROVED),
        (ReviewAction.REJECT, ReviewStatus.REJECTED),
        (ReviewAction.FALSE_POSITIVE, ReviewStatus.FALSE_POSITIVE),
    ):
        assert can_transition(opened, action)
        assert next_status(opened, action) is expected


@pytest.mark.parametrize(
    "action",
    [ReviewAction.APPROVE, ReviewAction.REJECT, ReviewAction.FALSE_POSITIVE],
)
def test_a_pending_case_cannot_be_decided_without_being_opened(action: ReviewAction) -> None:
    # The lifecycle exists to make "somebody looked at this" auditable. Deciding
    # straight from pending would leave no record that a review happened at all.
    assert not can_transition(ReviewStatus.PENDING, action)


def test_a_decided_case_cannot_be_decided_again_without_reopening() -> None:
    assert not can_transition(ReviewStatus.APPROVED, ReviewAction.REJECT)
    assert not can_transition(ReviewStatus.REJECTED, ReviewAction.APPROVE)
    assert not can_transition(ReviewStatus.FALSE_POSITIVE, ReviewAction.APPROVE)


def test_a_decision_can_be_corrected_by_reopening() -> None:
    # An analyst must be able to fix a mistake; the journal preserves that they did.
    assert can_transition(ReviewStatus.APPROVED, ReviewAction.REOPEN)
    reopened = next_status(ReviewStatus.APPROVED, ReviewAction.REOPEN)
    assert reopened is ReviewStatus.IN_REVIEW
    assert can_transition(reopened, ReviewAction.REJECT)


def test_needs_more_evidence_parks_a_case_without_deciding_it() -> None:
    assert can_transition(ReviewStatus.IN_REVIEW, ReviewAction.NEEDS_MORE_EVIDENCE)
    parked = next_status(ReviewStatus.IN_REVIEW, ReviewAction.NEEDS_MORE_EVIDENCE)
    assert parked is ReviewStatus.NEEDS_MORE_EVIDENCE
    assert not parked.is_decided
    # ...and can be decided directly once the evidence is in hand.
    assert can_transition(parked, ReviewAction.APPROVE)


@pytest.mark.parametrize("status", list(ReviewStatus))
def test_notes_and_exports_are_legal_from_every_state(status: ReviewStatus) -> None:
    # Refusing to let a reviewer annotate or export a case would be a workflow
    # defect, not a safeguard.
    assert can_transition(status, ReviewAction.NOTE)
    assert can_transition(status, ReviewAction.EXPORT)


@pytest.mark.parametrize("status", list(ReviewStatus))
def test_activity_never_silently_decides_a_case(status: ReviewStatus) -> None:
    assert next_status(status, ReviewAction.NOTE) is status
    assert next_status(status, ReviewAction.EXPORT) is status


def test_every_status_has_a_transition_entry() -> None:
    # A status missing from the table would silently reject every action.
    assert set(REVIEW_TRANSITIONS) == set(ReviewStatus)


def test_only_decisions_report_as_decided() -> None:
    decided = {status for status in ReviewStatus if status.is_decided}
    assert decided == {
        ReviewStatus.APPROVED,
        ReviewStatus.REJECTED,
        ReviewStatus.FALSE_POSITIVE,
    }


def test_false_positive_is_distinct_from_rejected() -> None:
    # One says the offence is not worth pursuing; the other says the system was
    # wrong. Collapsing them would destroy the only detector-quality signal the
    # review workflow produces.
    assert ReviewStatus.FALSE_POSITIVE is not ReviewStatus.REJECTED
    assert next_status(ReviewStatus.IN_REVIEW, ReviewAction.FALSE_POSITIVE) is (
        ReviewStatus.FALSE_POSITIVE
    )

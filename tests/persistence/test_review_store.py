"""The append-only analyst-review journal and the case folded from it (H9)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trafficpulse.contracts import ReviewEntry
from trafficpulse.contracts.enums import ReviewAction, ReviewStatus
from trafficpulse.persistence import ReviewStore
from trafficpulse.persistence.errors import CorruptRecordError

AT = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _entry(
    *,
    event_id: str = "evt-1",
    action: ReviewAction = ReviewAction.OPEN,
    before: ReviewStatus = ReviewStatus.PENDING,
    after: ReviewStatus = ReviewStatus.IN_REVIEW,
    reviewer: str = "analyst-a",
    seconds: int = 0,
    note: str | None = None,
    reason: str | None = None,
) -> ReviewEntry:
    return ReviewEntry(
        entry_id=f"rev-{event_id}-{seconds}",
        event_id=event_id,
        action=action,
        status_before=before,
        status_after=after,
        reviewer=reviewer,
        at=AT + timedelta(seconds=seconds),
        note=note,
        reason=reason,
    )


def test_an_unreviewed_event_is_pending_with_no_journal(tmp_path: Path) -> None:
    # The absence of a decision *is* the pending state, so a finished run writes
    # nothing and an untouched event still answers correctly.
    store = ReviewStore(tmp_path)
    assert store.history("evt-1") == ()
    assert store.status("evt-1") is ReviewStatus.PENDING
    case = store.case("evt-1", evidence_package_id="pkg-1")
    assert case.status is ReviewStatus.PENDING
    assert case.event_id == "evt-1"
    assert case.decided_at is None
    assert not store.journal_path("evt-1").exists()


def test_appending_is_additive_and_ordered(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    store.append(_entry(seconds=0))
    store.append(
        _entry(
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            seconds=10,
        )
    )

    history = store.history("evt-1")
    assert [entry.action for entry in history] == [ReviewAction.OPEN, ReviewAction.APPROVE]
    assert store.status("evt-1") is ReviewStatus.APPROVED


def test_the_journal_is_never_rewritten(tmp_path: Path) -> None:
    # The audit guarantee: a later decision appends, it does not replace. Reopening
    # and re-deciding must leave the original decision legible forever.
    store = ReviewStore(tmp_path)
    store.append(_entry(seconds=0))
    store.append(
        _entry(
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            seconds=1,
        )
    )
    first_pass = store.journal_path("evt-1").read_text(encoding="utf-8")

    store.append(
        _entry(
            action=ReviewAction.REOPEN,
            before=ReviewStatus.APPROVED,
            after=ReviewStatus.IN_REVIEW,
            seconds=2,
        )
    )
    store.append(
        _entry(
            action=ReviewAction.REJECT,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.REJECTED,
            seconds=3,
        )
    )

    after = store.journal_path("evt-1").read_text(encoding="utf-8")
    assert after.startswith(first_pass)  # every earlier line survives verbatim
    assert len(store.history("evt-1")) == 4
    assert store.status("evt-1") is ReviewStatus.REJECTED


def test_the_case_is_a_fold_over_the_journal(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    store.append(_entry(seconds=0))
    store.append(
        _entry(
            action=ReviewAction.NOTE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.IN_REVIEW,
            note="Rider clearly bare-headed",
            seconds=5,
        )
    )
    store.append(
        _entry(
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            reviewer="analyst-b",
            reason="Clear violation",
            seconds=9,
        )
    )

    case = store.case("evt-1", evidence_package_id="pkg-1")
    assert case.status is ReviewStatus.APPROVED
    assert case.reviewer_id == "analyst-b"  # whoever acted last
    assert case.created_at == AT  # the first action opened the case
    assert case.updated_at == AT + timedelta(seconds=9)
    assert case.decided_at == AT + timedelta(seconds=9)
    assert case.reason == "Clear violation"
    assert case.audit_ref is not None


def test_a_decision_without_a_note_does_not_erase_the_note(tmp_path: Path) -> None:
    # Approving without retyping the note must not silently drop the analyst's
    # only written record of why.
    store = ReviewStore(tmp_path)
    store.append(_entry(seconds=0))
    store.append(
        _entry(
            action=ReviewAction.NOTE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.IN_REVIEW,
            note="Plate legible in trigger frame",
            seconds=1,
        )
    )
    store.append(
        _entry(
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            seconds=2,
        )
    )

    assert store.case("evt-1", evidence_package_id="pkg-1").note == (
        "Plate legible in trigger frame"
    )


def test_decided_at_tracks_the_decision_not_a_later_note(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    store.append(_entry(seconds=0))
    store.append(
        _entry(
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            seconds=4,
        )
    )
    store.append(
        _entry(
            action=ReviewAction.NOTE,
            before=ReviewStatus.APPROVED,
            after=ReviewStatus.APPROVED,
            note="Filed",
            seconds=99,
        )
    )

    case = store.case("evt-1", evidence_package_id="pkg-1")
    assert case.decided_at == AT + timedelta(seconds=4)
    assert case.updated_at == AT + timedelta(seconds=99)


def test_events_have_independent_journals(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    store.append(_entry(event_id="evt-1", seconds=0))
    store.append(
        _entry(
            event_id="evt-2",
            action=ReviewAction.APPROVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.APPROVED,
            seconds=0,
        )
    )

    assert store.statuses(["evt-1", "evt-2", "evt-3"]) == {
        "evt-1": ReviewStatus.IN_REVIEW,
        "evt-2": ReviewStatus.APPROVED,
        "evt-3": ReviewStatus.PENDING,
    }


def test_decisions_survive_a_new_store_instance(tmp_path: Path) -> None:
    # The persistence guarantee: nothing about a decision lives in process memory.
    ReviewStore(tmp_path).append(
        _entry(
            action=ReviewAction.FALSE_POSITIVE,
            before=ReviewStatus.IN_REVIEW,
            after=ReviewStatus.FALSE_POSITIVE,
            reason="Shadow misread as a rider",
        )
    )

    reloaded = ReviewStore(tmp_path)
    assert reloaded.status("evt-1") is ReviewStatus.FALSE_POSITIVE
    assert reloaded.case("evt-1", evidence_package_id="pkg-1").reason == (
        "Shadow misread as a rider"
    )


def test_a_corrupt_line_is_reported_not_skipped(tmp_path: Path) -> None:
    # Silently dropping an unreadable audit line would understate the history.
    store = ReviewStore(tmp_path)
    store.append(_entry())
    with store.journal_path("evt-1").open("a", encoding="utf-8") as handle:
        handle.write('{"entry_id": "broken"}\n')

    with pytest.raises(CorruptRecordError):
        store.history("evt-1")


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    store.append(_entry())
    with store.journal_path("evt-1").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    assert len(store.history("evt-1")) == 1


def test_the_review_journal_never_touches_the_run_tree(tmp_path: Path) -> None:
    # The whole reason this store exists: review state must not be able to collide
    # with the write-once event records.
    store = ReviewStore(tmp_path)
    store.append(_entry())
    written = {
        path.relative_to(tmp_path).parts[0]
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert written == {"reviews"}

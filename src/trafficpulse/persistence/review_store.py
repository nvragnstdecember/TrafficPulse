"""Append-only analyst-review journal (H9).

The audit record architecture-review §21 says is "maintained elsewhere" and that
:attr:`~trafficpulse.contracts.ReviewCase.audit_ref` points at. Sibling of
:class:`~trafficpulse.persistence.store.EventStore`, same posture: deterministic
files, standard library plus ``pydantic`` only, no new dependency.

Why this is *not* stored with the events
-----------------------------------------
``EventStore`` is **write-once**: ``event_id`` is a content hash and re-persisting
differing content under an existing id raises
:class:`~trafficpulse.persistence.errors.EventConflictError`, which is what makes a
replay byte-identical and an event's record trustworthy. Review state is mutable by
nature -- an analyst decides, adds a note, reopens, decides again. Writing it into
``events/`` would force one of two bad outcomes: relax write-once (and lose the
guarantee that a persisted event is exactly what the reasoner concluded), or refuse
every decision after the first. So the journal lives beside the events and never
touches them. **Nothing in this module can alter an inference result.**

Layout
------
```
<root>/reviews/<event_id>.jsonl     # one ReviewEntry per line, append-only
```
Keyed by ``event_id`` alone, deliberately **not** nested under a run. Review state
is about an event, not about the run that happened to produce it; keeping it out of
the per-run tree means a decision stays addressable even when run indexing is
unavailable, and never risks colliding with the write-once run directory.

JSON Lines rather than one growing JSON array: appending is a single ``open(...,
"a")`` with no read-modify-write, so a decision cannot corrupt or lose earlier
entries, and ordering is inherent in the file rather than something to sort by.

Status is derived, never stored
-------------------------------
:meth:`ReviewStore.case` folds the journal into a
:class:`~trafficpulse.contracts.ReviewCase`. There is no status field on disk to
disagree with the history, which is the failure mode a separate status record
always eventually hits. An event with no journal folds to ``PENDING`` -- the
absence of a decision *is* the pending state, so nothing has to be written when a
run finishes.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..contracts import ReviewCase, ReviewEntry
from ..contracts.enums import ReviewStatus
from .errors import CorruptRecordError
from .store import DEFAULT_RUN_ROOT

_REVIEWS_DIR = "reviews"

#: Deterministic id for the case derived from one event's journal. The case is a
#: fold, so its identity is the event's -- there is no second thing to name.
_CASE_ID_PREFIX = "case-"

# ``ReviewCase.created_at`` is required by the contract, but a case that has never
# been reviewed has no creation instant to report. Rather than invent "now" -- which
# would make an untouched case look freshly created on every read, and would make
# the response non-deterministic -- an unreviewed case reports the epoch the rest of
# the system already uses as its "no information" anchor.
_UNSET_CREATED_AT = datetime(1970, 1, 1, tzinfo=UTC)


def _case_id(event_id: str) -> str:
    return f"{_CASE_ID_PREFIX}{event_id}"


class ReviewStore:
    """Append-only per-event review journal, with the case derived from it.

    Constructed with the same runtime output ``root`` as
    :class:`~trafficpulse.persistence.store.EventStore` so a deployment has one
    storage location. Holds no mutable state -- a thin filesystem adapter.
    """

    def __init__(self, root: Path | str = DEFAULT_RUN_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def journal_path(self, event_id: str) -> Path:
        """The journal file for one event (may not exist)."""

        return self._root / _REVIEWS_DIR / f"{event_id}.jsonl"

    # --- writing ------------------------------------------------------------
    def append(self, entry: ReviewEntry) -> ReviewEntry:
        """Append one action to an event's journal; return it unchanged.

        The only write this module performs. There is no update and no delete:
        correcting a decision means appending the correction, which is what keeps
        the journal an audit trail rather than a mutable summary.
        """

        path = self.journal_path(entry.event_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        return entry

    # --- reading ------------------------------------------------------------
    def history(self, event_id: str) -> tuple[ReviewEntry, ...]:
        """Every recorded action for ``event_id``, oldest first.

        An event that has never been reviewed has no journal and yields ``()``.
        That is not an error: it is the pending state.
        """

        path = self.journal_path(event_id)
        if not path.is_file():
            return ()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - unreadable file is environmental
            raise CorruptRecordError(f"cannot read review journal {path}") from exc

        entries: list[ReviewEntry] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(ReviewEntry.model_validate_json(line))
            except ValidationError as exc:
                raise CorruptRecordError(
                    f"review journal {path} line {number} is not a valid ReviewEntry"
                ) from exc
        return tuple(entries)

    def status(self, event_id: str) -> ReviewStatus:
        """The current status for ``event_id`` (``PENDING`` when never reviewed)."""

        history = self.history(event_id)
        return history[-1].status_after if history else ReviewStatus.PENDING

    def statuses(self, event_ids: Iterable[str]) -> dict[str, ReviewStatus]:
        """Statuses for many events, for badging a list without an N+1 fold."""

        return {event_id: self.status(event_id) for event_id in event_ids}

    def reviewed_event_ids(self) -> frozenset[str]:
        """Every event id an analyst has acted on, from one directory listing (H11).

        The cheap half of review reporting, and the same trick H10 uses for the
        event index: a journal's *name* already records which event it belongs to,
        so "has anybody touched this event" is answerable without opening a single
        file. That matters for the historical library, which summarises review
        progress across every video in the repository -- folding each journal to a
        status there would cost one read per event, per listing.

        It answers *touched*, deliberately not *decided*: opening a case writes a
        journal without concluding anything, and only :meth:`status` can tell those
        apart. Callers that need the outcome must fold.
        """

        directory = self._root / _REVIEWS_DIR
        if not directory.is_dir():
            return frozenset()
        return frozenset(path.stem for path in directory.glob("*.jsonl"))

    def case(self, event_id: str, *, evidence_package_id: str) -> ReviewCase:
        """Fold the journal into the current :class:`ReviewCase`.

        The note carried is the most recent one an analyst actually wrote, not the
        note attached to the latest entry -- approving without retyping the note
        must not silently erase it. The same applies to ``reason``.
        """

        history = self.history(event_id)
        if not history:
            # No journal: a pending case that has never been touched. ``created_at``
            # would be a fabrication, so the epoch-free honest answer is the
            # earliest thing we know -- and we know nothing, so the case is minted
            # with the status only and no timestamps.
            return ReviewCase(
                review_case_id=_case_id(event_id),
                evidence_package_id=evidence_package_id,
                event_id=event_id,
                status=ReviewStatus.PENDING,
                created_at=_UNSET_CREATED_AT,
            )

        latest = history[-1]
        # The entry that *moved* the case into a decided state -- not merely one
        # whose resulting status happens to be decided. A note added after an
        # approval carries ``status_after == APPROVED`` too, and treating that as
        # the decision would keep advancing ``decided_at`` every time somebody
        # annotated an already-closed case.
        decided = next(
            (
                entry
                for entry in reversed(history)
                if entry.status_after.is_decided and entry.status_after != entry.status_before
            ),
            None,
        )
        note = next((entry.note for entry in reversed(history) if entry.note), None)
        reason = next((entry.reason for entry in reversed(history) if entry.reason), None)
        return ReviewCase(
            review_case_id=_case_id(event_id),
            evidence_package_id=evidence_package_id,
            event_id=event_id,
            status=latest.status_after,
            reviewer_id=latest.reviewer,
            decided_at=decided.at if decided is not None else None,
            note=note,
            reason=reason,
            updated_at=latest.at,
            audit_ref=str(self.journal_path(event_id)),
            created_at=history[0].at,
        )

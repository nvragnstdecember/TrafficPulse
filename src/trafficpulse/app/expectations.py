"""Declared expectations for a controlled demonstration, and how they are compared.

The **ground-truth side** of a controlled demo, kept rigorously apart from the
detection side. An operator declares what a hand-authored clip was built to
contain; the reasoners independently decide what it does contain; and
:func:`compare` is the only place those two ever meet.

Why this is storage and not just a UI field
--------------------------------------------
A demonstration whose expectations live in one browser tab is not reproducible:
refresh it and the claim is gone, open it on the reviewer's machine and there is
nothing to compare against. Persisting the declaration beside the video is what
lets a reviewer restart the server, reopen the clip, and see the same expected /
detected table -- which is the whole point of §18 reproducibility. It is also what
keeps the declaration auditable: it is written once, dated, and attributed.

Why it cannot leak into the detection path
-------------------------------------------
Nothing in this module is reachable from the engine, a rule, a reasoner, or the
event store. The processing service never reads it; ``ProcessRequest`` has no
field for it; and :func:`compare` consumes ``EventSummary`` objects that were
already persisted, so an expectation cannot influence -- let alone create -- a
single event. A declared family with no matching event is reported ``missing``,
loudly, rather than quietly conjured.

Storage layout
--------------
One JSON file per video under ``<storage>/expectations/``, a sibling of
``videos/`` and ``scenes/`` rather than a file inside either. A declaration is
about a *demonstration*, outlives any one run, and is deliberately not part of the
video's recovery snapshot: an unreadable or absent declaration must cost the
comparison and nothing else, never the video or the startup.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..contracts.enums import ViolationType
from .models import (
    EventSummary,
    ExpectationComparison,
    ExpectationDeclaration,
    ExpectationOutcome,
    ExpectationRecord,
    ExpectationRow,
)

_logger = logging.getLogger("trafficpulse.app.expectations")

#: Directory name under the storage root. Named for what it holds, not for the
#: feature that writes it, so a reviewer browsing the repository can tell what it
#: is without reading any code.
EXPECTATIONS_DIR = "expectations"


class ExpectationStore:
    """A per-video declaration store: one JSON file each, last write wins.

    Deliberately **not** content-addressed and **not** write-once, unlike
    ``SceneStore`` and ``EventStore``. A scene revision and a confirmed event are
    findings that history must be able to resolve forever; a declaration is a
    statement of intent that its author is entitled to correct before the demo. So
    it is a plain mutable record, and the audit value comes from ``declared_at`` /
    ``declared_by`` rather than from immutability.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, video_id: str) -> Path:
        """The file holding one video's declaration (may not exist)."""

        return self._root / f"{video_id}.json"

    def put(self, record: ExpectationRecord) -> ExpectationRecord:
        """Write (or replace) one video's declaration; return it unchanged."""

        path = self.path(record.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(), encoding="utf-8")
        return record

    def get(self, video_id: str) -> ExpectationRecord | None:
        """Read one video's declaration, or ``None`` when it has none.

        An unreadable or malformed file is logged and treated as absent: a corrupt
        declaration must cost the comparison, never the request. Reporting every
        detected family as "unexpected" is the honest degradation -- it under-claims
        rather than inventing a declaration nobody made.
        """

        path = self.path(video_id)
        if not path.is_file():
            return None
        try:
            return ExpectationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            _logger.warning("expectation for video %s is unreadable: %s", video_id, exc)
            return None

    def delete(self, video_id: str) -> bool:
        """Remove one video's declaration; return whether there was one."""

        path = self.path(video_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


def record_from(
    declaration: ExpectationDeclaration,
    *,
    video_id: str,
    now: datetime | None = None,
) -> ExpectationRecord:
    """Stamp a client declaration into a storable record.

    ``now`` is injectable so a test can pin the instant; it defaults to the wall
    clock because "when did somebody declare this" is genuinely a wall-clock fact,
    unlike every media-time value in this system.
    """

    return ExpectationRecord(
        video_id=video_id,
        expected_violations=declaration.expected_violations,
        notes=declaration.notes,
        declared_by=declaration.declared_by,
        declared_at=now if now is not None else datetime.now(UTC),
    )


def compare(
    *,
    video_id: str,
    job_id: str | None,
    expectation: ExpectationRecord | None,
    events: Sequence[EventSummary],
) -> ExpectationComparison:
    """Compare a declaration with what a run actually confirmed. Pure.

    ``events`` are already-persisted summaries, read through the same
    ``EventService`` path the workspace lists from -- so the "detected" column of
    the demo table and the event list a reviewer clicks through cannot disagree.

    Rows cover every family that is expected **or** detected, in the fixed
    ``ViolationType`` declaration order, so the table does not reshuffle between
    requests. A family that is neither expected nor detected is omitted rather than
    listed as a zero: a demonstration's table should show what was claimed and what
    happened, not the whole ontology.

    No accuracy is computed. See :class:`ExpectationComparison` for why.
    """

    expected: frozenset[ViolationType] = (
        frozenset(expectation.expected_violations) if expectation is not None else frozenset()
    )
    detected: dict[ViolationType, list[str]] = {}
    for event in events:
        detected.setdefault(event.violation_type, []).append(event.event_id)

    rows: list[ExpectationRow] = []
    for violation in ViolationType:
        is_expected = violation in expected
        ids = tuple(detected.get(violation, ()))
        if not is_expected and not ids:
            continue
        if is_expected and ids:
            outcome = ExpectationOutcome.MATCHED
        elif is_expected:
            outcome = ExpectationOutcome.MISSING
        else:
            outcome = ExpectationOutcome.UNEXPECTED
        rows.append(
            ExpectationRow(
                violation_type=violation,
                expected=is_expected,
                detected_count=len(ids),
                event_ids=ids,
                outcome=outcome,
            )
        )

    return ExpectationComparison(
        video_id=video_id,
        job_id=job_id,
        expectation=expectation,
        rows=tuple(rows),
        expected_count=len(expected),
        detected_event_count=len(events),
        matched_count=sum(1 for row in rows if row.outcome is ExpectationOutcome.MATCHED),
        missing_count=sum(1 for row in rows if row.outcome is ExpectationOutcome.MISSING),
        unexpected_count=sum(1 for row in rows if row.outcome is ExpectationOutcome.UNEXPECTED),
    )


def detected_families(events: Iterable[EventSummary]) -> tuple[ViolationType, ...]:
    """The distinct families a run confirmed, in fixed order (for callers/logs)."""

    present = {event.violation_type for event in events}
    return tuple(violation for violation in ViolationType if violation in present)

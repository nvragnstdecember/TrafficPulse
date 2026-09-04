"""The expectation store and the expected-vs-detected comparison, in isolation.

Fast unit coverage of the pieces the controlled-demo e2e exercises end to end: the
per-video declaration store, the pure comparison, and the API-level guard rails.
Nothing here decodes a video or runs an engine -- the point is the *bookkeeping*,
and keeping it separate is what lets these cases be exhaustive without paying for a
processing run each.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config
from fastapi.testclient import TestClient
from pydantic import ValidationError

from trafficpulse.app.expectations import (
    ExpectationStore,
    compare,
    detected_families,
    record_from,
)
from trafficpulse.app.models import (
    EventSummary,
    ExpectationDeclaration,
    ExpectationOutcome,
    ExpectationRecord,
)
from trafficpulse.contracts import ConfidenceBreakdown
from trafficpulse.contracts.enums import ViolationType

_PINNED = datetime(2026, 1, 1, tzinfo=UTC)


def _summary(violation: ViolationType, event_id: str) -> EventSummary:
    return EventSummary(
        event_id=event_id,
        video_id="vid-1",
        job_id="job-1",
        violation_type=violation,
        camera_id="cam-1",
        track_ids=("t-1",),
        start_at=_PINNED,
        trigger_at=_PINNED,
        rule_id="rule-1",
        confidence=ConfidenceBreakdown(),
    )


def _record(*violations: ViolationType, video_id: str = "vid-1") -> ExpectationRecord:
    return record_from(
        ExpectationDeclaration(expected_violations=violations, notes="controlled demo"),
        video_id=video_id,
        now=_PINNED,
    )


# --- the store --------------------------------------------------------------------
def test_a_video_with_no_declaration_reads_back_as_none(tmp_path: Path) -> None:
    assert ExpectationStore(tmp_path).get("vid-1") is None


def test_a_declaration_round_trips(tmp_path: Path) -> None:
    store = ExpectationStore(tmp_path)
    stored = store.put(_record(ViolationType.WRONG_WAY, ViolationType.TRIPLE_RIDING))

    assert store.get("vid-1") == stored
    assert stored.declared_at == _PINNED


def test_redeclaring_replaces_rather_than_appends(tmp_path: Path) -> None:
    """A declaration is intent, which its author may correct -- unlike a finding."""

    store = ExpectationStore(tmp_path)
    store.put(_record(ViolationType.WRONG_WAY))
    store.put(_record(ViolationType.TRIPLE_RIDING))

    read = store.get("vid-1")
    assert read is not None
    assert read.expected_violations == (ViolationType.TRIPLE_RIDING,)


def test_a_corrupt_declaration_is_treated_as_absent(tmp_path: Path) -> None:
    """A broken file must cost the comparison, never the request.

    Reporting "nothing was declared" under-claims; inventing a declaration or
    raising would be worse in opposite directions.
    """

    store = ExpectationStore(tmp_path)
    store.put(_record(ViolationType.WRONG_WAY))
    store.path("vid-1").write_text("{ not json", encoding="utf-8")

    assert store.get("vid-1") is None


def test_deleting_reports_whether_there_was_anything_to_delete(tmp_path: Path) -> None:
    store = ExpectationStore(tmp_path)
    assert store.delete("vid-1") is False
    store.put(_record(ViolationType.WRONG_WAY))
    assert store.delete("vid-1") is True
    assert store.get("vid-1") is None


def test_declarations_are_stored_one_file_per_video(tmp_path: Path) -> None:
    store = ExpectationStore(tmp_path)
    store.put(_record(ViolationType.WRONG_WAY, video_id="vid-1"))
    store.put(_record(ViolationType.TRIPLE_RIDING, video_id="vid-2"))

    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["vid-1.json", "vid-2.json"]
    first = store.get("vid-1")
    assert first is not None and first.expected_violations == (ViolationType.WRONG_WAY,)


# --- the declaration model ---------------------------------------------------------
def test_a_repeated_family_is_rejected() -> None:
    """A duplicate would double-count in the summary and read as two findings."""

    with pytest.raises(ValidationError):
        ExpectationDeclaration(
            expected_violations=(ViolationType.WRONG_WAY, ViolationType.WRONG_WAY)
        )


def test_an_empty_declaration_is_legitimate() -> None:
    """"This clip was built to contain nothing" is a real claim, and a useful one."""

    declaration = ExpectationDeclaration()
    assert declaration.expected_violations == ()


# --- the comparison ----------------------------------------------------------------
def test_matched_missing_and_unexpected_are_each_reported() -> None:
    comparison = compare(
        video_id="vid-1",
        job_id="job-1",
        expectation=_record(ViolationType.WRONG_WAY, ViolationType.RED_LIGHT_JUMPING),
        events=[
            _summary(ViolationType.WRONG_WAY, "evt-a"),
            _summary(ViolationType.TRIPLE_RIDING, "evt-b"),
        ],
    )

    outcomes = {row.violation_type: row.outcome for row in comparison.rows}
    assert outcomes[ViolationType.WRONG_WAY] is ExpectationOutcome.MATCHED
    assert outcomes[ViolationType.RED_LIGHT_JUMPING] is ExpectationOutcome.MISSING
    assert outcomes[ViolationType.TRIPLE_RIDING] is ExpectationOutcome.UNEXPECTED
    assert (comparison.matched_count, comparison.missing_count) == (1, 1)
    assert comparison.unexpected_count == 1
    assert comparison.expected_count == 2
    assert comparison.detected_event_count == 2


def test_a_family_that_is_neither_expected_nor_detected_is_omitted() -> None:
    """A demonstration table shows what was claimed and what happened, not the ontology."""

    comparison = compare(
        video_id="vid-1",
        job_id=None,
        expectation=_record(ViolationType.WRONG_WAY),
        events=[_summary(ViolationType.WRONG_WAY, "evt-a")],
    )

    assert [row.violation_type for row in comparison.rows] == [ViolationType.WRONG_WAY]


def test_several_events_of_one_family_count_once_as_matched_but_report_every_id() -> None:
    """Matching is per family; the count and the ids stay per event."""

    comparison = compare(
        video_id="vid-1",
        job_id="job-1",
        expectation=_record(ViolationType.WRONG_WAY),
        events=[
            _summary(ViolationType.WRONG_WAY, "evt-a"),
            _summary(ViolationType.WRONG_WAY, "evt-b"),
        ],
    )

    (row,) = comparison.rows
    assert row.detected_count == 2
    assert row.event_ids == ("evt-a", "evt-b")
    assert comparison.matched_count == 1
    assert comparison.detected_event_count == 2


def test_no_declaration_makes_every_detection_unexpected() -> None:
    comparison = compare(
        video_id="vid-1",
        job_id=None,
        expectation=None,
        events=[_summary(ViolationType.TRIPLE_RIDING, "evt-a")],
    )

    assert comparison.expectation is None
    assert comparison.expected_count == 0
    assert comparison.unexpected_count == 1


def test_rows_follow_the_fixed_violation_order() -> None:
    """A table that reshuffles between requests is unreadable during a demo."""

    comparison = compare(
        video_id="vid-1",
        job_id=None,
        expectation=_record(*ViolationType),
        events=[],
    )

    assert [row.violation_type for row in comparison.rows] == list(ViolationType)


def test_the_comparison_carries_no_accuracy_metric() -> None:
    """Precision over one hand-authored clip would be arithmetic on its own answer."""

    fields = set(
        compare(video_id="vid-1", job_id=None, expectation=None, events=[]).model_dump()
    )
    assert not fields & {"precision", "recall", "f1", "accuracy", "score"}


def test_detected_families_reports_distinct_types_in_fixed_order() -> None:
    families = detected_families(
        [
            _summary(ViolationType.TRIPLE_RIDING, "evt-a"),
            _summary(ViolationType.WRONG_WAY, "evt-b"),
            _summary(ViolationType.WRONG_WAY, "evt-c"),
        ]
    )
    assert families == (ViolationType.TRIPLE_RIDING, ViolationType.WRONG_WAY)


# --- the API surface ---------------------------------------------------------------
@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path, config=make_config(tmp_path, scene_path=None))


def test_declaring_against_an_unknown_video_is_a_404(client: TestClient) -> None:
    """A declaration can never be attached to an id nothing can produce events for."""

    response = client.put(
        "/api/videos/nope/expectation", json={"expected_violations": ["wrong_way"]}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "video_not_found"


def test_reading_a_missing_declaration_is_a_404_naming_the_expectation(
    client: TestClient, tmp_path: Path
) -> None:
    from _app_helpers import upload_wrong_way_video

    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.get(f"/api/videos/{video_id}/expectation")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "expectation_not_found"


def test_a_malformed_declaration_is_a_422(client: TestClient, tmp_path: Path) -> None:
    from _app_helpers import upload_wrong_way_video

    video_id = upload_wrong_way_video(client, tmp_path)
    response = client.put(
        f"/api/videos/{video_id}/expectation",
        json={"expected_violations": ["not_a_violation"]},
    )
    assert response.status_code == 422


def test_the_process_request_has_no_expectation_field() -> None:
    """Structural, not aspirational: the engine's input cannot carry ground truth.

    If an ``expected``/``expectation`` field ever appears on ``ProcessRequest``, the
    separation stops being enforced by the type system and starts being a promise.
    """

    from trafficpulse.app.models import ProcessRequest

    fields = set(ProcessRequest.model_fields)
    assert not any("expect" in name for name in fields), fields


def test_the_engine_config_cannot_carry_an_expectation() -> None:
    """The other half of the same guarantee, one layer down."""

    from trafficpulse.engine import EngineConfig

    assert not any("expect" in name for name in EngineConfig.model_fields)

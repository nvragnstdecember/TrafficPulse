"""The ``TemporalRunReasoner`` episode-enricher hook (P4-U5).

The hook touches **shared** reasoning code that all three previously shipped
reasoners (wrong-way P1-U4, illegal-stopping P2-U4, red-light-adjacent P3 work)
run through, so the load-bearing tests here are the regression ones: with no
enricher injected, confirmation must behave exactly as it did before P4-U5.

The rest pins the hook's own contract -- in particular that enrichment happens
BEFORE ``event_id`` is computed, since ``track_ids`` is identity-bearing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.contracts import (
    ConfidenceBreakdown,
    HeadingVsLaneObservation,
    MeasuredValue,
    Producer,
)
from trafficpulse.contracts.enums import ProducerKind, ViolationType
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.temporal import (
    ConfirmationDetails,
    EpisodeExtras,
    TemporalRunReasoner,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="test", version="1", kind=ProducerKind.HEURISTIC)


def obs(second: float, *, track_id: str = "t-1") -> HeadingVsLaneObservation:
    return HeadingVsLaneObservation(
        observation_id=f"obs-{track_id}-{second}",
        camera_id="cam-1",
        track_id=track_id,
        timestamp=BASE + timedelta(seconds=second),
        producer=PRODUCER,
        lane_id="lane-1",
        heading_degrees=180.0,
        deviation_degrees=180.0,
        is_contradiction=True,
    )


def _details(start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
    return ConfirmationDetails(
        measurements=(
            MeasuredValue(
                name="persistence_seconds", value=(trigger_at - start_at).total_seconds()
            ),
        ),
        thresholds=(MeasuredValue(name="min_persistence", value=1.0),),
    )


def machine(enricher=None) -> TemporalRunReasoner:
    return TemporalRunReasoner(
        RuleEngine(),
        violation_type=ViolationType.WRONG_WAY,
        threshold_seconds=1.0,
        detail_builder=_details,
        scene_config_hash="a" * 64,
        rule_id="test_rule",
        rule_version="1",
        episode_enricher=enricher,
    )


def _steps() -> list[tuple[HeadingVsLaneObservation, bool]]:
    return [(obs(s), True) for s in (0.0, 0.5, 1.0, 1.5)]


# --- regression: no enricher = pre-P4-U5 behaviour ---------------------------
def test_default_is_no_enricher() -> None:
    events = machine().run(_steps())
    assert len(events) == 1


def test_without_an_enricher_track_ids_come_from_the_record_alone() -> None:
    assert machine().run(_steps())[0].track_ids == ("t-1",)


def test_without_an_enricher_confidence_is_the_contract_default() -> None:
    """Byte-identical to a pre-P4-U5 event, which never set confidence."""

    assert machine().run(_steps())[0].confidence == ConfidenceBreakdown()


def test_without_an_enricher_measurements_are_the_detail_builders_alone() -> None:
    events = machine().run(_steps())
    assert [m.name for m in events[0].measurements] == ["persistence_seconds"]


def test_an_inert_enricher_is_byte_identical_to_no_enricher() -> None:
    """The hook adds nothing unless the enricher supplies something."""

    baseline = machine().run(_steps())[0]
    enriched = machine(lambda _t, _s, _g: EpisodeExtras()).run(_steps())[0]

    assert enriched.model_dump_json() == baseline.model_dump_json()


# --- the hook's contract -----------------------------------------------------
def test_related_track_ids_are_merged_and_sorted() -> None:
    events = machine(lambda _t, _s, _g: EpisodeExtras(related_track_ids=("bike-9",))).run(
        _steps()
    )
    assert events[0].track_ids == ("bike-9", "t-1")


def test_enrichment_changes_the_event_id() -> None:
    """track_ids is identity-bearing: a different subject IS a different event.

    This is why enrichment must run before the id is computed -- patching
    ``track_ids`` afterwards would leave the id contradicting its own content.
    """

    baseline = machine().run(_steps())[0]
    enriched = machine(lambda _t, _s, _g: EpisodeExtras(related_track_ids=("bike-9",))).run(
        _steps()
    )[0]

    assert enriched.event_id != baseline.event_id


def test_event_id_is_stable_across_related_track_id_order() -> None:
    def run(ids: tuple[str, ...]) -> str:
        return machine(lambda _t, _s, _g: EpisodeExtras(related_track_ids=ids)).run(_steps())[
            0
        ].event_id

    assert run(("a", "b")) == run(("b", "a"))


def test_enricher_receives_the_episode_track_and_window() -> None:
    seen: list[tuple[str, datetime, datetime]] = []

    def enricher(track_id: str, start_at: datetime, trigger_at: datetime) -> EpisodeExtras:
        seen.append((track_id, start_at, trigger_at))
        return EpisodeExtras()

    machine(enricher).run(_steps())

    assert len(seen) == 1
    track_id, start_at, trigger_at = seen[0]
    assert track_id == "t-1"
    assert start_at == BASE
    assert (trigger_at - start_at).total_seconds() == pytest.approx(1.0)


def test_confidence_is_carried_onto_the_event() -> None:
    breakdown = ConfidenceBreakdown(classifier=0.8, temporal_consistency=0.5)
    events = machine(lambda _t, _s, _g: EpisodeExtras(confidence=breakdown)).run(_steps())

    assert events[0].confidence == breakdown


def test_extra_measurements_append_to_the_detail_builders() -> None:
    extra = MeasuredValue(name="confirming_observations", value=3.0, unit="count")
    events = machine(lambda _t, _s, _g: EpisodeExtras(measurements=(extra,))).run(_steps())

    assert [m.name for m in events[0].measurements] == [
        "persistence_seconds",
        "confirming_observations",
    ]


def test_thresholds_are_untouched_by_enrichment() -> None:
    events = machine(
        lambda _t, _s, _g: EpisodeExtras(measurements=(MeasuredValue(name="x", value=1.0),))
    ).run(_steps())

    assert [t.name for t in events[0].thresholds] == ["min_persistence"]


def test_enricher_is_not_called_when_nothing_confirms() -> None:
    calls: list[str] = []

    def enricher(track_id: str, _s: datetime, _g: datetime) -> EpisodeExtras:
        calls.append(track_id)
        return EpisodeExtras()

    machine(enricher).run([(obs(0.0), True), (obs(0.4), True)])  # under threshold

    assert calls == []


def test_episode_extras_defaults_are_empty() -> None:
    extras = EpisodeExtras()
    assert extras.related_track_ids == ()
    assert extras.confidence is None
    assert extras.measurements == ()

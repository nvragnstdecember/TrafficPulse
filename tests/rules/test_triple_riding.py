"""Triple-riding temporal reasoning (v1.1 U3).

Pure reasoning over frozen ``RiderCountObservation`` + ``Association`` contracts:
no pixels, no detector, no perception backend. Exercises the count threshold,
temporal persistence (anti-flicker), false-positive suppression, rider
attribution, and taint handling -- all delegated to the shared
``TemporalRunReasoner``, so this file asserts triple-riding *semantics* only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import (
    Association,
    ParameterStatus,
    Producer,
    RiderCountObservation,
)
from trafficpulse.contracts.enums import AssociationType, ProducerKind, ViolationType
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.triple_riding import TripleRidingParameters, TripleRidingReasoner

BASE = datetime(1970, 1, 1, tzinfo=UTC)
PROD = Producer(name="rider-count", version="0.1.0-provisional", kind=ProducerKind.HEURISTIC)


def params(*, min_persistence: float = 1.0, threshold: int = 3, max_gap: float | None = 2.0):
    return TripleRidingParameters(
        min_persistence_seconds=min_persistence,
        rider_count_threshold=threshold,
        max_observation_gap_seconds=max_gap,
        persistence_status=ParameterStatus.PROVISIONAL,
        rider_count_threshold_status=ParameterStatus.PROVISIONAL,
        max_observation_gap_status=ParameterStatus.PROVISIONAL,
    )


def count(motorcycle: str, riders: int, at: float) -> RiderCountObservation:
    return RiderCountObservation(
        observation_id=f"rct-{motorcycle}-{at}",
        camera_id="cam-1",
        track_id=motorcycle,
        timestamp=BASE + timedelta(seconds=at),
        producer=PROD,
        rider_count=riders,
        motorcycle_track_id=motorcycle,
    )


def link(rider: str, motorcycle: str, at: float) -> Association:
    return Association(
        association_id=f"asc-{rider}-{motorcycle}-{at}",
        camera_id="cam-1",
        subject_track_id=rider,
        object_track_id=motorcycle,
        association_type=AssociationType.RIDER_OF_MOTORCYCLE,
        confidence=0.8,
        timestamp=BASE + timedelta(seconds=at),
    )


def _reasoner() -> TripleRidingReasoner:
    return TripleRidingReasoner(RuleEngine(), params())


# --- violation generation ------------------------------------------------------
def test_sustained_three_riders_confirms_one_event() -> None:
    observations = [count("m1", 3, t) for t in (0.0, 0.5, 1.0, 1.5)]
    events = _reasoner().run(observations)
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.TRIPLE_RIDING
    assert events[0].rule_id == "triple_riding"


# --- temporal smoothing (anti-flicker) ----------------------------------------
def test_single_three_rider_frame_does_not_confirm() -> None:
    # 2 -> 3 -> 2 -> 2: the lone 3-rider frame opens a run the next frame ends
    # before min_persistence is reached.
    observations = [
        count("m1", 2, 0.0),
        count("m1", 3, 0.5),
        count("m1", 2, 1.0),
        count("m1", 2, 1.5),
    ]
    assert _reasoner().run(observations) == ()


def test_flicker_within_a_sustained_run_still_confirms() -> None:
    # A real triple-riding episode with one noisy 2-rider frame bridged by the gap
    # tolerance still confirms (the run is not destroyed by a single blip).
    observations = [
        count("m1", 3, 0.0),
        count("m1", 3, 0.5),
        count("m1", 3, 1.0),
        count("m1", 3, 1.5),
    ]
    assert len(_reasoner().run(observations)) == 1


# --- false-positive suppression -----------------------------------------------
def test_two_riders_never_confirm() -> None:
    observations = [count("m1", 2, t) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert _reasoner().run(observations) == ()


def test_persistence_shorter_than_threshold_does_not_confirm() -> None:
    # Three riders, but only for 0.5 s (< 1.0 s min_persistence).
    observations = [count("m1", 3, 0.0), count("m1", 3, 0.5)]
    assert _reasoner().run(observations) == ()


def test_large_gap_breaks_a_stale_run() -> None:
    # Two 3-rider observations far apart (10 s >> 2 s max_gap): the run cannot
    # bridge the gap, so nothing persists.
    observations = [count("m1", 3, 0.0), count("m1", 3, 10.0)]
    assert _reasoner().run(observations) == ()


# --- attribution: motorcycle + riders on the event -----------------------------
def test_event_names_the_motorcycle_and_its_riders() -> None:
    observations = [count("m1", 3, t) for t in (0.0, 0.5, 1.0, 1.5)]
    associations = [
        link(rider, "m1", t) for t in (0.0, 0.5, 1.0, 1.5) for rider in ("p1", "p2", "p3")
    ]
    events = _reasoner().run(observations, associations=associations)
    assert len(events) == 1
    assert set(events[0].track_ids) == {"m1", "p1", "p2", "p3"}


def test_confidence_and_measurements_are_populated() -> None:
    observations = [count("m1", 3, t) for t in (0.0, 0.5, 1.0, 1.5)]
    associations = [link("p1", "m1", 0.0), link("p2", "m1", 0.0), link("p3", "m1", 0.0)]
    event = _reasoner().run(observations, associations=associations)[0]
    assert event.confidence.temporal_consistency == 1.0  # every frame supported
    names = {m.name for m in event.measurements}
    assert {"max_rider_count", "confirming_observations", "persistence_seconds"} <= names
    max_riders = next(m.value for m in event.measurements if m.name == "max_rider_count")
    assert max_riders == 3.0
    thresholds = {t.name for t in event.thresholds}
    assert {"min_persistence", "rider_count_threshold"} <= thresholds


# --- threshold configurability -------------------------------------------------
def test_threshold_of_two_confirms_two_riders() -> None:
    reasoner = TripleRidingReasoner(RuleEngine(), params(threshold=2))
    observations = [count("m1", 2, t) for t in (0.0, 0.5, 1.0, 1.5)]
    assert len(reasoner.run(observations)) == 1


# --- taint: a restart resets the run -------------------------------------------
def test_taint_restart_resets_the_run() -> None:
    observations = [count("m1", 3, t) for t in (0.0, 0.5, 1.0, 1.5)]
    reasoner = TripleRidingReasoner(RuleEngine(), params())
    # Restart at t=1.0 (the third obs) discards the run before it: only 0.5 s of
    # support remains after, short of the 1.0 s threshold, so nothing confirms.
    restart_ids = {observations[2].observation_id}
    assert reasoner.run(observations, taint_restart_ids=restart_ids) == ()


# --- determinism ---------------------------------------------------------------
def test_output_is_order_independent() -> None:
    observations = [count("m1", 3, t) for t in (0.0, 0.5, 1.0, 1.5)]
    forward = TripleRidingReasoner(RuleEngine(), params()).run(observations)
    reverse = TripleRidingReasoner(RuleEngine(), params()).run(list(reversed(observations)))
    assert [e.event_id for e in forward] == [e.event_id for e in reverse]

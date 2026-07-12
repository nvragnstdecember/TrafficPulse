"""Tests for the generalized temporal-run reasoner base (P3-U1, composition).

Exercises :class:`~trafficpulse.rules.temporal.TemporalRunReasoner` directly,
independently of any violation, to pin the shared machine the two shipped
reasoners delegate to: the per-``(camera, track)`` support-run lifecycle, the
optional gap-break, taint reset, the >=2-observation confirmation floor,
run-level ``models`` stamping, and content-derived ``event_id`` determinism. The
base must carry no violation-specific vocabulary, so a ``SPEEDING`` carrier --
handled by neither shipped reasoner -- drives every case here.

Synthetic ``StationaryObservation`` carriers only (used opaquely as generic
``ObservationBase`` steps); no detector, tracker, or video.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trafficpulse.contracts import (
    ConfirmedEvent,
    MeasuredValue,
    ModelRef,
    Producer,
    StationaryObservation,
)
from trafficpulse.contracts.enums import ProducerKind, ViolationType
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.states import EngineState
from trafficpulse.rules.temporal import ConfirmationDetails, TemporalRunReasoner

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)
SCH = "a" * 64  # a valid scene_config_hash (64-char lowercase hex)
THRESHOLD = 1.0  # seconds


def _carrier(
    i: int, *, track: str = "tk", camera: str = "cam-1", spacing: float = 0.1
) -> StationaryObservation:
    return StationaryObservation(
        observation_id=f"o-{track}-{i:03d}",
        camera_id=camera,
        track_id=track,
        timestamp=BASE + timedelta(seconds=i * spacing),
        producer=PRODUCER,
        is_stationary=True,
    )


def _details(start_at: datetime, trigger_at: datetime) -> ConfirmationDetails:
    return ConfirmationDetails(
        measurements=(
            MeasuredValue(
                name="support_seconds",
                value=(trigger_at - start_at).total_seconds(),
                unit="seconds",
            ),
        ),
        thresholds=(MeasuredValue(name="threshold", value=THRESHOLD, unit="seconds"),),
    )


def _reasoner(
    *, models: tuple[ModelRef, ...] = (), max_gap: float | None = None
) -> TemporalRunReasoner:
    return TemporalRunReasoner(
        RuleEngine(),
        violation_type=ViolationType.SPEEDING,
        threshold_seconds=THRESHOLD,
        detail_builder=_details,
        scene_config_hash=SCH,
        rule_id="generic_rule",
        rule_version="0.0.1",
        max_observation_gap_seconds=max_gap,
        models=models,
    )


def _steps(active_flags: list[bool]) -> list[tuple[StationaryObservation, bool]]:
    return [(_carrier(i), flag) for i, flag in enumerate(active_flags)]


# --- run lifecycle -----------------------------------------------------------
def test_sustained_support_confirms_once() -> None:
    r = _reasoner()
    events = r.run(_steps([True] * 11))  # t: 0.0 .. 1.0 -> confirm
    assert len(events) == 1
    assert events[0].violation_type is ViolationType.SPEEDING


def test_confirmation_requires_at_least_two_observations() -> None:
    # A single active step cannot confirm (a run needs a later observation than
    # the one that opened it) -- architecture-review §13.
    r = _reasoner()
    assert r.observe(_carrier(0), active=True) is None
    assert r.events == ()


def test_threshold_boundary_inclusive() -> None:
    at = _reasoner()
    assert len(at.run(_steps([True] * 11))) == 1  # 0.0 .. 1.0 -> confirm
    under = _reasoner()
    assert under.run(_steps([True] * 10)) == ()  # 0.0 .. 0.9 -> no


def test_recovery_before_threshold_does_not_confirm() -> None:
    r = _reasoner()
    events = r.run(_steps([True, True, True, False, True, True]))
    assert events == ()
    records = r.engine.records()
    assert len(records) == 2  # abandoned run + a fresh short run
    assert any(rec.state is EngineState.ABANDONED for rec in records)


def test_inactive_only_never_confirms() -> None:
    r = _reasoner()
    assert r.run(_steps([False] * 20)) == ()
    assert r.engine.records() == ()  # no run ever opened


def test_only_one_event_per_episode() -> None:
    r = _reasoner()
    events = r.run(_steps([True] * 30))
    assert len(events) == 1


def test_confirmed_hypothesis_is_active() -> None:
    r = _reasoner()
    events = r.run(_steps([True] * 11))
    record = r.engine.get(events[0].source_hypothesis_id)
    assert record is not None
    assert record.state is EngineState.ACTIVE


# --- taint reset -------------------------------------------------------------
def test_taint_restart_breaks_support_run() -> None:
    # Support before a taint cannot bridge to support after it: the restart ends
    # the open run, so a would-be confirmation is split into two sub-threshold
    # runs and nothing confirms.
    r = _reasoner()
    r.observe(_carrier(0), active=True)
    r.observe(_carrier(5), active=True)  # 0.5 s in, still below threshold
    # Restart at 0.6 s resets; only 0.4 s of clean support remains afterwards.
    r.observe(_carrier(6), active=True, is_taint_restart=True)
    for i in range(7, 11):
        r.observe(_carrier(i), active=True)
    assert r.events == ()
    assert any(rec.state is EngineState.ABANDONED for rec in r.engine.records())


def test_clean_run_after_taint_confirms_from_restart() -> None:
    r = _reasoner()
    r.observe(_carrier(0), active=True)
    r.observe(_carrier(6), active=True, is_taint_restart=True)  # fresh run starts here
    events = [r.observe(_carrier(i), active=True) for i in range(7, 17)]
    confirmed = [e for e in events if e is not None]
    assert len(confirmed) == 1
    # Persistence is measured from the post-taint restart, not the pre-taint start.
    assert confirmed[0].start_at == _carrier(6).timestamp


# --- optional gap-break ------------------------------------------------------
def test_gap_break_splits_run_when_configured() -> None:
    # With a 0.25 s max gap, a 0.5 s hole ends the run; the post-gap support alone
    # is below threshold, so nothing confirms.
    r = _reasoner(max_gap=0.25)
    r.observe(_carrier(0), active=True)
    r.observe(_carrier(5), active=True)  # 0.5 s gap > 0.25 -> run broken, fresh run opens
    for i in range(6, 10):
        r.observe(_carrier(i), active=True)  # only 0.4 s of fresh support
    assert r.events == ()


def test_no_gap_break_when_unset_bridges_timestamp() -> None:
    # The identical hole with max_gap unset bridges by timestamp and confirms.
    r = _reasoner(max_gap=None)
    r.observe(_carrier(0), active=True)
    r.observe(_carrier(5), active=True)
    events = [r.observe(_carrier(i), active=True) for i in range(6, 12)]
    assert len([e for e in events if e is not None]) == 1


# --- determinism & ordering --------------------------------------------------
def test_event_id_is_deterministic_across_instances() -> None:
    a = _reasoner().run(_steps([True] * 11))[0]
    b = _reasoner().run(_steps([True] * 11))[0]
    assert a.event_id == b.event_id
    assert a.model_dump_json() == b.model_dump_json()


def test_out_of_order_matches_in_order() -> None:
    ordered = _steps([True] * 15)
    shuffled = [ordered[i] for i in (7, 0, 14, 3, 9, 1, 12, 5, 2, 11, 8, 4, 13, 6, 10)]
    a = _reasoner().run(ordered)
    b = _reasoner().run(shuffled)
    assert len(a) == len(b) == 1
    assert a[0].event_id == b[0].event_id


def test_duplicate_carriers_are_idempotent() -> None:
    base = _steps([True] * 11)
    plain = _reasoner().run(base)
    dup = _reasoner().run(base + [(_carrier(10), True), (_carrier(10), True)])
    assert len(dup) == 1
    assert dup[0].event_id == plain[0].event_id


def test_replay_same_reasoner_no_duplicate() -> None:
    steps = _steps([True] * 11)
    r = _reasoner()
    first = r.run(steps)
    second = r.run(steps)
    assert len(first) == 1
    assert second == ()


def test_untracked_carrier_is_ignored() -> None:
    r = _reasoner()
    untracked = StationaryObservation(
        observation_id="o-none",
        camera_id="cam-1",
        track_id=None,
        timestamp=BASE,
        producer=PRODUCER,
        is_stationary=True,
    )
    assert r.observe(untracked, active=True) is None
    assert r.engine.records() == ()


# --- injected payload & provenance -------------------------------------------
def test_detail_builder_payload_flows_through() -> None:
    event = _reasoner().run(_steps([True] * 11))[0]
    measurements = {m.name: m.value for m in event.measurements}
    thresholds = {t.name: t.value for t in event.thresholds}
    assert "support_seconds" in measurements
    assert thresholds["threshold"] == THRESHOLD


def test_models_are_stamped_but_excluded_from_event_id() -> None:
    model = ModelRef(name="detector", version="1.0")
    with_models = _reasoner(models=(model,)).run(_steps([True] * 11))[0]
    without = _reasoner().run(_steps([True] * 11))[0]
    assert with_models.models == (model,)
    assert without.models == ()
    # Provenance never enters the identity preimage: same decision, same id.
    assert with_models.event_id == without.event_id


def test_events_are_confirmed_contracts() -> None:
    event = _reasoner().run(_steps([True] * 11))[0]
    assert isinstance(event, ConfirmedEvent)
    assert ConfirmedEvent.model_validate_json(event.model_dump_json()) == event
    assert event.scene_config_hash == SCH
    assert event.created_at == event.trigger_at  # deterministic, never wall-clock


# --- no violation-specific vocabulary in the base ----------------------------
def test_base_carries_no_violation_literals() -> None:
    import trafficpulse.rules.temporal as temporal_mod

    source = Path(temporal_mod.__file__).read_text(encoding="utf-8").lower()
    for token in ("wrong_way", "wrong-way", "illegal_stopping", "red_light", "red-light",
                  "is_contradiction", "stopped_in_zone", "heading", "stationary_duration"):
        assert token not in source, f"base leaked violation token {token!r}"

"""No-helmet temporal reasoning (P4-U5).

Reasoning over frozen contracts only: no pixels, no classifier, no ML. Every
observation here is constructed directly, so these tests pin the *rule*, not the
perception that fed it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import (
    Association,
    HelmetStateObservation,
    Producer,
    SceneConfig,
)
from trafficpulse.contracts.enums import (
    AssociationType,
    HelmetState,
    ProducerKind,
    ViolationType,
)
from trafficpulse.contracts.scene import ParameterStatus
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.no_helmet import (
    NoHelmetParameters,
    NoHelmetReasoner,
    no_helmet_parameters,
)

# Resolved locally rather than imported from tests/pipeline/_pipeline_helpers:
# pytest's prepend import mode only puts a test's OWN directory on sys.path.
SCENE_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example-scene.yaml"

BASE = datetime(1970, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="test", version="1", kind=ProducerKind.MODEL)

PARAMS = NoHelmetParameters(
    min_persistence_seconds=1.0,
    max_observation_gap_seconds=2.0,
    min_confirming_observations=2,
    persistence_status=ParameterStatus.PROVISIONAL,
    max_observation_gap_status=ParameterStatus.PROVISIONAL,
    min_confirming_observations_status=ParameterStatus.PROVISIONAL,
)


def obs(
    second: float,
    state: HelmetState,
    *,
    track_id: str = "rider-1",
    confidence: float | None = 0.9,
    crop_height_px: float | None = 40.0,
) -> HelmetStateObservation:
    return HelmetStateObservation(
        observation_id=f"hlm-{track_id}-{second}",
        camera_id="cam-1",
        track_id=track_id,
        timestamp=BASE + timedelta(seconds=second),
        confidence=confidence,
        producer=PRODUCER,
        helmet_state=state,
        crop_height_px=crop_height_px,
    )


def link(second: float, *, rider: str = "rider-1", bike: str = "bike-9", confidence: float = 0.8):
    return Association(
        association_id=f"asc-{rider}-{bike}-{second}",
        camera_id="cam-1",
        subject_track_id=rider,
        object_track_id=bike,
        association_type=AssociationType.RIDER_OF_MOTORCYCLE,
        confidence=confidence,
        timestamp=BASE + timedelta(seconds=second),
    )


def reasoner(params: NoHelmetParameters = PARAMS) -> NoHelmetReasoner:
    return NoHelmetReasoner(RuleEngine(), params, scene_config_hash="a" * 64)


# --- sustained no-helmet -----------------------------------------------------
def test_sustained_no_helmet_confirms_one_event() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(s) for s in (0.0, 0.5, 1.0, 1.5)],
    )

    assert len(events) == 1
    assert events[0].violation_type is ViolationType.NO_HELMET
    assert events[0].rule_id == "no_helmet"


def test_support_shorter_than_min_persistence_does_not_confirm() -> None:
    events = reasoner().run(
        [obs(0.0, HelmetState.NO_HELMET), obs(0.4, HelmetState.NO_HELMET)],
        associations=[link(0.0)],
    )
    assert events == ()


def test_single_observation_never_confirms() -> None:
    """The >=2-observation floor is structural (architecture-review §13)."""

    assert reasoner().run([obs(0.0, HelmetState.NO_HELMET)], associations=[link(0.0)]) == ()


def test_one_episode_confirms_only_once() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)],
        associations=[link(0.0)],
    )
    assert len(events) == 1


# --- helmet recovery ---------------------------------------------------------
def test_helmet_ends_the_run_before_confirmation() -> None:
    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(0.5, HelmetState.NO_HELMET),
            obs(0.7, HelmetState.HELMET),  # recovery before min_persistence
            obs(1.2, HelmetState.NO_HELMET),
        ],
        associations=[link(0.0)],
    )
    assert events == ()


def test_helmet_after_confirmation_does_not_retract_the_event() -> None:
    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(1.5, HelmetState.NO_HELMET),  # confirms here
            obs(2.0, HelmetState.HELMET),
        ],
        associations=[link(0.0)],
    )
    assert len(events) == 1


def test_a_helmeted_rider_never_confirms() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    assert events == ()


# --- intermittent uncertainty ------------------------------------------------
def test_uncertainty_does_not_break_an_episode() -> None:
    """Classifier instability/occlusion must not destroy an otherwise solid run."""

    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(0.5, HelmetState.UNCERTAIN, confidence=None),
            obs(1.0, HelmetState.UNCERTAIN, confidence=None),
            obs(1.5, HelmetState.NO_HELMET),
        ],
        associations=[link(0.0)],
    )
    assert len(events) == 1


def test_uncertainty_alone_never_confirms() -> None:
    """Abstention must never fabricate support."""

    events = reasoner().run(
        [obs(s, HelmetState.UNCERTAIN, confidence=None) for s in (0.0, 0.5, 1.0, 1.5, 2.0)],
        associations=[link(0.0)],
    )
    assert events == ()


def test_uncertainty_cannot_make_up_the_two_observation_floor() -> None:
    """One real observation bridged by uncertainty is still one observation."""

    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(0.5, HelmetState.UNCERTAIN, confidence=None),
            obs(1.5, HelmetState.UNCERTAIN, confidence=None),
        ],
        associations=[link(0.0)],
    )
    assert events == ()


def test_gap_wider_than_max_observation_gap_breaks_the_episode() -> None:
    """Uncertainty is tolerated, but not indefinitely."""

    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(10.0, HelmetState.NO_HELMET),  # 10s gap > 2.0s tolerance
        ],
        associations=[link(0.0), link(10.0)],
    )
    assert events == ()


def test_unbounded_gap_when_tolerance_is_unset() -> None:
    """With no configured tolerance the base's timestamp bridging applies."""

    params = NoHelmetParameters(
        min_persistence_seconds=1.0,
        max_observation_gap_seconds=None,
        min_confirming_observations=None,
        persistence_status=ParameterStatus.PROVISIONAL,
        max_observation_gap_status=ParameterStatus.UNSET,
        min_confirming_observations_status=ParameterStatus.UNSET,
    )
    events = reasoner(params).run(
        [obs(0.0, HelmetState.NO_HELMET), obs(10.0, HelmetState.NO_HELMET)],
        associations=[link(0.0)],
    )
    assert len(events) == 1


# --- turban exemption --------------------------------------------------------
def test_turban_rider_never_confirms() -> None:
    """A rider predominantly seen in a turban is exempt, even with a stray
    ``no_helmet`` frame among the turban observations."""

    events = reasoner().run(
        [
            obs(0.0, HelmetState.TURBAN),
            obs(0.5, HelmetState.NO_HELMET),  # a single stray bare-headed frame
            obs(1.0, HelmetState.TURBAN),
            obs(1.5, HelmetState.TURBAN),
            obs(2.0, HelmetState.TURBAN),
        ],
        associations=[link(0.0)],
    )
    assert events == ()


def test_predominant_turban_exemption_latches_across_the_whole_track() -> None:
    """Once turban is the predominant reading, a later bare-headed stretch (e.g.
    classifier noise) must not confirm -- the exemption latches over the clip."""

    events = reasoner().run(
        [obs(s, HelmetState.TURBAN) for s in (0.0, 0.5, 1.0, 1.5, 2.0)]
        + [obs(s, HelmetState.NO_HELMET) for s in (5.0, 5.5, 6.0, 6.5)],
        associations=[link(0.0)],
    )
    assert events == ()


def test_sparse_turban_noise_does_not_exempt_a_sustained_violation() -> None:
    """Regression (H8, real New Delhi footage): a rider observed bare-headed on a
    strict majority of frames must still confirm, even when the untuned zero-shot
    backend misreads the bare head as a ``turban`` on a minority of frames. Under
    the old single-frame latch these two stray turbans silently exempted a rider
    seen ``no_helmet`` for the whole clip, so the violation was never emitted."""

    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
        + [obs(s, HelmetState.TURBAN) for s in (3.5, 4.0)],
        associations=[link(0.0)],
    )
    assert len(events) >= 1
    assert events[0].violation_type is ViolationType.NO_HELMET


def test_exempt_riders_are_recorded_not_silent() -> None:
    r = reasoner()
    r.run([obs(0.0, HelmetState.TURBAN)], associations=[link(0.0)])

    assert r.exempt_track_ids == frozenset({"rider-1"})


def test_exemption_is_per_rider_and_does_not_leak() -> None:
    events = reasoner().run(
        [obs(0.0, HelmetState.TURBAN, track_id="rider-1")]
        + [obs(s, HelmetState.NO_HELMET, track_id="rider-2") for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0), link(0.0, rider="rider-2")],
    )

    assert len(events) == 1
    assert "rider-2" in events[0].track_ids


# --- taint -------------------------------------------------------------------
def test_taint_restart_prevents_bridging_an_id_switch() -> None:
    observations = [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)]
    events = reasoner().run(
        observations,
        associations=[link(0.0)],
        taint_restart_ids={observations[2].observation_id},  # restart at t=1.0
    )
    # Support before the restart is discarded; 1.0 -> 1.5 is only 0.5s.
    assert events == ()


def test_a_clean_segment_after_taint_confirms_on_its_own() -> None:
    observations = [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5, 2.5)]
    events = reasoner().run(
        observations,
        associations=[link(s) for s in (0.0, 0.5, 1.0, 1.5, 2.5)],
        taint_restart_ids={observations[1].observation_id},  # restart at t=0.5
    )
    assert len(events) == 1
    assert events[0].start_at == BASE + timedelta(seconds=0.5)


def test_taint_restart_on_an_uncertain_observation_still_breaks_the_run() -> None:
    """The discontinuity must survive even when the resuming frame is unreadable."""

    observations = [
        obs(0.0, HelmetState.NO_HELMET),
        obs(0.5, HelmetState.NO_HELMET),
        obs(0.9, HelmetState.UNCERTAIN, confidence=None),
        obs(1.2, HelmetState.NO_HELMET),
    ]
    events = reasoner().run(
        observations,
        associations=[link(0.0)],
        taint_restart_ids={observations[2].observation_id},
    )
    assert events == ()


# --- association / attribution ----------------------------------------------
def test_event_attributes_to_the_associated_motorcycle() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(s) for s in (0.0, 0.5, 1.0, 1.5)],
    )

    assert set(events[0].track_ids) == {"rider-1", "bike-9"}


def test_modal_motorcycle_wins_when_a_rider_links_to_several() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[
            link(0.0, bike="bike-a"),
            link(0.5, bike="bike-b"),
            link(1.0, bike="bike-b"),
            link(1.5, bike="bike-b"),
        ],
    )

    assert "bike-b" in events[0].track_ids
    assert "bike-a" not in events[0].track_ids


def test_missing_association_never_invents_a_vehicle() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)], associations=[]
    )

    assert len(events) == 1
    assert events[0].track_ids == ("rider-1",)
    assert events[0].confidence.association is None


def test_association_confidence_is_the_weakest_link_in_the_window() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[
            link(0.0, confidence=0.9),
            link(0.5, confidence=0.4),
            link(1.0, confidence=0.95),
            link(1.5, confidence=0.9),
        ],
    )

    assert events[0].confidence.association == pytest.approx(0.4)


def test_track_ids_are_sorted_so_the_event_id_is_stable() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    assert list(events[0].track_ids) == sorted(events[0].track_ids)


# --- confidence model --------------------------------------------------------
def test_confidence_is_not_merely_the_classifier_score() -> None:
    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET, confidence=0.9),
            obs(0.5, HelmetState.UNCERTAIN, confidence=None),
            obs(1.5, HelmetState.NO_HELMET, confidence=0.7),
        ],
        associations=[link(0.0)],
    )
    breakdown = events[0].confidence

    assert breakdown.classifier == pytest.approx(0.8)  # mean of 0.9, 0.7
    assert breakdown.temporal_consistency == pytest.approx(2 / 3)  # 2 supporting of 3
    assert breakdown.classifier != events[0].confidence.temporal_consistency


def test_temporal_consistency_degrades_with_instability() -> None:
    stable = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    unstable = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET),
            obs(0.3, HelmetState.UNCERTAIN, confidence=None),
            obs(0.6, HelmetState.UNCERTAIN, confidence=None),
            obs(0.9, HelmetState.UNCERTAIN, confidence=None),
            obs(1.5, HelmetState.NO_HELMET),
        ],
        associations=[link(0.0)],
    )

    assert stable[0].confidence.temporal_consistency == pytest.approx(1.0)
    assert unstable[0].confidence.temporal_consistency == pytest.approx(0.4)


def test_uncalibrated_components_are_never_collapsed_into_an_aggregate() -> None:
    """§13: no aggregate that could be read as a calibrated probability of guilt."""

    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    assert events[0].confidence.aggregate is None


def test_unmeasured_components_are_none_not_fabricated() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    breakdown = events[0].confidence

    assert breakdown.detector is None  # never travels on the observation stream
    assert breakdown.track_continuity is None  # taint-free by construction, not measured
    assert breakdown.calibration_quality is None


def test_gated_observations_contribute_no_fabricated_score() -> None:
    """A confirming observation whose score was never measured contributes nothing."""

    events = reasoner().run(
        [
            obs(0.0, HelmetState.NO_HELMET, confidence=0.8),
            obs(1.5, HelmetState.NO_HELMET, confidence=None),
        ],
        associations=[link(0.0)],
    )
    assert events[0].confidence.classifier == pytest.approx(0.8)  # not (0.8+0)/2


# --- measurements + thresholds ----------------------------------------------
def test_measurements_record_the_evidence_behind_the_confirmation() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    names = {m.name: m.value for m in events[0].measurements}

    assert names["persistence_seconds"] == pytest.approx(1.0)
    assert names["confirming_observations"] == 3.0  # 0.0, 0.5, 1.0 within the window
    assert names["min_crop_height_px"] == pytest.approx(40.0)


def test_thresholds_record_the_configured_policy() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    names = {t.name for t in events[0].thresholds}

    assert {"min_persistence", "max_observation_gap", "min_confirming_observations"} <= names


# --- determinism / replay ----------------------------------------------------
def test_replay_is_byte_identical() -> None:
    def run() -> list[str]:
        return [
            e.model_dump_json()
            for e in reasoner().run(
                [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
                associations=[link(s) for s in (0.0, 0.5, 1.0, 1.5)],
            )
        ]

    assert run() == run()


def test_outcome_is_independent_of_input_order() -> None:
    observations = [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)]
    links = [link(s) for s in (0.0, 0.5, 1.0, 1.5)]

    forward = reasoner().run(observations, associations=links)
    backward = reasoner().run(list(reversed(observations)), associations=list(reversed(links)))

    assert [e.model_dump_json() for e in forward] == [e.model_dump_json() for e in backward]


def test_no_wall_clock_in_the_event() -> None:
    events = reasoner().run(
        [obs(s, HelmetState.NO_HELMET) for s in (0.0, 0.5, 1.0, 1.5)],
        associations=[link(0.0)],
    )
    assert events[0].created_at == events[0].trigger_at


# --- parameter loading -------------------------------------------------------
def _scene() -> SceneConfig:
    return SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))


def test_parameters_load_from_the_example_scene() -> None:
    params = no_helmet_parameters(_scene())

    assert params.min_persistence_seconds == 1.0
    assert params.max_observation_gap_seconds == 2.0
    assert params.persistence_status is ParameterStatus.PROVISIONAL


def test_missing_block_fails_fast() -> None:
    scene = _scene()
    stripped = scene.model_copy(
        update={
            "rule_parameters": tuple(
                b for b in scene.rule_parameters if b.violation_type is not ViolationType.NO_HELMET
            )
        }
    )
    with pytest.raises(ValueError, match="no no_helmet rule-parameter block"):
        no_helmet_parameters(stripped)


def test_min_confirming_observations_above_the_structural_floor_fails_fast() -> None:
    """A configured value we cannot honour must never be silently ignored."""

    scene = _scene()
    block = next(
        b for b in scene.rule_parameters if b.violation_type is ViolationType.NO_HELMET
    )
    bumped = block.model_copy(
        update={
            "parameters": tuple(
                p.model_copy(update={"value": 5.0})
                if p.id == "min_confirming_observations"
                else p
                for p in block.parameters
            )
        }
    )
    scene = scene.model_copy(
        update={
            "rule_parameters": tuple(
                bumped if b.violation_type is ViolationType.NO_HELMET else b
                for b in scene.rule_parameters
            )
        }
    )

    with pytest.raises(ValueError, match="exceeds"):
        no_helmet_parameters(scene)

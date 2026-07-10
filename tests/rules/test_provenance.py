"""Reasoner-level model-provenance stamping (P2-U1).

Proves the ``WrongWayReasoner`` carries the run-level ``models`` tuple onto every
minted ``ConfirmedEvent`` as inert provenance -- and that provenance never touches
the *decision*: the events that confirm, their ids, and their timing are
byte-identical with or without ``models``. The reasoner stamps what the composition
boundary supplies verbatim (de-duplication/ordering is the pipeline's job, tested
in ``tests/pipeline/test_provenance_propagation.py``); this file only exercises the
reasoner and the frozen contracts, with synthetic observations -- no detector,
tracker, video, or pipeline.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from trafficpulse.contracts import (
    ConfirmedEvent,
    HeadingVsLaneObservation,
    ModelRef,
    Producer,
    SceneConfig,
    scene_config_hash,
)
from trafficpulse.contracts.enums import ProducerKind
from trafficpulse.contracts.observations import (
    InZoneObservation,
    ObservationBase,
    StationaryObservation,
)
from trafficpulse.rules.engine import RuleEngine
from trafficpulse.rules.wrong_way import WrongWayReasoner, wrong_way_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"
SCENE = SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))
PARAMS = wrong_way_parameters(SCENE)
SCH = scene_config_hash(SCENE)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PRODUCER = Producer(name="synthetic", version="0", kind=ProducerKind.HEURISTIC)

# Truthful-shaped refs (name + version; weights_hash None, nothing hashed).
DET_REF = ModelRef(name="rtdetr-r50vd", version="provisional")
TRK_REF = ModelRef(name="iou-tracker", version="0.1.0-provisional")


def _cobs(i: int, *, track: str = "tk", spacing: float = 0.1) -> HeadingVsLaneObservation:
    """One wrong-way (contradiction) observation at ``i * spacing`` seconds."""

    return HeadingVsLaneObservation(
        observation_id=f"o-{track}-{i:03d}",
        camera_id="cam-1",
        track_id=track,
        timestamp=BASE + timedelta(seconds=i * spacing),
        producer=PRODUCER,
        lane_id="lane",
        heading_degrees=90.0,
        legal_heading_degrees=270.0,
        deviation_degrees=180.0,
        is_contradiction=True,
    )


def _confirm(models: tuple[ModelRef, ...] = ()) -> tuple[ConfirmedEvent, ...]:
    """Run a sustained wrong-way run (spans 1.4 s > 1.0 s min_persistence)."""

    reasoner = WrongWayReasoner(RuleEngine(), PARAMS, scene_config_hash=SCH, models=models)
    return reasoner.run([_cobs(i) for i in range(15)])


# --- stamping ----------------------------------------------------------------
def test_reasoner_stamps_supplied_models_on_event() -> None:
    (event,) = _confirm(models=(TRK_REF, DET_REF))
    assert event.models == (TRK_REF, DET_REF)  # stamped verbatim, in the given order


def test_default_models_is_empty_tuple() -> None:
    (event,) = _confirm()
    assert event.models == ()


def test_stub_run_supplying_no_refs_yields_empty_models() -> None:
    # Explicitly the honest stub posture: nothing supplied -> nothing invented.
    (event,) = _confirm(models=())
    assert event.models == ()


# --- decision / identity invariance under provenance -------------------------
def test_models_do_not_change_which_events_confirm() -> None:
    without = _confirm(models=())
    withrefs = _confirm(models=(DET_REF, TRK_REF))
    assert len(without) == len(withrefs) == 1


def test_models_do_not_change_event_id() -> None:
    (bare,) = _confirm(models=())
    (stamped,) = _confirm(models=(DET_REF, TRK_REF))
    assert bare.event_id == stamped.event_id  # provenance is absent from _event_id


def test_only_the_models_field_differs_with_and_without_provenance() -> None:
    # The whole event is identical once models is normalised away -- so timing,
    # measurements, thresholds, and identity are all provenance-independent.
    (bare,) = _confirm(models=())
    (stamped,) = _confirm(models=(DET_REF, TRK_REF))
    assert bare == stamped.model_copy(update={"models": ()})


def test_event_id_independent_of_model_ordering() -> None:
    (a,) = _confirm(models=(DET_REF, TRK_REF))
    (b,) = _confirm(models=(TRK_REF, DET_REF))
    assert a.event_id == b.event_id


# --- no provenance leakage into the model-free observation layer -------------
def test_observation_contracts_carry_no_model_ref_fields() -> None:
    # Observations carry a Producer, never a ModelRef: the reasoning log stays
    # decoupled from model identity (architecture-review §15; plan §6.3).
    for contract in (ObservationBase, HeadingVsLaneObservation, InZoneObservation,
                     StationaryObservation):
        fields = contract.model_fields
        assert "models" not in fields
        assert "source_model" not in fields
        assert "tracker" not in fields
        assert "producer" in ObservationBase.model_fields


def test_weights_hash_is_none_on_stamped_refs() -> None:
    (event,) = _confirm(models=(DET_REF, TRK_REF))
    assert all(ref.weights_hash is None for ref in event.models)

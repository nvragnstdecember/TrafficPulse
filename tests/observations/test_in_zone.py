"""Tests for in-zone observation derivation (P2-U2).

Deterministic, model-free tests over the frozen ``InZoneObservation`` contract:
inside/outside/on-edge/on-vertex membership (bottom-center vs no-stopping
polygons), multi-zone and overlapping emission, eligible-zone/disabled filtering,
two-state minimum, tainted-step skip + restart marking, provenance/identity
propagation, timezone-aware timestamp preservation, serialization round-trip,
determinism, input immutability, and boundary/import-boundary audits. Uses
synthetic TrackStates and the real example SceneConfig only.
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from trafficpulse.contracts import (
    BoundingBox,
    InZoneObservation,
    ObjectClass,
    ObservationAdapter,
    Producer,
    ProducerKind,
    SceneConfig,
    TrackState,
    TrackStatus,
    ZoneKind,
)
from trafficpulse.contracts.scene import Zone, ZoneType
from trafficpulse.geometry import point_in_polygon
from trafficpulse.observations import zones as zones_module
from trafficpulse.observations.zones import (
    DEFAULT_IN_ZONE_PRODUCER,
    InZoneDerivation,
    derive_in_zone_observations,
    derive_in_zone_observations_with_taint,
)
from trafficpulse.synth import build_track, linear_positions

# The example scene's no-stopping zone polygon (configs/scenes/example-scene.yaml).
NO_STOP_POLY: tuple[tuple[float, float], ...] = (
    (1260.0, 1060.0),
    (1520.0, 1060.0),
    (1470.0, 660.0),
    (1310.0, 660.0),
)
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_HALF_W = 20.0
_HEIGHT = 40.0

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "example-scene.yaml"


def _no_stop_zone(
    zone_id: str = "zone-no-stop", polygon=NO_STOP_POLY, *, enabled: bool = True
) -> Zone:
    return Zone(
        zone_id=zone_id, zone_type=ZoneType.NO_STOPPING, enabled=enabled, polygon=tuple(polygon)
    )


def _track_at(
    bottom_center: tuple[float, float],
    *,
    n: int = 3,
    camera_id: str = "cam-x",
    track_id: str = "trk-x",
    tainted_indices: tuple[int, ...] = (),
) -> list[TrackState]:
    """A track of ``n`` identical states whose bbox bottom-center == the target."""

    x, y = bottom_center
    states: list[TrackState] = []
    for i in range(n):
        bbox = BoundingBox(x1=x - _HALF_W, y1=y - _HEIGHT, x2=x + _HALF_W, y2=y)
        states.append(
            TrackState(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=_BASE + timedelta(seconds=i),
                frame_index=i,
                object_class=ObjectClass.CAR,
                bbox=bbox,
                status=TrackStatus.ACTIVE,
                tainted=(i in tainted_indices),
            )
        )
    return states


def _derive(track, zones):  # type: ignore[no-untyped-def]
    return derive_in_zone_observations(track, zones=zones)


# --- membership vs bottom-center --------------------------------------------
def test_point_clearly_inside() -> None:
    obs = _derive(_track_at((1390.0, 880.0)), [_no_stop_zone()])
    assert obs
    assert all(o.is_inside for o in obs)


def test_point_clearly_outside() -> None:
    obs = _derive(_track_at((1000.0, 880.0)), [_no_stop_zone()])
    assert obs
    assert all(not o.is_inside for o in obs)  # negative membership is still emitted


def test_point_on_edge_is_inside() -> None:
    # Bottom edge of the polygon runs along y = 1060 from x=1260 to x=1520.
    obs = _derive(_track_at((1390.0, 1060.0)), [_no_stop_zone()])
    assert all(o.is_inside for o in obs)


def test_point_on_vertex_is_inside() -> None:
    obs = _derive(_track_at((1260.0, 1060.0)), [_no_stop_zone()])
    assert all(o.is_inside for o in obs)


def test_membership_matches_geometry_primitive() -> None:
    # The derivation must not diverge from the shared point_in_polygon result.
    for target in ((1390.0, 880.0), (1000.0, 880.0), (1390.0, 1060.0), (1260.0, 1060.0)):
        obs = _derive(_track_at(target), [_no_stop_zone()])
        expected = point_in_polygon(target, NO_STOP_POLY)
        assert all(o.is_inside is expected for o in obs)


def test_bottom_center_not_bbox_center() -> None:
    # A box whose *center* is outside but whose *bottom-center* is inside must
    # read as inside -- proving the ground-contact reference point is used.
    # Center y = 640 (above the polygon top y=660); bottom-center y = 700 (inside).
    x = 1390.0
    bbox = BoundingBox(x1=x - _HALF_W, y1=640.0 - _HEIGHT, x2=x + _HALF_W, y2=700.0)
    state = TrackState(
        track_id="t", camera_id="c", timestamp=_BASE, frame_index=0,
        object_class=ObjectClass.CAR, bbox=bbox, status=TrackStatus.ACTIVE,
    )
    later = state.model_copy(update={"timestamp": _BASE + timedelta(seconds=1), "frame_index": 1})
    track = [state, later]
    center_y = (bbox.y1 + bbox.y2) / 2.0
    assert not point_in_polygon((x, center_y), NO_STOP_POLY)  # center is outside
    obs = _derive(track, [_no_stop_zone()])
    assert obs and all(o.is_inside for o in obs)  # bottom-center is inside


# --- multi-zone / overlap ----------------------------------------------------
def test_multiple_zones_emit_one_each_in_order() -> None:
    zone_a = _no_stop_zone("zone-a")
    zone_b = _no_stop_zone("zone-b")
    obs = _derive(_track_at((1390.0, 880.0), n=3), [zone_a, zone_b])
    # 3 states -> 2 steps; per step one observation per zone, in input zone order.
    assert [o.zone_id for o in obs] == ["zone-a", "zone-b", "zone-a", "zone-b"]
    assert all(o.is_inside for o in obs)


def test_overlapping_zones_both_inside() -> None:
    poly_left = ((1200.0, 1000.0), (1400.0, 1000.0), (1400.0, 700.0), (1200.0, 700.0))
    poly_right = ((1300.0, 1000.0), (1500.0, 1000.0), (1500.0, 700.0), (1300.0, 700.0))
    zones = [_no_stop_zone("left", poly_left), _no_stop_zone("right", poly_right)]
    obs = _derive(_track_at((1350.0, 850.0), n=2), zones)  # point in the overlap
    assert {o.zone_id for o in obs} == {"left", "right"}
    assert all(o.is_inside for o in obs)


# --- eligibility filtering ---------------------------------------------------
def test_non_no_stopping_zones_excluded() -> None:
    lane = Zone(zone_id="lane", zone_type=ZoneType.LANE, enabled=True, polygon=NO_STOP_POLY)
    no_stop = _no_stop_zone("keep")
    obs = _derive(_track_at((1390.0, 880.0)), [lane, no_stop])
    assert {o.zone_id for o in obs} == {"keep"}
    assert all(o.zone_kind is ZoneKind.NO_STOPPING for o in obs)


def test_disabled_no_stopping_zone_excluded() -> None:
    obs = _derive(_track_at((1390.0, 880.0)), [_no_stop_zone("off", enabled=False)])
    assert obs == []


def test_empty_zone_set_yields_nothing() -> None:
    assert _derive(_track_at((1390.0, 880.0)), []) == []


def test_no_eligible_zones_yields_nothing() -> None:
    lane = Zone(zone_id="lane", zone_type=ZoneType.LANE, enabled=True, polygon=NO_STOP_POLY)
    assert _derive(_track_at((1390.0, 880.0)), [lane]) == []


# --- two-state minimum / empty input -----------------------------------------
def test_single_state_yields_nothing() -> None:
    assert _derive(_track_at((1390.0, 880.0), n=1), [_no_stop_zone()]) == []


def test_empty_track_yields_nothing() -> None:
    assert _derive([], [_no_stop_zone()]) == []


def test_observation_emitted_for_current_state_only() -> None:
    # N states -> N-1 observations (one per zone); the first state never emits.
    track = _track_at((1390.0, 880.0), n=4)
    obs = _derive(track, [_no_stop_zone()])
    assert [o.timestamp for o in obs] == [s.timestamp for s in track[1:]]


# --- taint handling ----------------------------------------------------------
def test_tainted_steps_skipped_and_restart_marked() -> None:
    # 5 states; state index 2 is tainted -> steps (1,2) and (2,3) drop; the next
    # clean step (3,4) resumes and is flagged a taint restart.
    track = _track_at((1390.0, 880.0), n=5, tainted_indices=(2,))
    result = derive_in_zone_observations_with_taint(track, zones=[_no_stop_zone()])
    emitted_ts = [o.timestamp for o in result.observations]
    assert emitted_ts == [track[1].timestamp, track[4].timestamp]  # step (0,1) and (3,4)
    restart = [o for o in result.observations if o.observation_id in result.taint_restart_ids]
    assert len(restart) == 1
    assert restart[0].timestamp == track[4].timestamp


def test_all_tainted_yields_nothing() -> None:
    track = _track_at((1390.0, 880.0), n=4, tainted_indices=(0, 1, 2, 3))
    result = derive_in_zone_observations_with_taint(track, zones=[_no_stop_zone()])
    assert result.observations == ()
    assert result.taint_restart_ids == frozenset()


def test_restart_flags_all_zone_observations_of_the_step() -> None:
    track = _track_at((1390.0, 880.0), n=3, tainted_indices=(0,))  # step (0,1) tainted
    result = derive_in_zone_observations_with_taint(
        track, zones=[_no_stop_zone("a"), _no_stop_zone("b")]
    )
    # Only step (1,2) survives; both its zone observations are restarts.
    assert len(result.observations) == 2
    assert {o.observation_id for o in result.observations} == set(result.taint_restart_ids)


def test_clean_track_has_no_restarts() -> None:
    result = derive_in_zone_observations_with_taint(
        _track_at((1390.0, 880.0), n=4), zones=[_no_stop_zone()]
    )
    assert result.taint_restart_ids == frozenset()
    assert len(result.observations) == 3


# --- provenance / identity ---------------------------------------------------
def test_zone_metadata_correct() -> None:
    obs = _derive(_track_at((1390.0, 880.0), n=2), [_no_stop_zone("zone-no-stop")])
    assert obs[0].zone_id == "zone-no-stop"
    assert obs[0].zone_kind is ZoneKind.NO_STOPPING
    assert obs[0].obs_type == "in_zone"


def test_track_and_camera_identity_preserved() -> None:
    track = _track_at((1390.0, 880.0), n=3, camera_id="cam-42", track_id="trk-7")
    obs = _derive(track, [_no_stop_zone()])
    assert all(o.camera_id == "cam-42" for o in obs)
    assert all(o.track_id == "trk-7" for o in obs)


def test_timestamps_are_timezone_aware_and_from_trackstate() -> None:
    track = _track_at((1390.0, 880.0), n=4)
    obs = _derive(track, [_no_stop_zone()])
    assert [o.timestamp for o in obs] == [s.timestamp for s in track[1:]]
    assert all(o.timestamp.tzinfo is not None for o in obs)


def test_default_producer_is_heuristic() -> None:
    obs = _derive(_track_at((1390.0, 880.0), n=2), [_no_stop_zone()])
    assert obs[0].producer == DEFAULT_IN_ZONE_PRODUCER
    assert obs[0].producer.kind is ProducerKind.HEURISTIC


def test_custom_producer_is_used() -> None:
    prod = Producer(name="custom", version="9.9", kind=ProducerKind.HEURISTIC)
    obs = derive_in_zone_observations(
        _track_at((1390.0, 880.0), n=2), zones=[_no_stop_zone()], producer=prod
    )
    assert all(o.producer == prod for o in obs)


def test_observation_ids_deterministic_and_unique() -> None:
    track = _track_at((1390.0, 880.0), n=5)
    ids_a = [o.observation_id for o in _derive(track, [_no_stop_zone()])]
    ids_b = [o.observation_id for o in _derive(track, [_no_stop_zone()])]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)
    assert all(i.startswith("inz-") for i in ids_a)


# --- serialization -----------------------------------------------------------
def test_serialization_round_trip() -> None:
    obs = _derive(_track_at((1390.0, 880.0), n=2), [_no_stop_zone()])[0]
    assert ObservationAdapter.validate_python(obs.model_dump()) == obs
    assert ObservationAdapter.validate_json(obs.model_dump_json()) == obs


# --- determinism / immutability ----------------------------------------------
def test_repeated_calls_are_identical() -> None:
    track = _track_at((1390.0, 880.0), n=5)
    zones = [_no_stop_zone("a"), _no_stop_zone("b")]
    a = derive_in_zone_observations_with_taint(track, zones=zones)
    b = derive_in_zone_observations_with_taint(track, zones=zones)
    assert a == b


def test_inputs_not_mutated() -> None:
    track = _track_at((1390.0, 880.0), n=4)
    zones = [_no_stop_zone()]
    track_before = [s.model_dump() for s in track]
    zones_before = [z.model_dump() for z in zones]
    derive_in_zone_observations_with_taint(track, zones=zones)
    assert [s.model_dump() for s in track] == track_before
    assert [z.model_dump() for z in zones] == zones_before


def test_result_is_frozen_dataclass() -> None:
    result = derive_in_zone_observations_with_taint(
        _track_at((1390.0, 880.0), n=2), zones=[_no_stop_zone()]
    )
    assert isinstance(result, InZoneDerivation)
    with pytest.raises(FrozenInstanceError):
        result.observations = ()  # type: ignore[misc]


# --- no temporal / provenance-metadata coupling ------------------------------
def test_observation_has_no_temporal_or_dwell_fields() -> None:
    # P2-U2 emits pure per-step membership; dwell/stationarity/speed live
    # elsewhere and must not leak into the in-zone contract usage.
    fields = set(InZoneObservation.model_fields)
    assert "dwell_seconds" not in fields
    assert "is_stationary" not in fields
    assert "speed_estimate" not in fields


def test_repeated_inside_frames_are_independent_facts() -> None:
    # No accumulation: every in-zone step is an independent boolean, not a
    # growing dwell count.
    obs = _derive(_track_at((1390.0, 880.0), n=6), [_no_stop_zone()])
    assert len(obs) == 5
    assert all(o.is_inside for o in obs)


def test_no_dependency_on_model_provenance() -> None:
    # The derivation neither reads nor emits ModelRef provenance (P2-U1 is a
    # run-level event concern); the observation carries only a Producer.
    obs = _derive(_track_at((1390.0, 880.0), n=2), [_no_stop_zone()])[0]
    assert not hasattr(obs, "models")
    assert isinstance(obs.producer, Producer)


# --- synthetic + real-scene integration --------------------------------------
def test_synthetic_track_entering_zone_transitions_false_to_true() -> None:
    # A track sweeping right across the polygon boundary at y=880: bottom-center
    # is inside once x >= ~1282.5. Centers -> bottom-centers (px, 880).
    positions = linear_positions((1200.0, 860.0), (1.0, 0.0), 50.0, 5)
    track = build_track(positions, track_id="sweep")
    obs = _derive(track, [_no_stop_zone()])
    # Steps emit for states x = 1250, 1300, 1350, 1400.
    assert [o.is_inside for o in obs] == [False, True, True, True]


def test_real_example_scene_zones_integration() -> None:
    scene = SceneConfig.model_validate(yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8")))
    track = _track_at((1390.0, 880.0), n=3)  # inside the example zone-no-stop
    obs = _derive(track, scene.zones)  # all 6 zones; only zone-no-stop is eligible
    assert {o.zone_id for o in obs} == {"zone-no-stop"}
    assert all(o.zone_kind is ZoneKind.NO_STOPPING and o.is_inside for o in obs)


# --- source/import boundary audit --------------------------------------------
def test_module_imports_no_backend_or_reasoning() -> None:
    source = Path(zones_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "torch", "transformers", "cv2", "import av", "numpy", "sqlite",
        "requests", "urllib", "socket",
        "..rules", "..persistence", "..detector", "..tracking", "..pipeline",
    )
    for token in forbidden:
        assert token not in source, f"unexpected import token in zones.py: {token!r}"

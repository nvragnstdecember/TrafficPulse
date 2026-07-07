"""Discriminated-union tests for all seven observation variants."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trafficpulse.contracts import (
    OBSERVATION_VARIANTS,
    HeadingVsLaneObservation,
    HelmetState,
    HelmetStateObservation,
    InZoneObservation,
    ObservationAdapter,
    Producer,
    ProducerKind,
    RiderCountObservation,
    RiderSlot,
    SignalState,
    SignalStateObservation,
    SpeedObservation,
    SpeedUnit,
    StationaryObservation,
    ZoneKind,
)

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
PRODUCER = Producer(name="perception", version="0.1.0", kind=ProducerKind.MODEL)

EXPECTED_OBS_TYPES = {
    "in_zone",
    "signal_state",
    "heading_vs_lane",
    "stationary",
    "rider_count",
    "helmet_state",
    "speed",
}


def _base(**extra: object) -> dict[str, object]:
    data: dict[str, object] = {
        "observation_id": "o1",
        "camera_id": "cam1",
        "timestamp": TS.isoformat(),
        "producer": PRODUCER.model_dump(mode="json"),
    }
    data.update(extra)
    return data


def test_seven_variants_registered() -> None:
    assert len(OBSERVATION_VARIANTS) == 7
    got = {v.model_fields["obs_type"].default for v in OBSERVATION_VARIANTS}
    assert got == EXPECTED_OBS_TYPES


def test_in_zone_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(
            obs_type="in_zone",
            track_id="t1",
            zone_id="z1",
            zone_kind="no_stopping",
            is_inside=True,
        )
    )
    assert isinstance(obs, InZoneObservation)
    assert obs.zone_kind is ZoneKind.NO_STOPPING


def test_signal_state_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(obs_type="signal_state", signal_state="red", roi_id="light1")
    )
    assert isinstance(obs, SignalStateObservation)
    assert obs.signal_state is SignalState.RED
    assert obs.track_id is None


def test_heading_vs_lane_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(
            obs_type="heading_vs_lane",
            track_id="t1",
            lane_id="lane1",
            heading_degrees=180.0,
            deviation_degrees=175.0,
            is_contradiction=True,
        )
    )
    assert isinstance(obs, HeadingVsLaneObservation)
    assert obs.is_contradiction is True


def test_stationary_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(obs_type="stationary", track_id="t1", is_stationary=True, dwell_seconds=42.0)
    )
    assert isinstance(obs, StationaryObservation)
    assert obs.dwell_seconds == 42.0


def test_rider_count_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(obs_type="rider_count", track_id="moto1", rider_count=3)
    )
    assert isinstance(obs, RiderCountObservation)
    assert obs.rider_count == 3


def test_helmet_state_parses() -> None:
    obs = ObservationAdapter.validate_python(
        _base(
            obs_type="helmet_state",
            track_id="rider1",
            helmet_state="turban",
            rider_slot="driver",
            crop_height_px=48.0,
        )
    )
    assert isinstance(obs, HelmetStateObservation)
    assert obs.helmet_state is HelmetState.TURBAN
    assert obs.rider_slot is RiderSlot.DRIVER


def test_speed_parses_with_uncertainty() -> None:
    obs = ObservationAdapter.validate_python(
        _base(
            obs_type="speed",
            track_id="t1",
            speed_value=54.0,
            speed_sigma=2.5,
            unit="km_per_h",
            sample_count=10,
        )
    )
    assert isinstance(obs, SpeedObservation)
    assert obs.unit is SpeedUnit.KM_PER_H
    assert obs.speed_sigma == 2.5


def test_union_roundtrip_preserves_variant() -> None:
    original = SpeedObservation(
        observation_id="o1",
        camera_id="cam1",
        track_id="t1",
        timestamp=TS,
        producer=PRODUCER,
        speed_value=54.0,
        speed_sigma=2.5,
        unit=SpeedUnit.KM_PER_H,
    )
    payload = ObservationAdapter.dump_json(original)
    restored = ObservationAdapter.validate_json(payload)
    assert isinstance(restored, SpeedObservation)
    assert restored == original


def test_unknown_discriminator_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservationAdapter.validate_python(_base(obs_type="not_a_real_type"))


def test_missing_discriminator_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservationAdapter.validate_python(_base(zone_id="z1"))

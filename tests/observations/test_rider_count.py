"""Rider-count observation derivation (v1.1 U3).

Pure geometry over frozen contracts: no pixels, no detector, no ML. Verifies the
count is read from the reused perception layer and stamped as the frozen
``RiderCountObservation``, keyed by the motorcycle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trafficpulse.contracts import RiderCountObservation, TrackState
from trafficpulse.contracts.enums import AssociationType, ObjectClass, ProducerKind, TrackStatus
from trafficpulse.contracts.primitives import BoundingBox
from trafficpulse.observations.rider_count import derive_rider_count_observations

BASE = datetime(1970, 1, 1, tzinfo=UTC)


def state(
    track_id: str,
    object_class: ObjectClass,
    box: tuple[float, float, float, float],
    *,
    tainted: bool = False,
    frame_index: int = 0,
) -> TrackState:
    x1, y1, x2, y2 = box
    return TrackState(
        track_id=track_id,
        camera_id="cam-1",
        timestamp=BASE + timedelta(seconds=frame_index),
        frame_index=frame_index,
        object_class=object_class,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        status=TrackStatus.ACTIVE,
        tainted=tainted,
    )


def bike(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.MOTORCYCLE, box, **kw)  # type: ignore[arg-type]


def person(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.PERSON, box, **kw)  # type: ignore[arg-type]


def test_counts_associated_riders_per_motorcycle() -> None:
    derivation = derive_rider_count_observations(
        [
            bike("m1", (0, 0, 60, 60)),
            person("p1", (5, 0, 30, 40)),
            person("p2", (30, 0, 55, 40)),
        ]
    )
    assert len(derivation.observations) == 1
    obs = derivation.observations[0]
    assert isinstance(obs, RiderCountObservation)
    assert obs.rider_count == 2
    # Keyed by the motorcycle (the vehicle the count is a property of).
    assert obs.track_id == "m1" and obs.motorcycle_track_id == "m1"
    # The rider↔motorcycle links travel alongside for attribution.
    assert len(derivation.associations) == 2
    assert all(
        a.association_type is AssociationType.RIDER_OF_MOTORCYCLE for a in derivation.associations
    )


def test_motorcycle_with_no_riders_reports_zero() -> None:
    derivation = derive_rider_count_observations([bike("m1", (0, 0, 40, 60))])
    assert derivation.observations[0].rider_count == 0
    assert derivation.associations == ()


def test_multiple_motorcycles_counted_independently() -> None:
    derivation = derive_rider_count_observations(
        [
            bike("m1", (0, 0, 60, 60)),
            person("p1", (5, 0, 30, 40)),
            person("p2", (30, 0, 55, 40)),
            bike("m2", (200, 0, 240, 60)),
            person("p3", (205, 0, 230, 40)),
        ]
    )
    counts = {o.motorcycle_track_id: o.rider_count for o in derivation.observations}
    assert counts == {"m1": 2, "m2": 1}


def test_ignores_non_motorcycle_classes_and_tainted() -> None:
    derivation = derive_rider_count_observations(
        [
            state("c1", ObjectClass.CAR, (0, 0, 50, 50)),
            bike("m1", (0, 0, 40, 60), tainted=True),
            person("p1", (5, 0, 30, 40)),
        ]
    )
    assert derivation.observations == ()  # tainted motorcycle abstains; car ignored


def test_ids_are_content_derived_and_provenance_is_provisional() -> None:
    first = derive_rider_count_observations([bike("m1", (0, 0, 40, 60))])
    again = derive_rider_count_observations([bike("m1", (0, 0, 40, 60))])
    assert first.observations[0].observation_id.startswith("rct-")
    assert first.observations[0].observation_id == again.observations[0].observation_id
    assert first.observations[0].producer.kind is ProducerKind.HEURISTIC
    assert "provisional" in first.observations[0].producer.version

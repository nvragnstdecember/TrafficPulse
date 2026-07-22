"""Motorcycle perception foundation (v1.1 U1).

Pure aggregation over frozen ``TrackState`` / ``Association`` contracts: no
pixels, no detector, no tracker, no ML. Verifies the perception observations,
the reuse of the rider↔motorcycle association, rider indexing, and the temporal
track summary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trafficpulse.association import RiderAssociationConfig
from trafficpulse.contracts import TrackState
from trafficpulse.contracts.enums import AssociationType, ObjectClass, ProducerKind, TrackStatus
from trafficpulse.contracts.primitives import BoundingBox
from trafficpulse.perception import (
    MotorcycleObservation,
    MotorcycleTrackObservation,
    RiderObservation,
    derive_perception_frame,
    summarize_motorcycle_track,
    summarize_motorcycle_tracks,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)


def state(
    track_id: str,
    object_class: ObjectClass,
    box: tuple[float, float, float, float],
    *,
    tainted: bool = False,
    frame_index: int = 0,
    confidence: float | None = 0.9,
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
        confidence=confidence,
    )


def bike(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.MOTORCYCLE, box, **kw)  # type: ignore[arg-type]


def person(track_id: str, box: tuple[float, float, float, float], **kw: object) -> TrackState:
    return state(track_id, ObjectClass.PERSON, box, **kw)  # type: ignore[arg-type]


# --- per-frame detection filtering + observation shape -------------------------
def test_motorcycle_observation_exposes_id_bbox_confidence_and_riders() -> None:
    # A rider box sitting on the motorcycle box (IoMin >= default 0.30).
    frame = derive_perception_frame([bike("m1", (0, 0, 40, 60)), person("p1", (10, 0, 30, 40))])

    assert len(frame.motorcycles) == 1
    moto = frame.motorcycles[0]
    assert isinstance(moto, MotorcycleObservation)
    assert moto.motorcycle_track_id == "m1"
    assert moto.bbox.x2 == 40 and moto.confidence == 0.9
    assert moto.rider_track_ids == ("p1",)
    assert moto.rider_count == 1


def test_ignores_non_motorcycle_non_person_classes() -> None:
    frame = derive_perception_frame(
        [state("c1", ObjectClass.CAR, (0, 0, 50, 50)), bike("m1", (0, 0, 40, 60))]
    )
    assert [m.motorcycle_track_id for m in frame.motorcycles] == ["m1"]
    assert frame.riders == ()


def test_motorcycle_with_no_riders_is_still_observed() -> None:
    frame = derive_perception_frame([bike("m1", (0, 0, 40, 60))])
    assert len(frame.motorcycles) == 1
    assert frame.motorcycles[0].rider_track_ids == ()
    assert frame.riders == ()


# --- association reuse ---------------------------------------------------------
def test_reuses_rider_association_and_emits_associations() -> None:
    frame = derive_perception_frame([bike("m1", (0, 0, 40, 60)), person("p1", (10, 0, 30, 40))])
    assert len(frame.associations) == 1
    assoc = frame.associations[0]
    assert assoc.association_type is AssociationType.RIDER_OF_MOTORCYCLE
    assert assoc.subject_track_id == "p1" and assoc.object_track_id == "m1"

    rider = frame.riders[0]
    assert isinstance(rider, RiderObservation)
    assert rider.rider_track_id == "p1" and rider.motorcycle_track_id == "m1"
    # Association confidence is the geometric overlap carried from the reused layer.
    assert rider.association_confidence == pytest.approx(assoc.confidence)


def test_association_config_is_forwarded() -> None:
    # A marginal overlap that passes at the default but fails a stricter threshold.
    states = [bike("m1", (0, 0, 40, 60)), person("p1", (30, 0, 60, 40))]
    assert derive_perception_frame(states).riders  # default threshold links them
    strict = RiderAssociationConfig(min_overlap=0.99)
    assert derive_perception_frame(states, association_config=strict).riders == ()


# --- multiple riders + rider indexing (triple-riding foundation) ---------------
def test_multiple_riders_share_a_motorcycle_with_stable_indices() -> None:
    frame = derive_perception_frame(
        [
            bike("m1", (0, 0, 60, 60)),
            person("p2", (30, 0, 55, 40)),
            person("p1", (5, 0, 30, 40)),
        ]
    )
    moto = frame.motorcycles[0]
    assert moto.rider_count == 2
    # Riders are ordered deterministically by track id, with a stable ordinal.
    assert moto.rider_track_ids == ("p1", "p2")
    assert [(r.rider_track_id, r.rider_index) for r in frame.riders] == [("p1", 0), ("p2", 1)]


# --- taint abstention ----------------------------------------------------------
def test_tainted_motorcycle_abstains() -> None:
    frame = derive_perception_frame(
        [bike("m1", (0, 0, 40, 60), tainted=True), person("p1", (10, 0, 30, 40))]
    )
    assert frame.motorcycles == ()
    assert frame.riders == ()


# --- determinism ---------------------------------------------------------------
def test_output_is_order_independent_and_deterministic() -> None:
    a = [bike("m2", (100, 0, 140, 60)), bike("m1", (0, 0, 40, 60)), person("p1", (10, 0, 30, 40))]
    b = list(reversed(a))
    frame_a, frame_b = derive_perception_frame(a), derive_perception_frame(b)
    assert [m.motorcycle_track_id for m in frame_a.motorcycles] == ["m1", "m2"]
    assert frame_a.motorcycles == frame_b.motorcycles
    assert frame_a.riders == frame_b.riders


def test_observation_ids_are_content_derived() -> None:
    frame = derive_perception_frame([bike("m1", (0, 0, 40, 60))])
    assert frame.motorcycles[0].observation_id.startswith("mot-")
    # Same inputs -> identical ids (no wall-clock, no counter).
    again = derive_perception_frame([bike("m1", (0, 0, 40, 60))])
    assert frame.motorcycles[0].observation_id == again.motorcycles[0].observation_id


# --- provenance ----------------------------------------------------------------
def test_default_producer_is_a_provisional_heuristic() -> None:
    moto = derive_perception_frame([bike("m1", (0, 0, 40, 60))]).motorcycles[0]
    assert moto.producer.kind is ProducerKind.HEURISTIC
    assert "provisional" in moto.producer.version


# --- temporal track summary (stable-track foundation) --------------------------
def _run_frames() -> list:
    # A motorcycle "m1" seen over three frames; a second rider appears in frame 2.
    frames = []
    frames.append(
        derive_perception_frame(
            [bike("m1", (0, 0, 60, 60), frame_index=0), person("p1", (5, 0, 30, 40), frame_index=0)]
        )
    )
    frames.append(
        derive_perception_frame(
            [
                bike("m1", (0, 0, 60, 60), frame_index=1),
                person("p1", (5, 0, 30, 40), frame_index=1),
                person("p2", (30, 0, 55, 40), frame_index=1),
            ]
        )
    )
    frames.append(derive_perception_frame([bike("m1", (0, 0, 60, 60), frame_index=2)]))
    return frames


def test_summarize_motorcycle_track_folds_frames() -> None:
    frames = _run_frames()
    summary = summarize_motorcycle_track([f.motorcycles[0] for f in frames])
    assert isinstance(summary, MotorcycleTrackObservation)
    assert summary.motorcycle_track_id == "m1"
    assert summary.frame_count == 3
    assert summary.first_frame_index == 0 and summary.last_frame_index == 2
    assert summary.max_rider_count == 2  # peak in frame 2
    assert summary.associated_rider_track_ids == ("p1", "p2")  # union across the track
    assert summary.observation_id.startswith("mtk-")


def test_summarize_motorcycle_tracks_groups_by_track() -> None:
    frames = _run_frames()
    # Add a second, single-frame motorcycle in the last frame.
    frames.append(derive_perception_frame([bike("m9", (200, 0, 240, 60), frame_index=2)]))
    summaries = summarize_motorcycle_tracks(frames)
    assert [s.motorcycle_track_id for s in summaries] == ["m1", "m9"]
    assert summaries[0].frame_count == 3 and summaries[1].frame_count == 1


def test_summarize_rejects_empty_or_mixed_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_motorcycle_track([])
    frames = _run_frames()
    mixed = [frames[0].motorcycles[0], summarize_helper_other_track()]
    with pytest.raises(ValueError, match="one motorcycle_track_id"):
        summarize_motorcycle_track(mixed)


def summarize_helper_other_track() -> MotorcycleObservation:
    return derive_perception_frame([bike("m2", (0, 0, 40, 60))]).motorcycles[0]

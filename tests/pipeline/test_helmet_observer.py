"""The helmet ``FrameObserver``: association -> crop -> classify -> observe (P4-U4).

Exercises the wiring through the P4-U2 seam with a scripted stub classifier: no
model, no weights, no ML. Also asserts the unit's headline invariant -- that
observing produces observations and **nothing else**: no event, no persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from trafficpulse.classifier import (
    Crop,
    RawHelmetPrediction,
    StubHelmetClassifier,
)
from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import (
    AssociationType,
    HelmetState,
    ObjectClass,
    RiderSlot,
    TrackStatus,
)
from trafficpulse.detector.frame import Frame
from trafficpulse.observations.helmet import HeadCropConfig, HelmetObservationConfig
from trafficpulse.pipeline.helmet_observer import HelmetFrameObserver

BASE = datetime(1970, 1, 1, tzinfo=UTC)
NO_HELMET = RawHelmetPrediction("no_helmet", 0.85)
HELMET = RawHelmetPrediction("helmet", 0.9)


def image(height: int = 400, width: int = 400) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = (((ys // 2) + (xs // 2)) % 2 * 255).astype(np.uint8)
    return np.stack([pattern] * 3, axis=-1)


def frame(frame_index: int = 0, *, with_pixels: bool = True) -> Frame:
    return Frame(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=BASE + timedelta(seconds=frame_index),
        image=image() if with_pixels else None,
    )


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


def rider_on_bike(frame_index: int = 0, *, tainted: bool = False) -> list[TrackState]:
    return [
        state("m1", ObjectClass.MOTORCYCLE, (50, 150, 150, 300), frame_index=frame_index),
        state(
            "p1",
            ObjectClass.PERSON,
            (60, 50, 140, 280),
            tainted=tainted,
            frame_index=frame_index,
        ),
    ]


class _RecordingClassifier(StubHelmetClassifier):
    """A stub that records the batches it was handed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.batches: list[list[Crop]] = []

    def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
        self.batches.append(list(crops))
        return super().classify(crops)


def observer(
    classifier: StubHelmetClassifier | None = None,
    config: HelmetObservationConfig | None = None,
) -> HelmetFrameObserver:
    return HelmetFrameObserver(
        classifier=classifier or StubHelmetClassifier(per_track={"p1": NO_HELMET}),
        config=config,
    )


# --- the chain end to end ----------------------------------------------------
def test_rider_on_a_motorcycle_produces_a_helmet_observation() -> None:
    obs = observer()
    obs.observe(frame(), rider_on_bike())

    derivation = obs.derivation()
    assert len(derivation.observations) == 1
    observation = derivation.observations[0]
    assert observation.track_id == "p1"
    assert observation.helmet_state is HelmetState.NO_HELMET
    assert observation.rider_slot is RiderSlot.DRIVER
    assert observation.confidence == 0.85


def test_association_is_recorded_alongside_the_observation() -> None:
    """The observation names the rider; the Association names their motorcycle."""

    obs = observer()
    obs.observe(frame(), rider_on_bike())

    associations = obs.associations()
    assert len(associations) == 1
    assert associations[0].subject_track_id == "p1"
    assert associations[0].object_track_id == "m1"
    assert associations[0].association_type is AssociationType.RIDER_OF_MOTORCYCLE


def test_pedestrian_far_from_any_bike_produces_nothing() -> None:
    obs = observer()
    obs.observe(
        frame(),
        [
            state("m1", ObjectClass.MOTORCYCLE, (50, 150, 150, 300)),
            state("p9", ObjectClass.PERSON, (350, 50, 390, 280)),
        ],
    )

    assert obs.derivation().observations == ()


def test_frame_with_no_states_produces_nothing() -> None:
    obs = observer()
    obs.observe(frame(), [])
    assert obs.derivation().observations == ()


def test_observing_produces_no_event_or_persistence() -> None:
    """The unit's headline boundary: observations only, nothing decided."""

    obs = observer()
    obs.observe(frame(), rider_on_bike())

    derivation = obs.derivation()
    assert all(
        type(o).__name__ == "HelmetStateObservation" for o in derivation.observations
    )
    assert not hasattr(obs, "persist")
    assert not hasattr(obs, "events")


# --- batching (the reason the seam takes a sequence) -------------------------
def test_all_riders_in_a_frame_are_classified_in_one_batch() -> None:
    classifier = _RecordingClassifier(per_track={"p1": NO_HELMET, "p2": HELMET})
    obs = observer(classifier)

    obs.observe(
        frame(),
        [
            state("m1", ObjectClass.MOTORCYCLE, (50, 150, 200, 300)),
            state("p1", ObjectClass.PERSON, (60, 50, 140, 280)),
            state("p2", ObjectClass.PERSON, (120, 50, 195, 280)),
        ],
    )

    assert len(classifier.batches) == 1
    assert len(classifier.batches[0]) == 2


def test_gated_crops_never_reach_the_classifier() -> None:
    """An unusable crop must cost no inference."""

    classifier = _RecordingClassifier(per_track={"p1": NO_HELMET})
    obs = observer(
        classifier, HelmetObservationConfig(head_crop=HeadCropConfig(min_crop_height_px=1000.0))
    )

    obs.observe(frame(), rider_on_bike())

    assert classifier.batches == []
    assert obs.derivation().observations[0].helmet_state is HelmetState.UNCERTAIN


def test_classifier_is_not_called_when_no_rider_is_associated() -> None:
    classifier = _RecordingClassifier()
    obs = observer(classifier)

    obs.observe(frame(), [state("m1", ObjectClass.MOTORCYCLE, (50, 150, 150, 300))])

    assert classifier.batches == []


def test_crops_carry_frame_and_rider_identity() -> None:
    classifier = _RecordingClassifier(per_track={"p1": NO_HELMET})
    obs = observer(classifier)

    obs.observe(frame(7), rider_on_bike(frame_index=7))

    crop = classifier.batches[0][0]
    assert crop.frame_index == 7
    assert crop.track_id == "p1"
    assert crop.camera_id == "cam-1"
    assert crop.image is not None


# --- quality gating ----------------------------------------------------------
def test_missing_frame_pixels_abstain_without_fabricating_confidence() -> None:
    obs = observer()
    obs.observe(frame(with_pixels=False), rider_on_bike())

    observation = obs.derivation().observations[0]
    assert observation.helmet_state is HelmetState.UNCERTAIN
    assert observation.confidence is None


def test_abstention_reasons_are_recorded_as_diagnostics() -> None:
    obs = observer(
        config=HelmetObservationConfig(head_crop=HeadCropConfig(min_crop_height_px=1000.0))
    )
    obs.observe(frame(), rider_on_bike())

    assert len(obs.derivation().abstentions) == 1
    assert "below" in obs.derivation().abstentions[0]


# --- taint -------------------------------------------------------------------
def test_tainted_rider_emits_no_observation() -> None:
    obs = observer()
    obs.observe(frame(), rider_on_bike(tainted=True))

    assert obs.derivation().observations == ()


def test_first_clean_observation_after_taint_is_a_restart() -> None:
    """Reasoning must never bridge an ID-switch discontinuity (§13)."""

    obs = observer()
    obs.observe(frame(0), rider_on_bike(0, tainted=True))
    obs.observe(frame(1), rider_on_bike(1))

    derivation = obs.derivation()
    assert len(derivation.observations) == 1
    assert derivation.observations[0].observation_id in derivation.taint_restart_ids


def test_clean_stream_has_no_taint_restarts() -> None:
    obs = observer()
    obs.observe(frame(0), rider_on_bike(0))
    obs.observe(frame(1), rider_on_bike(1))

    assert obs.derivation().taint_restart_ids == frozenset()


# --- temporal readiness ------------------------------------------------------
def test_observations_accumulate_across_frames_in_timestamp_order() -> None:
    """P4-U5 consumes this stream; it must be ordered and complete."""

    obs = observer()
    for index in (2, 0, 1):  # even out of order, the stream sorts
        obs.observe(frame(index), rider_on_bike(index))

    observations = obs.derivation().observations
    assert len(observations) == 3
    assert [o.timestamp for o in observations] == sorted(o.timestamp for o in observations)


def test_each_frame_yields_a_distinct_observation_id() -> None:
    obs = observer()
    obs.observe(frame(0), rider_on_bike(0))
    obs.observe(frame(1), rider_on_bike(1))

    assert len({o.observation_id for o in obs.derivation().observations}) == 2


# --- determinism + reset -----------------------------------------------------
def test_reset_clears_the_stream_for_replay() -> None:
    obs = observer()
    obs.observe(frame(), rider_on_bike())
    obs.reset()

    assert obs.derivation().observations == ()
    assert obs.associations() == ()
    assert obs.derivation().abstentions == ()


def test_replay_after_reset_reproduces_an_identical_stream() -> None:
    obs = observer()

    def run() -> list[str]:
        obs.reset()
        for index in range(3):
            obs.observe(frame(index), rider_on_bike(index))
        return [o.model_dump_json() for o in obs.derivation().observations]

    assert run() == run()


def test_output_is_independent_of_state_order_within_a_frame() -> None:
    def run(reverse: bool) -> list[str]:
        obs = observer(StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": HELMET}))
        states = [
            state("m1", ObjectClass.MOTORCYCLE, (50, 150, 200, 300)),
            state("p1", ObjectClass.PERSON, (60, 50, 140, 280)),
            state("p2", ObjectClass.PERSON, (120, 50, 195, 280)),
        ]
        obs.observe(frame(), list(reversed(states)) if reverse else states)
        return [o.model_dump_json() for o in obs.derivation().observations]

    assert run(False) == run(True)


def test_two_riders_on_one_bike_are_both_unknown_slot() -> None:
    obs = observer(StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": NO_HELMET}))
    obs.observe(
        frame(),
        [
            state("m1", ObjectClass.MOTORCYCLE, (50, 150, 200, 300)),
            state("p1", ObjectClass.PERSON, (60, 50, 140, 280)),
            state("p2", ObjectClass.PERSON, (120, 50, 195, 280)),
        ],
    )

    observations = obs.derivation().observations
    assert len(observations) == 2
    assert all(o.rider_slot is RiderSlot.UNKNOWN for o in observations)


# --- the observer satisfies the FrameObserver protocol -----------------------
def test_observer_plugs_into_the_composition_pipeline_hook() -> None:
    """Structural conformance to the P4-U2 FrameObserver protocol."""

    from trafficpulse.pipeline.base import FrameObserver

    def accepts(_: FrameObserver) -> bool:
        return True

    assert accepts(observer())

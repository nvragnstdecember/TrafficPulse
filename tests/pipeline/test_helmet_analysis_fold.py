"""Helmet analysis: perception without enforcement, and the fold that reports it.

Driven through the **real** observer with a scripted stub classifier -- association,
head-crop geometry and the quality gate are the shipped ones, so what is asserted here
is what a real run produces, minus the model. No ML, no weights, no network.

The headline invariants, and why each is load-bearing:

* an analysis mints no event and has no strategy to mint one with;
* a multi-rider motorcycle yields **no driver**, under any reading of the geometry;
* a classification and an enforcement status are separate values that never collapse.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trafficpulse.classifier import Crop, RawHelmetPrediction, StubHelmetClassifier
from trafficpulse.contracts import BoundingBox, TrackState
from trafficpulse.contracts.enums import ObjectClass, RiderSlot, TrackStatus
from trafficpulse.detector.frame import Frame
from trafficpulse.observations.helmet import HeadCropConfig, HelmetObservationConfig
from trafficpulse.observations.helmet_stability import HelmetStabilizationConfig
from trafficpulse.pipeline.helmet_analysis import (
    HelmetAnalysisObserver,
    RiderEnforcementStatus,
)

BASE = datetime(1970, 1, 1, tzinfo=UTC)
NO_HELMET = RawHelmetPrediction("no_helmet", 0.88)
HELMET = RawHelmetPrediction("helmet", 0.93)


def _image(height: int = 400, width: int = 400) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([(((ys // 2) + (xs // 2)) % 2 * 255).astype(np.uint8)] * 3, axis=-1)


def _frame(index: int) -> Frame:
    return Frame(
        camera_id="cam-1",
        frame_index=index,
        timestamp=BASE + timedelta(seconds=index),
        image=_image(),
    )


def _state(
    track_id: str,
    object_class: ObjectClass,
    box: tuple[float, float, float, float],
    *,
    frame_index: int,
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
    )


def _one_rider(frame_index: int) -> list[TrackState]:
    return [
        _state("m1", ObjectClass.MOTORCYCLE, (50, 150, 150, 300), frame_index=frame_index),
        _state("p1", ObjectClass.PERSON, (60, 50, 140, 280), frame_index=frame_index),
    ]


def _two_riders(frame_index: int) -> list[TrackState]:
    """Two people astride one motorcycle -- the case with no defensible driver."""

    return [
        _state("m1", ObjectClass.MOTORCYCLE, (50, 150, 150, 300), frame_index=frame_index),
        _state("p1", ObjectClass.PERSON, (60, 50, 140, 280), frame_index=frame_index),
        _state("p2", ObjectClass.PERSON, (65, 60, 145, 285), frame_index=frame_index),
    ]


def _run(
    states_for: object,
    *,
    classifier: StubHelmetClassifier,
    frames: int = 6,
    config: HelmetObservationConfig | None = None,
    stabilization: HelmetStabilizationConfig | None = None,
) -> HelmetAnalysisObserver:
    observer = HelmetAnalysisObserver(
        classifier=classifier,
        config=config,
        capture_overlay=True,
        stabilization=stabilization,
    )
    for index in range(frames):
        observer.observe(_frame(index), states_for(index))  # type: ignore[operator]
    return observer


# --- the structural guarantee ------------------------------------------------------
def test_an_analysis_observer_has_no_way_to_produce_an_event() -> None:
    """The absence is structural, not a convention someone has to remember."""

    observer = HelmetAnalysisObserver(classifier=StubHelmetClassifier())

    assert not hasattr(observer, "finalize")
    assert not hasattr(observer, "strategy")


def test_it_delegates_to_the_shipped_observer_rather_than_reimplementing_it() -> None:
    """An analysis run and a rule run must see the same pixels and the same labels."""

    from trafficpulse.pipeline.helmet_observer import HelmetFrameObserver

    classifier = StubHelmetClassifier(per_track={"p1": NO_HELMET})
    analysis = _run(_one_rider, classifier=classifier)
    rule_side = HelmetFrameObserver(classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET}))
    for index in range(6):
        rule_side.observe(_frame(index), _one_rider(index))

    assert analysis.derivation() == rule_side.derivation()


def test_it_is_not_a_helmet_frame_observer_so_the_overlay_registry_can_tell_them_apart() -> None:
    """Dispatch is by ``isinstance``; a subclass would silently draw as a rule run."""

    from trafficpulse.pipeline.helmet_observer import HelmetFrameObserver

    observer = HelmetAnalysisObserver(classifier=StubHelmetClassifier())

    assert not isinstance(observer, HelmetFrameObserver)


# --- single rider -------------------------------------------------------------------
def test_a_lone_rider_is_classified_and_reported_eligible() -> None:
    report = _run(
        _one_rider, classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET})
    ).report()

    assert report.riders_observed == 1
    assert report.motorcycles_associated == 1
    rider = report.riders[0]
    assert rider.rider_track_id == "p1"
    assert rider.motorcycle_track_id == "m1"
    assert rider.rider_count == 1
    assert rider.multi_rider is False
    assert rider.label == "no_helmet"
    assert rider.confidence == pytest.approx(0.88)
    assert rider.enforcement is RiderEnforcementStatus.ELIGIBLE


def test_eligible_is_not_a_violation_claim() -> None:
    """The report carries classification and eligibility as separate facts."""

    report = _run(
        _one_rider, classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET})
    ).report()

    # A no_helmet reading and an "eligible" status coexist without becoming an event.
    assert report.riders[0].label == "no_helmet"
    assert report.eligible_riders == 1
    assert not hasattr(report, "events")
    assert not hasattr(report, "violations")


# --- multiple riders ----------------------------------------------------------------
def test_two_riders_on_one_motorcycle_are_both_unresolved() -> None:
    """No driver is chosen: not the front-most, largest, lowest or first-tracked."""

    report = _run(
        _two_riders,
        classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": HELMET}),
    ).report()

    assert report.riders_observed == 2
    assert report.multi_rider_riders == 2
    assert report.multi_rider_motorcycles == 1
    assert {rider.enforcement for rider in report.riders} == {
        RiderEnforcementStatus.MULTI_RIDER_UNRESOLVED
    }
    assert all(rider.rider_count == 2 for rider in report.riders)
    assert report.eligible_riders == 0


def test_a_multi_rider_still_carries_its_classification() -> None:
    """Unresolved means "not attributable", not "not observed".

    Suppressing the label would be its own kind of dishonesty -- the classifier did
    read the crop, and hiding that would misrepresent what the system did.
    """

    report = _run(
        _two_riders,
        classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": HELMET}),
    ).report()

    labels = {rider.rider_track_id: rider.label for rider in report.riders}
    assert labels == {"p1": "no_helmet", "p2": "helmet"}


def test_the_rider_slot_contract_is_not_worked_around() -> None:
    """The upstream derivation still refuses to name a slot, and the fold agrees."""

    observer = _run(
        _two_riders,
        classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": HELMET}),
    )

    slots = {observation.rider_slot for observation in observer.derivation().observations}
    assert slots == {RiderSlot.UNKNOWN}


# --- abstention ---------------------------------------------------------------------
def test_a_gated_crop_abstains_rather_than_guessing() -> None:
    """A crop below the size floor never reaches the classifier and reports uncertain."""

    report = _run(
        _one_rider,
        classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET}),
        config=HelmetObservationConfig(head_crop=HeadCropConfig(min_crop_height_px=10_000)),
    ).report()

    rider = report.riders[0]
    assert rider.label == "uncertain"
    # "Not measured" stays None; it is never fabricated as 0.0.
    assert rider.confidence is None
    assert rider.enforcement is RiderEnforcementStatus.CLASSIFICATION_ABSTAINED
    assert report.gate_abstentions > 0


def test_an_unsettled_track_is_reported_as_such_rather_than_as_a_firm_call() -> None:
    report = _run(
        _one_rider,
        classifier=StubHelmetClassifier(per_track={"p1": HELMET}),
        frames=1,
        stabilization=HelmetStabilizationConfig(min_samples=3),
    ).report()

    assert report.riders[0].settled is False
    assert report.riders[0].enforcement is RiderEnforcementStatus.UNSTABLE


def test_multi_rider_outranks_every_other_blocker() -> None:
    """An unattributable rider is unattributable however confident the classifier is."""

    report = _run(
        _two_riders,
        classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": NO_HELMET}),
        frames=1,
        stabilization=HelmetStabilizationConfig(min_samples=3),
    ).report()

    assert {rider.enforcement for rider in report.riders} == {
        RiderEnforcementStatus.MULTI_RIDER_UNRESOLVED
    }


# --- instability + determinism --------------------------------------------------------
def test_the_report_discloses_the_raw_per_frame_instability() -> None:
    """The demo must be able to show the flicker it is smoothing, not hide it."""

    flipping = StubHelmetClassifier(
        per_crop={
            (index, "p1"): (HELMET if index % 2 == 0 else NO_HELMET) for index in range(6)
        }
    )
    report = _run(_one_rider, classifier=flipping).report()

    assert report.riders[0].raw_flips == 5
    assert report.riders[0].stabilized_flips < report.riders[0].raw_flips


def test_the_fold_is_deterministic() -> None:
    def build() -> object:
        return _run(
            _two_riders,
            classifier=StubHelmetClassifier(per_track={"p1": NO_HELMET, "p2": HELMET}),
        ).report()

    assert build() == build()


def test_a_clip_with_no_riders_folds_to_an_honest_empty_report() -> None:
    """No motorcycles is a correct outcome, not a failure."""

    observer = HelmetAnalysisObserver(classifier=StubHelmetClassifier())
    observer.observe(
        _frame(0), [_state("c1", ObjectClass.CAR, (10, 10, 40, 40), frame_index=0)]
    )
    report = observer.report()

    assert report.riders_observed == 0
    assert report.riders == ()
    assert report.frames_observed == 0


def test_reset_returns_the_observer_to_its_pre_stream_state() -> None:
    observer = _run(_one_rider, classifier=StubHelmetClassifier(per_track={"p1": HELMET}))
    assert observer.report().riders_observed == 1

    observer.reset()

    assert observer.report().riders_observed == 0


# --- the classifier seam is still the only place a model is called -------------------
def test_the_classifier_is_called_once_per_frame_in_one_batch() -> None:
    class _Recording(StubHelmetClassifier):
        def __init__(self) -> None:
            super().__init__(per_track={"p1": HELMET, "p2": HELMET})
            self.batches: list[int] = []

        def classify(self, crops: Sequence[Crop]) -> Sequence[RawHelmetPrediction]:
            self.batches.append(len(crops))
            return super().classify(crops)

    classifier = _Recording()
    _run(_two_riders, classifier=classifier, frames=3)

    assert classifier.batches == [2, 2, 2]

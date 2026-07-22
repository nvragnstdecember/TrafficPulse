"""Unit tests for the real greedy-IoU tracker backend (P1-U9).

Exercises the concrete ``IouTracker`` behind the P1-U8 ``Tracker`` seam with
deterministic, in-memory ``Detection`` sequences: interface conformance, the
no-leak boundary, config validation, IoU/greedy association (including
class-constrained matching), field carry-through, id/lifecycle semantics, reset
and replay determinism, empty/lost/removed behaviour, and the temporal guards.
No network, GPU, model weights, or external tracker dependency is involved.
"""

import importlib
import sys

import pytest
from _builders import BASE, make_detection
from pydantic import ValidationError

from trafficpulse.contracts import ModelRef, TrackState
from trafficpulse.contracts.enums import ObjectClass, TrackStatus
from trafficpulse.tracking import (
    IouTracker,
    IouTrackerConfig,
    Tracker,
    TrackerConfig,
)
from trafficpulse.tracking.errors import (
    InconsistentDetectionBatchError,
    NonMonotonicFrameError,
)

# ML / tracker frameworks the permissive-only backend must NOT pull in (ADR-001).
_FORBIDDEN_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
    "boxmot",
    "bytetrack",
    "yolox",
    "lap",
    "lapx",
    "scipy",
    "filterpy",
    "cv2",
    "onnxruntime",
    "tensorflow",
)

# Boxes chosen so "same place" fully overlaps (IoU 1.0) and "far" never overlaps.
_HERE = (10.0, 10.0, 30.0, 30.0)
_FAR = (200.0, 10.0, 220.0, 30.0)


# --- interface conformance & boundary ----------------------------------------
def test_iou_tracker_satisfies_the_tracker_interface() -> None:
    assert isinstance(IouTracker(), Tracker)


def test_output_is_exactly_the_frozen_trackstate_contract() -> None:
    tracker = IouTracker()
    states = tracker.update([make_detection(0, box=_HERE)])
    assert all(type(s) is TrackState for s in states)  # no subclass / native leak


def test_running_the_backend_imports_no_ml_or_tracker_framework() -> None:
    # Evict-first (see test_track_boundary): assert that constructing + RUNNING
    # the real backend re-imports no forbidden framework, independent of what
    # other suites (e.g. the H4B training tests) loaded into this process.
    evicted = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in _FORBIDDEN_MODULES
    }
    try:
        importlib.reload(importlib.import_module("trafficpulse.tracking"))
        tracker = IouTracker()
        for i in range(3):
            tracker.update([make_detection(i, box=_HERE)])
        tracker.reset()
        leaked = [n for n in sys.modules if n.split(".")[0] in _FORBIDDEN_MODULES]
        assert leaked == [], f"running the IoU backend pulled in: {leaked}"
    finally:
        sys.modules.update(evicted)


# --- configuration validation ------------------------------------------------
def test_config_defaults_are_sane() -> None:
    config = IouTrackerConfig()
    assert 0.0 <= config.iou_threshold <= 1.0
    assert config.max_age >= 0
    assert config.min_hits >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iou_threshold": -0.1},
        {"iou_threshold": 1.5},
        {"max_age": -1},
        {"min_hits": 0},
        {"unknown_knob": 1},  # extra='forbid'
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    # Idiomatic pydantic ValidationError, exactly like RTDetrConfig (P1-U7).
    with pytest.raises(ValidationError):
        IouTrackerConfig(**kwargs)  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    config = IouTrackerConfig()
    with pytest.raises(ValidationError):
        config.iou_threshold = 0.9  # type: ignore[misc]


# --- IoU / greedy association ------------------------------------------------
def test_overlapping_same_class_detection_keeps_identity() -> None:
    tracker = IouTracker()
    ids = [tracker.update([make_detection(i, box=_HERE)])[0].track_id for i in range(4)]
    assert ids == ["iou-1", "iou-1", "iou-1", "iou-1"]


def test_non_overlapping_detection_starts_a_new_track() -> None:
    tracker = IouTracker()
    first = tracker.update([make_detection(0, box=_HERE)])[0]
    second = tracker.update([make_detection(1, box=_FAR)])[0]
    assert first.track_id == "iou-1"
    assert second.track_id == "iou-2"  # no IoU overlap -> fresh identity


def test_below_threshold_overlap_starts_a_new_track() -> None:
    # A box sharing only a sliver (IoU below the default 0.3) must not match.
    tracker = IouTracker()
    tracker.update([make_detection(0, box=(0.0, 0.0, 20.0, 20.0))])
    # Overlap region is (18..20)x(18..20)=4 over union ~= 400+400-4 -> IoU ~= 0.005.
    state = tracker.update([make_detection(1, box=(18.0, 18.0, 38.0, 38.0))])[0]
    assert state.track_id == "iou-2"


def test_greedy_matching_assigns_by_iou_not_input_order() -> None:
    tracker = IouTracker()
    # Frame 0: two tracks, "here" (iou-1) and "far" (iou-2).
    tracker.update([make_detection(0, 0, box=_HERE), make_detection(0, 1, box=_FAR)])
    # Frame 1: the "far" detection comes first in input order; matching is by IoU,
    # so it must still bind to iou-2 and the "here" detection to iou-1.
    states = tracker.update(
        [make_detection(1, 0, box=_FAR), make_detection(1, 1, box=_HERE)]
    )
    assert [s.track_id for s in states] == ["iou-2", "iou-1"]  # output in input order


def test_matching_is_class_constrained() -> None:
    tracker = IouTracker()
    tracker.update([make_detection(0, box=_HERE, object_class=ObjectClass.CAR)])
    # Same location, different class: must NOT absorb the car track.
    state = tracker.update(
        [make_detection(1, box=_HERE, object_class=ObjectClass.MOTORCYCLE)]
    )[0]
    assert state.track_id == "iou-2"
    assert state.object_class is ObjectClass.MOTORCYCLE


# --- multiple detections / classes -------------------------------------------
def test_multiple_simultaneous_tracks_get_distinct_ids() -> None:
    tracker = IouTracker()
    states = tracker.update(
        [make_detection(0, 0, box=_HERE), make_detection(0, 1, box=_FAR)]
    )
    assert [s.track_id for s in states] == ["iou-1", "iou-2"]


def test_multiple_object_classes_are_preserved() -> None:
    tracker = IouTracker()
    states = tracker.update(
        [
            make_detection(0, 0, box=_HERE, object_class=ObjectClass.CAR),
            make_detection(0, 1, box=_FAR, object_class=ObjectClass.BUS),
        ]
    )
    assert [s.object_class for s in states] == [ObjectClass.CAR, ObjectClass.BUS]


# --- determinism -------------------------------------------------------------
def _run(tracker: IouTracker) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for i in range(4):
        states = tracker.update([make_detection(i, box=_HERE)])
        out.append(tuple(s.track_id for s in states))
    return out


def test_fresh_instances_replay_equal() -> None:
    assert _run(IouTracker()) == _run(IouTracker())


def test_reset_replays_the_same_stream_including_ids() -> None:
    tracker = IouTracker()
    first = _run(tracker)
    tracker.reset()
    assert _run(tracker) == first
    # After reset the id counter is rewound: identity restarts at iou-1.
    assert first[0] == ("iou-1",)


def test_full_track_states_equal_across_repeated_runs() -> None:
    a = [IouTracker().update([make_detection(i, box=_HERE)]) for i in range(3)]
    b = [IouTracker().update([make_detection(i, box=_HERE)]) for i in range(3)]
    assert a == b  # frozen TrackState equality across independent runs


# --- lifecycle: tentative -> active, lost, removed ---------------------------
def test_min_hits_promotes_tentative_to_active() -> None:
    tracker = IouTracker(IouTrackerConfig(min_hits=3))
    statuses = [tracker.update([make_detection(i, box=_HERE)])[0].status for i in range(3)]
    assert statuses == [TrackStatus.TENTATIVE, TrackStatus.TENTATIVE, TrackStatus.ACTIVE]


def test_lost_track_rematches_within_max_age() -> None:
    tracker = IouTracker(IouTrackerConfig(max_age=5))
    tracker.update([make_detection(0, box=_HERE)])  # iou-1 here
    # Frame 1: the here-object is absent (only a far detection). iou-1 is unmatched
    # (ages, kept alive), a new track spawns for the far detection.
    tracker.update([make_detection(1, box=_FAR)])
    # Frame 2: the here-object returns and re-binds to the still-alive iou-1.
    state = tracker.update([make_detection(2, box=_HERE)])[0]
    assert state.track_id == "iou-1"


def test_removed_track_does_not_rematch_past_max_age() -> None:
    tracker = IouTracker(IouTrackerConfig(max_age=0))  # remove on first miss
    tracker.update([make_detection(0, box=_HERE)])  # iou-1
    tracker.update([make_detection(1, box=_FAR)])  # iou-1 missed -> removed; iou-2 spawns
    # The here-object returns, but iou-1 was retired: a fresh id is issued.
    state = tracker.update([make_detection(2, box=_HERE)])[0]
    assert state.track_id != "iou-1"


# --- field carry-through semantics -------------------------------------------
def test_confidence_class_box_carry_through_from_detection() -> None:
    tracker = IouTracker()
    detection = make_detection(0, box=_HERE, confidence=0.77, object_class=ObjectClass.TRUCK)
    (state,) = tracker.update([detection])
    assert state.confidence == detection.confidence
    assert state.object_class == detection.object_class
    assert state.bbox == detection.bbox  # matched-detection box, not a predicted box
    assert state.frame_index == detection.frame_index
    assert state.timestamp == detection.timestamp
    assert state.camera_id == detection.camera_id


def test_velocity_is_none_and_taint_is_false() -> None:
    tracker = IouTracker()
    (state,) = tracker.update([make_detection(0, box=_HERE)])
    assert state.velocity is None  # no Kalman -> no interpretable velocity
    assert state.tainted is False  # greedy IoU exposes no trustworthy id-switch signal


def test_track_ids_are_non_empty_and_prefixed() -> None:
    tracker = IouTracker()
    states = tracker.update(
        [make_detection(0, 0, box=_HERE), make_detection(0, 1, box=_FAR)]
    )
    assert all(s.track_id.startswith("iou-") for s in states)
    assert all(s.track_id for s in states)  # adapter never sees an empty id


def test_provenance_modelref_is_stamped_when_configured() -> None:
    ref = ModelRef(name="iou-associator", version="0.1.0")
    tracker = IouTracker(tracker_config=TrackerConfig(tracker=ref))
    (state,) = tracker.update([make_detection(0, box=_HERE)])
    assert state.tracker == ref


def test_provenance_is_unset_by_default() -> None:
    (state,) = IouTracker().update([make_detection(0, box=_HERE)])
    assert state.tracker is None


# --- temporal guards (shared P1-U8 sequencing, must still hold) --------------
def test_mixed_frame_batch_is_rejected() -> None:
    tracker = IouTracker()
    with pytest.raises(InconsistentDetectionBatchError):
        tracker.update([make_detection(0, 0, box=_HERE), make_detection(1, 1, box=_FAR)])


def test_out_of_order_frame_is_rejected() -> None:
    tracker = IouTracker()
    tracker.update([make_detection(5, box=_HERE)])
    with pytest.raises(NonMonotonicFrameError):
        tracker.update([make_detection(3, box=_HERE)])


def test_regressing_timestamp_is_rejected() -> None:
    tracker = IouTracker()
    tracker.update([make_detection(5, box=_HERE)])
    with pytest.raises(NonMonotonicFrameError):
        tracker.update([make_detection(6, box=_HERE, timestamp=BASE)])


def test_reset_clears_progress_for_out_of_order_replay() -> None:
    tracker = IouTracker()
    tracker.update([make_detection(5, box=_HERE)])
    tracker.reset()
    assert tracker.update([make_detection(3, box=_HERE)])[0].track_id == "iou-1"


# --- empty-frame semantics ---------------------------------------------------
def test_empty_frame_yields_empty_and_is_inert() -> None:
    assert IouTracker().update([]) == ()


def test_empty_frame_between_populated_frames_preserves_identity() -> None:
    tracker = IouTracker()
    assert tracker.update([make_detection(0, box=_HERE)])[0].track_id == "iou-1"
    assert tracker.update([]) == ()  # inert: no aging
    assert tracker.update([make_detection(2, box=_HERE)])[0].track_id == "iou-1"

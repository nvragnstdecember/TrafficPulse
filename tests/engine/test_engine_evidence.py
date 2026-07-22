"""Evidence frame picking + engine manifests (H6)."""

from __future__ import annotations

import pytest
from _engine_helpers import event_at

from trafficpulse.contracts.enums import ArtifactKind
from trafficpulse.engine import (
    EvidenceConfig,
    FrameStamp,
    build_engine_manifest,
    media_seconds,
    pick_evidence_frames,
)
from trafficpulse.persistence.evidence_stub import build_evidence_manifest

_CONFIG = EvidenceConfig(before_seconds=1.0, after_seconds=1.0)


def _stamps(count: int, *, interval: float = 0.5) -> list[FrameStamp]:
    return [
        FrameStamp(
            camera_id="cam-synthetic-01",
            frame_id=f"vfrm-{index}",
            frame_index=index,
            timestamp_seconds=index * interval,
        )
        for index in range(count)
    ]


# --- picking ------------------------------------------------------------------------
def test_picks_trigger_before_and_after_with_full_margins() -> None:
    stamps = _stamps(11)  # 0.0 .. 5.0 s at 0.5 s
    before, trigger, after = pick_evidence_frames(
        stamps, trigger_seconds=2.5, config=_CONFIG
    )
    assert trigger is not None and trigger.timestamp_seconds == 2.5
    assert before is not None and before.timestamp_seconds == 1.5  # trigger - 1.0
    assert after is not None and after.timestamp_seconds == 3.5  # trigger + 1.0


def test_trigger_is_the_latest_frame_at_or_before_the_event() -> None:
    stamps = _stamps(11)
    _, trigger, _ = pick_evidence_frames(stamps, trigger_seconds=2.74, config=_CONFIG)
    assert trigger is not None and trigger.timestamp_seconds == 2.5  # not 3.0


def test_stream_start_inside_the_margin_falls_back_to_the_first_frame() -> None:
    stamps = _stamps(11)
    before, trigger, _ = pick_evidence_frames(
        stamps, trigger_seconds=0.5, config=_CONFIG
    )
    assert trigger is not None and trigger.timestamp_seconds == 0.5
    assert before is not None and before.timestamp_seconds == 0.0  # earliest preceding


def test_stream_end_inside_the_margin_falls_back_to_the_last_frame() -> None:
    stamps = _stamps(11)
    _, trigger, after = pick_evidence_frames(
        stamps, trigger_seconds=4.5, config=_CONFIG
    )
    assert trigger is not None and trigger.timestamp_seconds == 4.5
    assert after is not None and after.timestamp_seconds == 5.0  # latest following


def test_trigger_on_the_first_frame_has_no_before() -> None:
    before, trigger, after = pick_evidence_frames(
        _stamps(3), trigger_seconds=0.0, config=_CONFIG
    )
    assert trigger is not None and trigger.frame_index == 0
    assert before is None  # honest absence, never a duplicate reference
    assert after is not None


def test_trigger_on_the_last_frame_has_no_after() -> None:
    before, trigger, after = pick_evidence_frames(
        _stamps(3), trigger_seconds=1.0, config=_CONFIG
    )
    assert trigger is not None and trigger.frame_index == 2
    assert after is None
    assert before is not None


def test_single_frame_stream_has_neither_margin() -> None:
    before, trigger, after = pick_evidence_frames(
        _stamps(1), trigger_seconds=0.0, config=_CONFIG
    )
    assert trigger is not None
    assert (before, after) == (None, None)


def test_empty_record_picks_nothing() -> None:
    assert pick_evidence_frames([], trigger_seconds=1.0, config=_CONFIG) == (
        None,
        None,
        None,
    )


def test_zero_margins_pick_the_adjacent_frames() -> None:
    # Margins smaller than the inter-frame gap must yield the neighbours, never
    # a duplicate reference to the trigger frame itself.
    config = EvidenceConfig(before_seconds=0.0, after_seconds=0.0)
    before, trigger, after = pick_evidence_frames(
        _stamps(5), trigger_seconds=1.0, config=config
    )
    assert trigger is not None and trigger.frame_index == 2
    assert before is not None and before.frame_index == 1
    assert after is not None and after.frame_index == 3


# --- manifests ------------------------------------------------------------------------
def test_manifest_references_actually_processed_frames() -> None:
    stamps = _stamps(11)
    event = event_at(2.5, start_seconds=1.0)
    manifest = build_engine_manifest(event, stamps, config=_CONFIG)
    assert manifest.event_id == event.event_id
    assert manifest.trigger_frame is not None
    assert manifest.trigger_frame.kind is ArtifactKind.TRIGGER_FRAME
    assert manifest.trigger_frame.locator == "frames/cam-synthetic-01/vfrm-5"
    assert manifest.trigger_frame.sha256 is None  # nothing rendered, nothing hashed
    assert manifest.before_frame is not None
    assert manifest.before_frame.locator == "frames/cam-synthetic-01/vfrm-3"
    assert manifest.after_frame is not None
    assert manifest.after_frame.locator == "frames/cam-synthetic-01/vfrm-7"


def test_manifest_trace_records_the_picked_frames() -> None:
    manifest = build_engine_manifest(event_at(2.5), _stamps(11), config=_CONFIG)
    step = manifest.rule_trace[-1]
    assert step.label == "evidence-frames"
    assert step.note is not None and "trigger=vfrm-5" in step.note
    names = [measurement.name for measurement in step.measurements]
    assert names == [
        "before_frame_media_time",
        "trigger_frame_media_time",
        "after_frame_media_time",
    ]


def test_manifest_degrades_to_the_stub_with_no_processed_frames() -> None:
    event = event_at(2.5)
    assert build_engine_manifest(event, [], config=_CONFIG) == build_evidence_manifest(event)


def test_manifest_is_deterministic() -> None:
    event = event_at(2.5, start_seconds=1.0)
    first = build_engine_manifest(event, _stamps(11), config=_CONFIG)
    second = build_engine_manifest(event, _stamps(11), config=_CONFIG)
    assert first.model_dump_json() == second.model_dump_json()


def test_media_seconds_round_trips_the_epoch_anchor() -> None:
    assert media_seconds(event_at(3.25).trigger_at) == pytest.approx(3.25)

"""FrameRecord -> detector Frame conversion (P1-U10).

The one conversion the orchestration owns: it must preserve frame identity, frame
index, and PTS media-time semantics, resolve a non-empty camera id, and never
introduce wall-clock time or FPS-derived time.
"""

from datetime import UTC, datetime, timedelta

from _pipeline_helpers import CAMERA, FRAME_INTERVAL_S, make_frame_record

from trafficpulse.pipeline import frame_record_to_frame
from trafficpulse.pipeline.wrong_way import _MEDIA_TIME_EPOCH

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def test_epoch_anchor_is_fixed_utc() -> None:
    assert _MEDIA_TIME_EPOCH == _EPOCH
    assert _MEDIA_TIME_EPOCH.tzinfo is not None


def test_frame_index_and_camera_preserved() -> None:
    record = make_frame_record(7, camera_id="cam-explicit")
    frame = frame_record_to_frame(record, camera_id="cam-explicit")
    assert frame.frame_index == 7
    assert frame.camera_id == "cam-explicit"


def test_image_carried_through_without_copy() -> None:
    record = make_frame_record(0)
    frame = frame_record_to_frame(record, camera_id=CAMERA)
    assert frame.image is record.image  # opaque payload passed by reference, not copied


def test_timestamp_is_pts_anchored_at_epoch() -> None:
    record = make_frame_record(0, timestamp_seconds=2.5)
    frame = frame_record_to_frame(record, camera_id=CAMERA)
    assert frame.timestamp == _EPOCH + timedelta(seconds=2.5)
    assert frame.timestamp.tzinfo is not None  # timezone-aware, as the seam requires


def test_inter_frame_delta_equals_pts_delta() -> None:
    # Media-time *semantics*: the load-bearing quantity for min_persistence is the
    # delta between frames, which must equal the PTS delta exactly (no FPS drift).
    a = frame_record_to_frame(make_frame_record(3), camera_id=CAMERA)
    b = frame_record_to_frame(make_frame_record(9), camera_id=CAMERA)
    assert (b.timestamp - a.timestamp).total_seconds() == 6 * FRAME_INTERVAL_S


def test_conversion_is_deterministic() -> None:
    record = make_frame_record(4)
    first = frame_record_to_frame(record, camera_id=CAMERA)
    second = frame_record_to_frame(record, camera_id=CAMERA)
    assert first.timestamp == second.timestamp
    assert first == second  # Frame equality ignores the image payload

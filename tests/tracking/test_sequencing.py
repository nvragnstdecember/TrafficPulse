"""Frame-sequencing invariant tests (P1-U8).

Single-frame batch consistency (``single_frame_key``) and strictly-ascending
frame progression (``FrameProgress``) -- the deterministic temporal guarantees
every tracker behind the seam upholds.
"""

from datetime import timedelta

import pytest
from _builders import BASE, make_detection

from trafficpulse.tracking import FrameKey, FrameProgress, single_frame_key
from trafficpulse.tracking.errors import (
    InconsistentDetectionBatchError,
    NonMonotonicFrameError,
)


# --- single_frame_key --------------------------------------------------------
def test_empty_batch_has_no_key() -> None:
    assert single_frame_key([]) is None


def test_consistent_batch_returns_shared_key() -> None:
    key = single_frame_key([make_detection(3, 0), make_detection(3, 1)])
    assert key == FrameKey("cam1", 3, BASE + timedelta(seconds=3 / 30.0))


def test_mixed_frame_index_rejected() -> None:
    with pytest.raises(InconsistentDetectionBatchError):
        single_frame_key([make_detection(3, 0), make_detection(4, 1)])


def test_mixed_camera_rejected() -> None:
    with pytest.raises(InconsistentDetectionBatchError):
        single_frame_key([make_detection(3, 0), make_detection(3, 1, camera_id="cam2")])


def test_mixed_timestamp_rejected() -> None:
    with pytest.raises(InconsistentDetectionBatchError):
        single_frame_key([make_detection(3, 0), make_detection(3, 1, timestamp=BASE)])


# --- FrameProgress -----------------------------------------------------------
def _key(frame_index: int, seconds: float) -> FrameKey:
    return FrameKey("cam1", frame_index, BASE + timedelta(seconds=seconds))


def test_strictly_ascending_frames_accepted() -> None:
    progress = FrameProgress()
    progress.advance(_key(0, 0.0))
    progress.advance(_key(1, 1.0))
    progress.advance(_key(5, 5.0))  # gaps are fine, only strict increase required


def test_equal_frame_index_rejected() -> None:
    progress = FrameProgress()
    progress.advance(_key(2, 2.0))
    with pytest.raises(NonMonotonicFrameError):
        progress.advance(_key(2, 3.0))


def test_decreasing_frame_index_rejected() -> None:
    progress = FrameProgress()
    progress.advance(_key(5, 5.0))
    with pytest.raises(NonMonotonicFrameError):
        progress.advance(_key(3, 6.0))


def test_non_advancing_timestamp_rejected() -> None:
    progress = FrameProgress()
    progress.advance(_key(5, 5.0))
    # frame_index advances but timestamp regresses -> rejected on the timestamp.
    with pytest.raises(NonMonotonicFrameError):
        progress.advance(_key(6, 1.0))


def test_equal_timestamp_rejected() -> None:
    progress = FrameProgress()
    progress.advance(_key(5, 5.0))
    with pytest.raises(NonMonotonicFrameError):
        progress.advance(_key(6, 5.0))


def test_reset_allows_replay_from_start() -> None:
    progress = FrameProgress()
    progress.advance(_key(5, 5.0))
    progress.reset()
    progress.advance(_key(0, 0.0))  # would have failed without reset

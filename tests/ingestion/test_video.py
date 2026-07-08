"""Tests for deterministic local-video ingestion (P1-U5).

Covers source validation, sequential frame iteration, deterministic source/frame
identity, media-relative timestamp semantics, resource lifecycle, and the
integration boundary (ingestion imports no reasoning layers). Fixtures are tiny
videos generated at test time with PyAV -- no downloads, no network, no system
FFmpeg, no committed binaries.
"""

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from trafficpulse.ingestion import (
    FrameRecord,
    MissingTimestampError,
    NoDecodableFramesError,
    NotARegularFileError,
    SourceNotFoundError,
    UnreadableVideoError,
    VideoIngestionError,
    VideoSourceMetadata,
    open_video,
)
from trafficpulse.ingestion.video import _media_timestamp

FRAMES = 5
WIDTH = 32
HEIGHT = 24
FPS = 10


def _write_mp4(path: Path, *, frames: int = FRAMES, width: int = WIDTH, height: int = HEIGHT,
               fps: int = FPS) -> Path:
    """Write a tiny deterministic mpeg4/mp4 clip (portable in bundled FFmpeg)."""

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        array = np.full((height, width, 3), (i * 40) % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    return _write_mp4(tmp_path / "clip.mp4")


def _write_vfr_mjpeg(path: Path) -> Path:
    """Write a tiny mjpeg/avi clip with authored non-uniform (VFR) PTS."""

    container = av.open(str(path), "w")
    stream = container.add_stream("mjpeg", rate=30)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuvj420p"
    stream.time_base = Fraction(1, 1000)
    for i, pts in enumerate((0, 100, 300, 600)):  # ms: 0.0, 0.1, 0.3, 0.6 s
        frame = av.VideoFrame.from_ndarray(
            np.full((HEIGHT, WIDTH, 3), (i * 40) % 256, dtype=np.uint8), format="rgb24"
        )
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _write_audio_only(path: Path) -> Path:
    container = av.open(str(path), "w")
    stream = container.add_stream("pcm_s16le", rate=8000)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format="s16",
                                       layout="mono")
    frame.sample_rate = 8000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


# --- source validation -------------------------------------------------------
def test_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        open_video(tmp_path / "missing.mp4")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(NotARegularFileError):
        open_video(tmp_path)


def test_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 20)
    with pytest.raises(UnreadableVideoError):
        open_video(bad)


def test_audio_only_has_no_decodable_frames(tmp_path: Path) -> None:
    with pytest.raises(NoDecodableFramesError):
        open_video(_write_audio_only(tmp_path / "audio.wav"))


def test_valid_source_opens_and_metadata_populated(tiny_video: Path) -> None:
    with open_video(tiny_video, camera_id="cam-1") as reader:
        meta = reader.metadata
    assert isinstance(meta, VideoSourceMetadata)
    assert meta.width == WIDTH
    assert meta.height == HEIGHT
    assert meta.fps == float(FPS)
    assert meta.frame_count == FRAMES
    assert meta.camera_id == "cam-1"
    assert meta.codec == "mpeg4"
    assert meta.duration_seconds == pytest.approx(FRAMES / FPS)


def test_repeated_opens_stable_source_identity(tiny_video: Path) -> None:
    a = open_video(tiny_video)
    b = open_video(tiny_video)
    assert a.metadata.source_id == b.metadata.source_id
    assert a.metadata == b.metadata
    a.close()
    b.close()


# --- frame iteration ---------------------------------------------------------
def test_first_frame_index_is_zero(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        assert next(iter(reader)).frame_index == 0


def test_frame_indices_are_sequential(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        assert [r.frame_index for r in reader] == list(range(FRAMES))


def test_frame_ids_unique_within_source(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        ids = [r.frame_id for r in reader]
    assert len(set(ids)) == len(ids) == FRAMES


def test_repeated_reads_identical_frame_ids(tiny_video: Path) -> None:
    first = [r.frame_id for r in open_video(tiny_video)]
    second = [r.frame_id for r in open_video(tiny_video)]
    assert first == second


def test_payload_shape_dtype_and_contiguity(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        record = next(iter(reader))
    assert isinstance(record.image, np.ndarray)
    assert record.image.shape == (HEIGHT, WIDTH, 3)
    assert record.image.dtype == np.uint8
    assert record.image.flags["C_CONTIGUOUS"]


def test_frame_dimensions_match_metadata(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        meta = reader.metadata
        assert all(r.width == meta.width and r.height == meta.height for r in reader)


def test_eof_terminates_cleanly_without_duplicates(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        records = list(reader)
    assert len(records) == FRAMES
    assert len({r.frame_index for r in records}) == FRAMES


# --- timestamps --------------------------------------------------------------
def test_timestamp_origin_is_zero(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        assert next(iter(reader)).timestamp_seconds == 0.0


def test_timestamps_follow_pts_time_base(tiny_video: Path) -> None:
    # Prove ingestion timestamps equal pts * time_base by re-decoding directly.
    container = av.open(str(tiny_video))
    stream = container.streams.video[0]
    time_base = stream.time_base
    expected = [float(frame.pts * time_base) for frame in container.decode(stream)]
    container.close()
    got = [r.timestamp_seconds for r in open_video(tiny_video)]
    assert got == expected


def test_non_uniform_pts_is_preserved(tmp_path: Path) -> None:
    # A VFR clip with authored non-uniform PTS. Timestamps must follow PTS, not a
    # nominal frame rate: at rate=30, frame_index/fps would give 0.0, 0.033,
    # 0.067, 0.1 -- never 0.3 or 0.6.
    path = _write_vfr_mjpeg(tmp_path / "vfr.avi")
    timestamps = [r.timestamp_seconds for r in open_video(path)]
    assert timestamps == [0.0, 0.1, 0.3, 0.6]


def test_media_timestamp_is_pts_times_time_base() -> None:
    assert _media_timestamp(1024, Fraction(1, 10240)) == pytest.approx(0.1)
    assert _media_timestamp(0, Fraction(1, 1000)) == 0.0
    # Non-uniform PTS values map straight through (no frame-rate assumption).
    tb = Fraction(1, 1000)
    assert [_media_timestamp(p, tb) for p in (0, 100, 300, 600)] == [0.0, 0.1, 0.3, 0.6]


def test_media_timestamp_missing_pts_raises() -> None:
    with pytest.raises(MissingTimestampError):
        _media_timestamp(None, Fraction(1, 1000))


def test_media_timestamp_missing_time_base_raises() -> None:
    with pytest.raises(MissingTimestampError):
        _media_timestamp(1024, None)


def test_timestamps_monotonic_increasing(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        ts = [r.timestamp_seconds for r in reader]
    assert all(ts[i + 1] > ts[i] for i in range(len(ts) - 1))


def test_timestamp_sequence_deterministic(tiny_video: Path) -> None:
    first = [r.timestamp_seconds for r in open_video(tiny_video)]
    second = [r.timestamp_seconds for r in open_video(tiny_video)]
    assert first == second


# --- identity ----------------------------------------------------------------
def test_same_source_same_index_same_frame_id(tiny_video: Path) -> None:
    a = {r.frame_index: r.frame_id for r in open_video(tiny_video)}
    b = {r.frame_index: r.frame_id for r in open_video(tiny_video)}
    assert a == b


def test_different_indices_different_frame_ids(tiny_video: Path) -> None:
    records = list(open_video(tiny_video))
    assert records[0].frame_id != records[1].frame_id


def test_identity_is_cryptographic_not_builtin_hash(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    records = list(reader)
    canonical = Path(tiny_video).resolve().as_posix()
    expected_source = "vsrc-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert reader.metadata.source_id == expected_source
    expected_frame0 = "vfrm-" + hashlib.sha256(
        f"{expected_source}\x1f0".encode()
    ).hexdigest()[:16]
    assert records[0].frame_id == expected_frame0


def test_explicit_source_id_override(tiny_video: Path) -> None:
    with open_video(tiny_video, source_id="logical-source") as reader:
        assert reader.metadata.source_id == "logical-source"
        assert next(iter(reader)).source_id == "logical-source"


# --- resource lifecycle ------------------------------------------------------
def test_context_manager_releases_and_blocks_reuse(tiny_video: Path) -> None:
    with open_video(tiny_video) as reader:
        assert len(list(reader)) == FRAMES
    with pytest.raises(VideoIngestionError):
        iter(reader)  # closed after the context


def test_explicit_close_is_idempotent(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    reader.close()
    reader.close()  # must not raise
    with pytest.raises(VideoIngestionError):
        iter(reader)


def test_partial_iteration_then_close(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    iterator = iter(reader)
    assert next(iterator).frame_index == 0
    reader.close()  # release mid-stream


def test_single_pass_iteration(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    list(reader)  # full pass auto-closes
    with pytest.raises(VideoIngestionError):
        iter(reader)


def test_failed_open_does_not_block_subsequent_opens(tiny_video: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 20)
    with pytest.raises(UnreadableVideoError):
        open_video(bad)
    with open_video(tiny_video) as reader:  # no leaked/locked resources
        assert len(list(reader)) == FRAMES


# --- property-style invariants -----------------------------------------------
def test_invariant_indices_increase_by_one(tiny_video: Path) -> None:
    indices = [r.frame_index for r in open_video(tiny_video)]
    assert all(indices[i + 1] - indices[i] == 1 for i in range(len(indices) - 1))


def test_invariant_repeated_reads_same_ordered_identities(tiny_video: Path) -> None:
    seq1 = [(r.frame_index, r.frame_id, r.timestamp_seconds) for r in open_video(tiny_video)]
    seq2 = [(r.frame_index, r.frame_id, r.timestamp_seconds) for r in open_video(tiny_video)]
    assert seq1 == seq2


def test_invariant_frame_access_does_not_mutate_metadata(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    before = reader.metadata
    records = list(reader)
    _ = records[0].image[0, 0]  # touch the payload
    assert reader.metadata == before


def _record_key(record: FrameRecord) -> tuple[str, int, float, int, int]:
    return (
        record.frame_id, record.frame_index, record.timestamp_seconds,
        record.width, record.height,
    )


def test_invariant_close_does_not_alter_emitted_records(tiny_video: Path) -> None:
    reader = open_video(tiny_video)
    records = list(reader)  # auto-closes at EOF
    snapshot = [_record_key(r) for r in records]
    reader.close()
    assert [_record_key(r) for r in records] == snapshot


# --- integration boundary ----------------------------------------------------
def _ingestion_sources() -> list[Path]:
    import trafficpulse.ingestion as pkg

    return sorted(Path(pkg.__file__).resolve().parent.glob("*.py"))


def test_frame_record_consumable_without_reasoning_layers(tiny_video: Path) -> None:
    record = next(iter(open_video(tiny_video)))
    assert isinstance(record, FrameRecord)
    # Consuming a record needs nothing from the reasoning layers.
    assert record.frame_index == 0
    assert record.image.shape == (HEIGHT, WIDTH, 3)


def test_ingestion_does_not_import_reasoning_layers() -> None:
    combined = "\n".join(p.read_text(encoding="utf-8") for p in _ingestion_sources())
    for token in (
        "trafficpulse.rules",
        "trafficpulse.synth",
        "trafficpulse.observations",
        "trafficpulse.contracts",
        "TrackState",
        "ConfirmedEvent",
        "Observation",
        "wrong_way",
    ):
        assert token not in combined, f"ingestion leaked {token!r}"

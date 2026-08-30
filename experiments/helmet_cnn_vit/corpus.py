"""Crop-corpus construction from the HELMET annotations (P4-U5).

Turns the 910 per-video annotation CSVs into a deterministic list of
:class:`CropRecord`\\ s -- one per *sampled* (track, frame) pair -- under a
:class:`SamplingPolicy` that is frozen and committed **before** any image is read
or any model is trained.

Why sample at all
-----------------
A track is annotated on up to 100 consecutive frames at 10 fps, so its crops are
near-duplicates of each other. Training on all 283,377 rows would inflate the
apparent dataset size roughly seven-fold without adding information, and would let
a handful of long tracks dominate the loss. The policy therefore takes every
``frame_stride``-th frame and then caps each track at ``max_crops_per_track``,
evenly spaced across the frames that survive.

This is a *sampling* decision, not a filtering one: it is applied identically to
both model families and to all three splits, and every excluded row is counted in
:class:`CorpusStatistics` rather than silently dropped.

Determinism
-----------
Records are ordered by ``(video_id, track_id, frame_index)`` and the selection
rule is pure integer arithmetic, so the same annotations plus the same policy
always yield a byte-identical corpus -- no RNG is involved anywhere in this module.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from helmet_rtdetr.models import _Model
from helmet_rtdetr.unified import BBox
from pydantic import Field

from .errors import InconsistentTrackLabelError, MissingAnnotationError
from .labels import HelmetState, parse_label

#: Columns every HELMET annotation CSV carries (verified across all 910 files).
ANNOTATION_COLUMNS: tuple[str, ...] = ("track_id", "frame_id", "x", "y", "w", "h", "label")

#: Frame numbering within a clip, inclusive (verified: every clip is 1..100).
FIRST_FRAME = 1
LAST_FRAME = 100


class SamplingPolicy(_Model):
    """The frozen rule for turning annotation rows into training crops.

    Serialised verbatim into the split manifest, so the corpus can be rebuilt
    exactly from the committed configuration.
    """

    frame_stride: int = Field(default=5, ge=1, le=LAST_FRAME)
    frame_offset: int = Field(default=FIRST_FRAME, ge=FIRST_FRAME, le=LAST_FRAME)
    max_crops_per_track: int = Field(default=6, ge=1)
    #: Minimum shorter-side box length. Guards against degenerate boxes only; on
    #: this corpus the median crop is ~214 px tall, so it excludes ~2% of tracks.
    min_box_side_px: float = Field(default=16.0, gt=0.0)

    def selects_frame(self, frame_index: int) -> bool:
        """Whether ``frame_index`` survives the stride filter."""

        return (frame_index - self.frame_offset) % self.frame_stride == 0


@dataclass(frozen=True, slots=True)
class CropRecord:
    """One sampled motorcycle crop: where it is, and what the driver is wearing."""

    video_id: str
    site_id: str
    track_id: str
    frame_index: int
    bbox: BBox
    driver_state: HelmetState
    rider_count: int
    any_no_helmet: bool
    source_label: str

    @property
    def crop_id(self) -> str:
        """Deterministic, content-derived id (mirrors ``UnifiedObject.object_id``)."""

        preimage = "\x1f".join(
            (self.video_id, self.track_id, str(self.frame_index), self.bbox.quantised_key())
        )
        return "crop-" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16]


class CorpusStatistics(_Model):
    """What the corpus contains, and what the policy excluded and why.

    Every count here is reported in the experiment's results, so the sampling
    policy's effect on the class balance is auditable rather than assumed.
    """

    videos: int
    tracks: int
    crops: int
    crops_per_class: dict[str, int]
    tracks_per_class: dict[str, int]
    tracks_per_rider_count: dict[int, int]
    rows_read: int
    tracks_excluded_small_box: int
    tracks_excluded_no_sampled_frame: int
    rows_excluded_by_stride: int


class CropCorpus(_Model):
    """A deterministically ordered crop corpus plus the policy that produced it."""

    policy: SamplingPolicy
    records: tuple[CropRecord, ...]
    statistics: CorpusStatistics

    model_config = _Model.model_config | {"arbitrary_types_allowed": True}

    def __len__(self) -> int:
        return len(self.records)

    def content_hash(self) -> str:
        """SHA-256 over the policy and the ordered crop ids + labels."""

        digest = hashlib.sha256()
        digest.update(self.policy.model_dump_json().encode("utf-8"))
        for record in self.records:
            digest.update(b"\n")
            digest.update(f"{record.crop_id}:{record.driver_state.value}".encode())
        return digest.hexdigest()


def site_of(video_id: str) -> str:
    """The observation site a clip belongs to (``Mandalay_1_28`` -> ``Mandalay_1``).

    HELMET clip ids are ``<site>_<index>``; the twelve sites are the coarsest
    source unit available, and are recorded on every crop so test metrics can be
    sliced per site.
    """

    site, _, index = video_id.rpartition("_")
    if not site or not index.isdigit():
        raise ValueError(f"video id {video_id!r} is not of the form <site>_<index>")
    return site


def select_frames(frame_indices: Sequence[int], policy: SamplingPolicy) -> tuple[int, ...]:
    """Pick at most ``max_crops_per_track`` evenly spaced frames from a track.

    ``frame_indices`` must be sorted ascending and already stride-filtered. When a
    track has more surviving frames than the cap, indices ``floor(i * n / k)`` are
    taken -- pure integer arithmetic, so the choice is reproducible and spreads the
    crops across the track's whole trajectory rather than clustering at its start.
    """

    count = len(frame_indices)
    cap = policy.max_crops_per_track
    if count <= cap:
        return tuple(frame_indices)
    return tuple(frame_indices[(i * count) // cap] for i in range(cap))


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ANNOTATION_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise MissingAnnotationError(f"{path}: annotation is missing columns {missing}")
        yield from reader


def build_corpus(
    annotation_dir: Path,
    *,
    policy: SamplingPolicy | None = None,
    video_ids: Iterable[str] | None = None,
) -> CropCorpus:
    """Build the crop corpus from ``annotation_dir`` (one ``<video_id>.csv`` per clip).

    ``video_ids`` restricts the build to a subset (used by the tests); by default
    every CSV in the directory is read.
    """

    policy = policy or SamplingPolicy()
    if not annotation_dir.is_dir():
        raise MissingAnnotationError(f"annotation directory {annotation_dir} does not exist")

    wanted = set(video_ids) if video_ids is not None else None
    paths = sorted(p for p in annotation_dir.glob("*.csv") if wanted is None or p.stem in wanted)
    if wanted is not None:
        absent = sorted(wanted - {p.stem for p in paths})
        if absent:
            raise MissingAnnotationError(f"no annotation CSV for videos {absent}")

    records: list[CropRecord] = []
    rows_read = 0
    rows_excluded_by_stride = 0
    excluded_small = 0
    excluded_no_frame = 0
    tracks_per_class: dict[str, int] = {}
    tracks_per_rider_count: dict[int, int] = {}
    track_total = 0

    for path in paths:
        video_id = path.stem
        site_id = site_of(video_id)
        # track_id -> (label, {frame_index: BBox}) for the stride-surviving frames
        per_track: dict[str, tuple[str, dict[int, BBox]]] = {}
        for row in _read_rows(path):
            rows_read += 1
            frame_index = int(row["frame_id"])
            label = row["label"]
            track_id = row["track_id"]

            known = per_track.get(track_id)
            if known is None:
                per_track[track_id] = (label, {})
                known = per_track[track_id]
            elif known[0] != label:
                raise InconsistentTrackLabelError(
                    f"{path}: track {track_id!r} carries both {known[0]!r} and {label!r}"
                )

            if not policy.selects_frame(frame_index):
                rows_excluded_by_stride += 1
                continue
            box = BBox(x=float(row["x"]), y=float(row["y"]), w=float(row["w"]), h=float(row["h"]))
            if min(box.w, box.h) < policy.min_box_side_px:
                continue
            known[1][frame_index] = box

        for track_id, (label, boxes) in per_track.items():
            track_total += 1
            config = parse_label(label)
            state = config.driver_state
            tracks_per_class[state.value] = tracks_per_class.get(state.value, 0) + 1
            tracks_per_rider_count[config.rider_count] = (
                tracks_per_rider_count.get(config.rider_count, 0) + 1
            )
            if not boxes:
                # Either every frame failed the size floor or none survived the stride.
                excluded_small += 1
                continue
            chosen = select_frames(sorted(boxes), policy)
            if not chosen:  # pragma: no cover - select_frames never empties a non-empty input
                excluded_no_frame += 1
                continue
            records.extend(
                CropRecord(
                    video_id=video_id,
                    site_id=site_id,
                    track_id=track_id,
                    frame_index=frame_index,
                    bbox=boxes[frame_index],
                    driver_state=state,
                    rider_count=config.rider_count,
                    any_no_helmet=config.any_no_helmet,
                    source_label=label,
                )
                for frame_index in chosen
            )

    records.sort(key=lambda r: (r.video_id, r.track_id, r.frame_index))
    crops_per_class: dict[str, int] = {}
    for record in records:
        key = record.driver_state.value
        crops_per_class[key] = crops_per_class.get(key, 0) + 1

    statistics = CorpusStatistics(
        videos=len(paths),
        tracks=track_total,
        crops=len(records),
        crops_per_class=dict(sorted(crops_per_class.items())),
        tracks_per_class=dict(sorted(tracks_per_class.items())),
        tracks_per_rider_count=dict(sorted(tracks_per_rider_count.items())),
        rows_read=rows_read,
        tracks_excluded_small_box=excluded_small,
        tracks_excluded_no_sampled_frame=excluded_no_frame,
        rows_excluded_by_stride=rows_excluded_by_stride,
    )
    return CropCorpus(policy=policy, records=tuple(records), statistics=statistics)

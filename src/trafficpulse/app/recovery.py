"""Startup recovery: rebuild the runtime indices from persisted artifacts (H10).

TrafficPulse persists what it *concludes* -- confirmed events, evidence manifests,
review journals -- but the indices the HTTP layer needs to *address* those
conclusions (:class:`~trafficpulse.app.registry.VideoStore`,
:class:`~trafficpulse.app.registry.JobStore`) were built only while the process
stayed alive. A restart left the JSON on disk and the API unable to reach it.

This module closes that gap without touching the persistence model. ``EventStore``
stays write-once, ``ReviewStore`` stays an append-only journal, and both registries
stay lightweight in-memory dictionaries. What changes is that the *runtime* state
those registries hold is now itself written down, so it can be read back.

What has to be stored, and what must not be
--------------------------------------------
A persisted :class:`~trafficpulse.contracts.ConfirmedEvent` carries no video id --
it names a camera, not an upload -- and a run directory is named only by its job
id. So the **job -> video linkage exists nowhere on disk**, and neither does an
upload's client-supplied filename or its decoded metadata. Those are the facts this
module records, in two small sidecars:

```
<storage>/videos/<video_id>.json          # VideoSnapshot
<storage>/runs/<job_id>/run.json          # RunSnapshot
```

Everything else is **derived, not stored**, which is what keeps startup cheap:

* an event id *is* a filename under ``runs/<job_id>/events/``, so the event index
  is rebuilt from a directory listing with **no JSON deserialised**;
* overlay availability is a file-existence check;
* review state needs no recovery at all -- ``ReviewStore`` is stateless and reads
  the journal per event id, so H9's decisions already survive a restart and were
  merely unreachable until the job index came back.

Startup therefore costs O(runs + videos) small reads plus O(runs) directory
listings, and never parses a ``ConfirmedEvent``.

Per-entity files, not one index
--------------------------------
A single ``index.json`` would be a mutable hot file every job has to rewrite: write
amplification, a concurrent-writer hazard, and one corruption losing the whole
repository. Independent per-entity sidecars match the layout ``EventStore`` and
``ReviewStore`` already use, are written once per state change, and lose exactly one
entity when one goes bad.

Storage independence
--------------------
This class is the only place that knows the snapshots are files. Services call
:meth:`snapshot_video` / :meth:`snapshot_job`; startup calls :meth:`recover`.
Neither names a path. Moving to SQLite (ADR-002) means reimplementing this one
class, not touching the services or the registries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..engine import EngineMetrics
from .config import AppConfig
from .registry import (
    JobRecord,
    JobStatus,
    JobStore,
    OverlayStatus,
    VideoRecord,
    VideoStore,
)

_logger = logging.getLogger("trafficpulse.app.recovery")

# Bound to the snapshot base so the loader stays generic over both record kinds
# while returning the exact type it was asked for. A classic TypeVar rather than
# PEP 695 syntax, preserving the project's Python 3.11 floor (mypy target py311).
_M = TypeVar("_M", bound=BaseModel)

#: Sidecar filename for a run's recoverable job state, inside its run directory.
#: Deliberately a sibling of ``events/`` and ``manifests/`` rather than a file
#: inside them, so ``EventStore``'s ``events/*.json`` glob never sees it.
RUN_SNAPSHOT_NAME = "run.json"


def _media_mtime(path: Path) -> datetime | None:
    """The stored file's modification time, for a snapshot with no upload instant.

    A video stored before H11 recorded no timestamp. The file's mtime is when its
    bytes were written -- which *is* when the upload was accepted, since the
    service writes the file once and never rewrites it. Reported as ``None`` when
    the filesystem cannot answer, rather than substituting "now", which would make
    every pre-H11 upload look like it arrived at boot.
    """

    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:  # pragma: no cover - the caller already proved the file exists
        return None


class VideoSnapshot(BaseModel):
    """The recoverable state of one uploaded video.

    Mirrors :class:`~trafficpulse.app.registry.VideoRecord`. ``filename`` is the
    client-supplied original name -- the stored file is named by content id, so
    this is the only place it survives. The decoded metadata is recorded rather
    than re-derived because re-opening every video through PyAV at startup would
    trade a few kilobytes of disk for per-video decode latency on every boot.
    """

    video_id: str
    filename: str
    size_bytes: int
    width: int
    height: int
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    codec: str
    uploaded_at: datetime | None = None
    """When the upload was accepted (H11). Optional so a snapshot written before
    this field existed still validates; :meth:`to_record` then supplies the media
    file's modification time, which is the closest fact the filesystem retains."""

    scene_hash: str | None = None
    """The scene revision this video is calibrated against (H12).

    The second linkage that exists nowhere else on disk. A ``SceneConfig`` names a
    camera, not an upload -- exactly as a ``ConfirmedEvent`` does -- so without
    this field a restart would recover the scenes *and* the videos and have no way
    to tell which belongs to which, and every calibrated video would silently fall
    back to the server default."""

    @classmethod
    def of(cls, record: VideoRecord) -> VideoSnapshot:
        return cls(
            video_id=record.video_id,
            filename=record.filename,
            size_bytes=record.size_bytes,
            width=record.width,
            height=record.height,
            fps=record.fps,
            frame_count=record.frame_count,
            duration_seconds=record.duration_seconds,
            codec=record.codec,
            uploaded_at=record.uploaded_at,
            scene_hash=record.scene_hash,
        )

    def to_record(self, path: Path) -> VideoRecord:
        return VideoRecord(
            video_id=self.video_id,
            filename=self.filename,
            path=path,
            size_bytes=self.size_bytes,
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=self.frame_count,
            duration_seconds=self.duration_seconds,
            codec=self.codec,
            uploaded_at=self.uploaded_at or _media_mtime(path),
            scene_hash=self.scene_hash,
        )


class RunSnapshot(BaseModel):
    """The recoverable state of one processing job.

    ``video_id`` is the field that makes this file necessary: nothing else on disk
    records which upload a run belongs to. ``metrics`` is the engine's own frozen
    snapshot, reused verbatim so a recovered job can still report truthful frame
    counts and FPS instead of zeros.

    ``event_ids`` is deliberately **absent**: it is recoverable from the run's
    ``events/`` directory listing, and duplicating it here would create two
    representations of the same fact that could disagree.
    """

    job_id: str
    video_id: str
    status: JobStatus
    frames_total: int | None = None
    error: str | None = None
    overlay_status: OverlayStatus = OverlayStatus.NONE
    metrics: EngineMetrics | None = None

    @classmethod
    def of(cls, record: JobRecord) -> RunSnapshot:
        return cls(
            job_id=record.job_id,
            video_id=record.video_id,
            status=record.status,
            frames_total=record.frames_total,
            error=record.error,
            overlay_status=record.overlay_status,
            metrics=record.metrics(),
        )


@dataclass(frozen=True)
class RecoveryReport:
    """What startup recovery found, for logging and for tests to assert on.

    ``skipped`` holds one human-readable line per artifact that could not be
    recovered. Recovery never raises: an unreadable repository degrades to a
    smaller one, and the reason is reported rather than swallowed.
    """

    videos_recovered: int = 0
    runs_recovered: int = 0
    events_indexed: int = 0
    interrupted_jobs: int = 0
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return self.videos_recovered == 0 and self.runs_recovered == 0

    def describe(self) -> str:
        return (
            f"{self.videos_recovered} video(s), {self.runs_recovered} run(s), "
            f"{self.events_indexed} event(s) indexed, "
            f"{self.interrupted_jobs} interrupted, {len(self.skipped)} skipped"
        )


class RepositoryRecovery:
    """Writes the runtime-state sidecars, and rebuilds the registries from them.

    Both halves live together on purpose: the format is written and read in one
    place, so a change to what recovery needs cannot drift from what processing
    records.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # --- paths ---------------------------------------------------------------
    def video_snapshot_path(self, video_id: str) -> Path:
        return self._config.videos_dir / f"{video_id}.json"

    def run_snapshot_path(self, job_id: str) -> Path:
        return self._config.runs_dir / job_id / RUN_SNAPSHOT_NAME

    # --- writing -------------------------------------------------------------
    def snapshot_video(self, record: VideoRecord) -> None:
        """Record an upload's recoverable state. Never raises."""

        self._write(self.video_snapshot_path(record.video_id), VideoSnapshot.of(record))

    def snapshot_job(self, record: JobRecord) -> None:
        """Record a job's recoverable state. Never raises.

        Called on every job state change, so the sidecar always reflects the last
        state the process reached -- including a job still ``running`` when it
        died, which recovery then reports as interrupted rather than losing.
        """

        self._write(self.run_snapshot_path(record.job_id), RunSnapshot.of(record))

    def _write(self, path: Path, model: BaseModel) -> None:
        # Persisting runtime state must never break the request that triggered it:
        # a failed snapshot costs recoverability, not the upload or the run.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(model.model_dump_json(), encoding="utf-8")
        except OSError:
            _logger.exception("could not write recovery snapshot %s", path)

    # --- reading -------------------------------------------------------------
    def recover(self, *, videos: VideoStore, jobs: JobStore) -> RecoveryReport:
        """Rebuild both registries from disk; return what was found.

        Deterministic: videos and runs are processed in sorted id order, so two
        recoveries of the same repository produce identical registries.
        """

        skipped: list[str] = []
        video_count = self._recover_videos(videos, skipped)
        runs, events, interrupted = self._recover_runs(jobs, skipped)
        report = RecoveryReport(
            videos_recovered=video_count,
            runs_recovered=runs,
            events_indexed=events,
            interrupted_jobs=interrupted,
            skipped=tuple(skipped),
        )
        if not report.is_empty or report.skipped:
            _logger.info("repository recovery: %s", report.describe())
        for line in report.skipped:
            _logger.warning("repository recovery skipped: %s", line)
        return report

    def _recover_videos(self, videos: VideoStore, skipped: list[str]) -> int:
        directory = self._config.videos_dir
        if not directory.is_dir():
            return 0

        recovered = 0
        for snapshot_path in sorted(directory.glob("*.json")):
            snapshot = self._read(VideoSnapshot, snapshot_path, snapshot_path.name, skipped)
            if snapshot is None:
                continue
            media = self._locate_media(directory, snapshot.video_id)
            if media is None:
                # The metadata outlived its video. Recovering the record would
                # advertise a playable upload whose bytes are gone.
                skipped.append(f"{snapshot_path.name}: no media file for {snapshot.video_id}")
                continue
            record = snapshot.to_record(media)
            if record.scene_hash is not None and not self._scene_exists(record.scene_hash):
                # The binding outlived its scene. Keeping it would send processing
                # looking for geometry that is gone; dropping it degrades the video
                # to "not calibrated", which the analyst can see and fix.
                skipped.append(
                    f"{snapshot_path.name}: bound scene {record.scene_hash} is missing"
                )
                record = replace(record, scene_hash=None)
            videos.restore(record)
            recovered += 1
        return recovered

    def _scene_exists(self, scene_hash: str) -> bool:
        return (self._config.scenes_dir / f"{scene_hash}.json").is_file()

    @staticmethod
    def _locate_media(directory: Path, video_id: str) -> Path | None:
        """The stored video file for an id, whatever container extension it used."""

        for candidate in sorted(directory.glob(f"{video_id}.*")):
            if candidate.suffix.lower() != ".json":
                return candidate
        return None

    def _recover_runs(self, jobs: JobStore, skipped: list[str]) -> tuple[int, int, int]:
        directory = self._config.runs_dir
        if not directory.is_dir():
            return 0, 0, 0

        recovered = indexed = interrupted = 0
        for run_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            snapshot_path = run_dir / RUN_SNAPSHOT_NAME
            if not snapshot_path.is_file():
                # A run persisted by a build that predates H10, or the reviews
                # directory (which is not a run). Its events are on disk but
                # nothing records which upload they belong to, so it cannot be
                # addressed; say so rather than inventing a linkage.
                if self._holds_events(run_dir):
                    skipped.append(f"{run_dir.name}: no {RUN_SNAPSHOT_NAME} (pre-H10 run)")
                continue

            snapshot = self._read(RunSnapshot, snapshot_path, run_dir.name, skipped)
            if snapshot is None:
                continue

            event_ids = self._event_ids(run_dir)
            status, error = self._settle(snapshot)
            if status is not snapshot.status:
                interrupted += 1
            jobs.restore(
                JobRecord(
                    job_id=snapshot.job_id,
                    video_id=snapshot.video_id,
                    status=status,
                    frames_total=snapshot.frames_total,
                    event_ids=event_ids,
                    error=error,
                    overlay_video=self._overlay_for(snapshot.job_id),
                    overlay_status=self._overlay_status(snapshot),
                    restored_metrics=snapshot.metrics,
                ),
                event_ids,
            )
            recovered += 1
            indexed += len(event_ids)
        return recovered, indexed, interrupted

    @staticmethod
    def _settle(snapshot: RunSnapshot) -> tuple[JobStatus, str | None]:
        """Resolve a recovered job's status.

        A job recorded as pending or running did not finish -- the thread that
        would have advanced it died with the process. Restoring it as ``running``
        would leave a client polling a job that can never progress, so it is
        settled as failed with a message that says exactly what happened.
        """

        if snapshot.status.is_terminal:
            return snapshot.status, snapshot.error
        return (
            JobStatus.FAILED,
            "processing was interrupted by a backend restart and did not complete",
        )

    @staticmethod
    def _event_ids(run_dir: Path) -> tuple[str, ...]:
        """Event ids from filenames -- the cheap half of recovery.

        An event id *is* its filename stem, so the index is rebuilt without
        opening, reading, or validating a single ``ConfirmedEvent``.
        """

        events_dir = run_dir / "events"
        if not events_dir.is_dir():
            return ()
        return tuple(sorted(path.stem for path in events_dir.glob("*.json")))

    @staticmethod
    def _holds_events(run_dir: Path) -> bool:
        events_dir = run_dir / "events"
        return events_dir.is_dir() and any(events_dir.glob("*.json"))

    def _overlay_for(self, job_id: str) -> Path | None:
        path = self._config.overlays_dir / f"{job_id}.mp4"
        return path if path.is_file() else None

    def _overlay_status(self, snapshot: RunSnapshot) -> OverlayStatus:
        """The overlay state a client should see for a recovered job.

        A render in flight when the process died will never resume, so ``PENDING``
        must not survive recovery -- a client would poll for an artifact nothing is
        producing. The file on disk is the authority: present means ready, absent
        means there is nothing to play.
        """

        if self._overlay_for(snapshot.job_id) is not None:
            return OverlayStatus.READY
        if snapshot.overlay_status is OverlayStatus.FAILED:
            return OverlayStatus.FAILED
        return OverlayStatus.NONE

    def _read(
        self, model: type[_M], path: Path, label: str, skipped: list[str]
    ) -> _M | None:
        """Load one snapshot, or record why it could not be loaded.

        Corruption is contained to the entity it belongs to: an unreadable or
        invalid sidecar costs that one video or run, never the startup. ``label``
        identifies *which* entity -- every run's sidecar is called ``run.json``, so
        the filename alone would leave an operator unable to find the bad one.
        """

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            skipped.append(f"{label}: unreadable ({exc.__class__.__name__})")
            return None
        try:
            return model.model_validate_json(raw)
        except ValidationError:
            skipped.append(f"{label}: not a valid {model.__name__}")
            return None

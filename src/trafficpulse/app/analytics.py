"""Repository-wide analytics: the one aggregation layer (H15).

Answers "what is happening in this repository right now" in a single pass, and is
the **only** place in the system that aggregates across videos, runs, events,
evidence, and review. The frontend consumes the resulting
:class:`~trafficpulse.app.models.AnalyticsSummary`; it never derives repository
figures of its own.

```
VideoStore + JobStore + ReviewStore + RenderedArtifactStore + ArtifactStore
        -> AnalyticsService.summary()
        -> AnalyticsSummary  ->  GET /api/analytics/summary  ->  dashboard
```

Cost: O(videos + jobs), never O(events)
---------------------------------------
This service **never opens an event or manifest file**. Every figure comes from
somewhere that is already cheap:

* counts and per-type histograms from the in-memory ``JobStore`` -- the histogram
  is computed once when a run succeeds (H15) and carried in its snapshot, so the
  breakdown that would otherwise require deserialising every event costs a dict
  merge;
* review progress from one directory listing (``ReviewStore.reviewed_event_ids``);
* evidence coverage from one directory listing
  (``RenderedArtifactStore.rendered_event_ids``);
* storage from ``stat()`` on files already enumerated.

That is the same discipline ``VideoLibraryService`` established for the library
listing, and it is what keeps the dashboard cheap on a repository of any size.

Wall-clock time, and only wall-clock time
-----------------------------------------
Every dated figure here comes from an instant the system genuinely recorded:
``VideoRecord.uploaded_at`` (upload), the H15 job lifecycle timestamps
(processing), and ``ReviewEntry.at`` (review). Media time is **never** treated as
a calendar date -- ``trigger_at`` and ``created_at`` are PTS values anchored at a
fixed 1970 epoch, so plotting them as dates would put every violation in the
system on 1 January 1970.

Absence is reported, never imputed
----------------------------------
A repository predating H15 has runs with no timing and no histogram. Those runs
are *excluded* from averages and breakdowns and *counted* in ``timed_jobs`` /
``uncounted_jobs``, so a client can tell a partial answer from a complete one.
Nothing is substituted for a value the system did not measure.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from .. import __version__
from ..engine import EngineMetrics
from ..evidence import ArtifactStore
from ..persistence import RenderedArtifactStore, ReviewStore
from ..persistence.errors import CorruptRecordError
from .engine_provider import EngineProvider
from .models import (
    ActivityEntry,
    AnalyticsSummary,
    EvidenceStats,
    ProcessingStats,
    RepositoryHealth,
    RepositoryOverview,
    ReviewStats,
    ViolationCount,
    ViolationStats,
)
from .registry import JobRecord, JobStatus, JobStore, VideoRecord, VideoStore

_logger = logging.getLogger("trafficpulse.analytics")

#: How many activity entries the feed carries. A feed is a glance, not a log; the
#: existing per-resource endpoints are where a full history is browsed.
DEFAULT_ACTIVITY_LIMIT = 12


class AnalyticsService:
    """Composes the existing registries into one repository-wide summary.

    Holds no state and owns no storage: every collaborator is a store or service
    that already exists, and this class only reads and combines them. The review,
    rendered-artifact, and artifact-root collaborators are optional so an
    application can be built without those layers -- absent them the corresponding
    sections report honest zeroes rather than failing.
    """

    def __init__(
        self,
        *,
        videos: VideoStore,
        jobs: JobStore,
        provider: EngineProvider,
        reviews: ReviewStore | None = None,
        rendered: RenderedArtifactStore | None = None,
        artifacts: ArtifactStore | None = None,
        overlays_dir: Path | None = None,
        activity_limit: int = DEFAULT_ACTIVITY_LIMIT,
    ) -> None:
        self._videos = videos
        self._jobs = jobs
        self._provider = provider
        self._reviews = reviews
        self._rendered = rendered
        # H16: the store itself, not its directory. It maintains its own usage
        # figure, so storage statistics cost no filesystem walk per request.
        self._artifacts = artifacts
        self._overlays_dir = overlays_dir
        self._activity_limit = activity_limit

    # --- public surface -----------------------------------------------------------
    def summary(self) -> AnalyticsSummary:
        """The complete dashboard payload, from one read of each registry.

        The registries are snapshotted once up front so every section describes the
        same instant: computing sections from separate reads would let a run that
        finished mid-request appear in one figure and not another.
        """

        videos = self._videos.videos()
        jobs = self._jobs.jobs()
        event_ids = _distinct_event_ids(jobs)
        reviewed = self._reviewed_ids()
        rendered = self._rendered_ids()

        return AnalyticsSummary(
            repository=self._repository(videos, jobs),
            processing=self._processing(jobs),
            violations=self._violations(jobs),
            evidence=self._evidence(event_ids, rendered, jobs),
            review=self._review(event_ids, reviewed),
            health=self._health(videos, jobs),
            recent_activity=self._activity(videos, jobs),
            latest_run=_latest_metrics(jobs),
        )

    # --- sections -----------------------------------------------------------------
    def _repository(
        self, videos: Sequence[VideoRecord], jobs: Sequence[JobRecord]
    ) -> RepositoryOverview:
        processed = {
            job.video_id for job in jobs if job.status is JobStatus.SUCCEEDED
        }
        # Durations are summed only over videos that report one. A container that
        # does not declare a duration is unknown, not zero-length, so counting it as
        # zero would understate the repository's footage.
        measured = [v.duration_seconds for v in videos if v.duration_seconds is not None]
        return RepositoryOverview(
            videos_total=len(videos),
            videos_processed=sum(1 for v in videos if v.video_id in processed),
            videos_unprocessed=sum(1 for v in videos if v.video_id not in processed),
            videos_calibrated=sum(1 for v in videos if v.scene_hash is not None),
            footage_seconds=sum(measured) if measured else None,
            storage_bytes=sum(v.size_bytes for v in videos),
        )

    def _processing(self, jobs: Sequence[JobRecord]) -> ProcessingStats:
        by_status = {status: 0 for status in JobStatus}
        for job in jobs:
            by_status[job.status] += 1

        durations = [
            duration
            for job in jobs
            if (duration := job.duration_seconds()) is not None
        ]
        frames = 0
        for job in jobs:
            metrics = job.metrics()
            if metrics is not None:
                frames += metrics.frames_processed

        return ProcessingStats(
            jobs_total=len(jobs),
            jobs_pending=by_status[JobStatus.PENDING],
            jobs_running=by_status[JobStatus.RUNNING],
            jobs_succeeded=by_status[JobStatus.SUCCEEDED],
            jobs_failed=by_status[JobStatus.FAILED],
            jobs_cancelled=by_status[JobStatus.CANCELLED],
            average_duration_seconds=(
                sum(durations) / len(durations) if durations else None
            ),
            timed_jobs=len(durations),
            frames_processed=frames,
        )

    def _violations(self, jobs: Sequence[JobRecord]) -> ViolationStats:
        """Per-type counts from the persisted histograms -- no event file is opened.

        Reprocessing a video produces the same content-derived event ids, so summing
        every run's histogram would count one violation several times. Runs are
        therefore folded per video, keeping only the newest succeeded run of each --
        the same run ``VideoLibraryService`` opens, so the two surfaces agree.

        **One scope, for both figures.** ``events_total`` is counted over exactly the
        representative runs ``by_type`` is built from, not over every run that ever
        succeeded. Mixing the two -- a repository-wide total beside a
        representative-run breakdown -- let one response contradict itself the moment
        a video was reprocessed with a different rule set: the total counted an
        earlier run's events while the breakdown described only the newest run's.
        Sharing the scope makes ``events_total == sum(by_type)`` hold whenever the
        breakdown is complete, and ``uncounted_jobs`` remains the declared (and only)
        reason it can fall short -- a run recovered from a pre-H15 snapshot has its
        events on disk but no histogram, so its events are counted honestly while its
        types stay unknown rather than being guessed or dropped.

        The repository-wide total is unchanged and still reported, by
        :class:`EvidenceStats` and :class:`ReviewStats`, which are genuinely about
        every event that exists rather than about the current state of each video.
        """

        counts: dict[str, int] = {}
        event_ids: set[str] = set()
        counted = uncounted = 0
        for job in _representative_jobs(jobs):
            event_ids.update(job.event_ids)
            if not job.has_violation_counts:
                uncounted += 1
                continue
            counted += 1
            for violation_type, count in job.violation_counts.items():
                counts[violation_type] = counts.get(violation_type, 0) + count

        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        return ViolationStats(
            events_total=len(event_ids),
            by_type=tuple(
                ViolationCount(violation_type=name, count=count) for name, count in ordered
            ),
            counted_jobs=counted,
            uncounted_jobs=uncounted,
        )

    def _evidence(
        self,
        event_ids: frozenset[str],
        rendered: frozenset[str],
        jobs: Sequence[JobRecord],
    ) -> EvidenceStats:
        artifacts_total, artifact_bytes = (
            self._artifacts.usage() if self._artifacts is not None else (0, 0)
        )
        return EvidenceStats(
            events_total=len(event_ids),
            events_with_artifacts=len(event_ids & rendered),
            artifacts_total=artifacts_total,
            artifact_bytes=artifact_bytes,
            overlays_available=sum(
                1
                for job in jobs
                if job.overlay_video is not None and job.overlay_video.exists()
            ),
        )

    def _review(self, event_ids: frozenset[str], reviewed: frozenset[str]) -> ReviewStats:
        touched = len(event_ids & reviewed)
        return ReviewStats(
            events_total=len(event_ids),
            events_reviewed=touched,
            events_pending=len(event_ids) - touched,
        )

    def _health(
        self, videos: Sequence[VideoRecord], jobs: Sequence[JobRecord]
    ) -> RepositoryHealth:
        return RepositoryHealth(
            engine=self._provider.describe(),
            version=__version__,
            failed_jobs=sum(1 for job in jobs if job.status is JobStatus.FAILED),
            videos_missing_media=sum(1 for v in videos if not v.path.is_file()),
            videos_uncalibrated=sum(1 for v in videos if v.scene_hash is None),
            runs_without_timing=sum(
                1
                for job in jobs
                if job.status.is_terminal and job.duration_seconds() is None
            ),
        )

    def _activity(
        self, videos: Sequence[VideoRecord], jobs: Sequence[JobRecord]
    ) -> tuple[ActivityEntry, ...]:
        """The newest wall-clock-dated events across uploads, runs, and review.

        Entries with no recorded instant are omitted rather than sorted to one end:
        an undated row in a chronological feed is noise, and the figure it would
        represent is already counted in the sections above.
        """

        entries: list[ActivityEntry] = []
        for video in videos:
            if video.uploaded_at is not None:
                entries.append(
                    ActivityEntry(
                        kind="upload",
                        at=video.uploaded_at,
                        subject_id=video.video_id,
                        summary=f"Uploaded {video.filename}",
                    )
                )
        for job in jobs:
            at = job.finished_at or job.started_at or job.submitted_at
            if at is None:
                continue
            events = len(job.event_ids)
            entries.append(
                ActivityEntry(
                    kind="run",
                    at=at,
                    subject_id=job.job_id,
                    summary=(
                        f"Run {job.status.value}"
                        + (f" with {events} violation(s)" if events else "")
                    ),
                    status=job.status.value,
                )
            )
        entries.extend(self._review_activity())

        entries.sort(key=lambda entry: (entry.at, entry.subject_id), reverse=True)
        return tuple(entries[: self._activity_limit])

    def _review_activity(self) -> list[ActivityEntry]:
        """Recent analyst decisions, read from the append-only journals.

        Bounded by construction (H16): only the ``activity_limit`` most recently
        *modified* journals are opened, selected by mtime without reading anything.
        Previously every reviewed event's journal was parsed on every request so
        that all but a dozen entries could be discarded. An unreadable journal is
        skipped with a log line rather than failing the whole dashboard.
        """

        if self._reviews is None:
            return []
        entries: list[ActivityEntry] = []
        for event_id in self._reviews.recently_reviewed_event_ids(self._activity_limit):
            try:
                history = self._reviews.history(event_id)
            except CorruptRecordError:
                _logger.warning("skipping unreadable review journal for %s", event_id)
                continue
            for entry in history:
                entries.append(
                    ActivityEntry(
                        kind="review",
                        at=entry.at,
                        subject_id=event_id,
                        summary=f"Review: {entry.action.value} by {entry.reviewer}",
                        status=entry.status_after.value,
                    )
                )
        return entries

    # --- collaborator reads -------------------------------------------------------
    def _reviewed_ids(self) -> frozenset[str]:
        return self._reviews.reviewed_event_ids() if self._reviews is not None else frozenset()

    def _rendered_ids(self) -> frozenset[str]:
        return self._rendered.rendered_event_ids() if self._rendered is not None else frozenset()


# --- helpers ------------------------------------------------------------------------
def _distinct_event_ids(jobs: Iterable[JobRecord]) -> frozenset[str]:
    """Every confirmed event id in the repository, deduplicated across runs.

    Reprocessing a video yields the same content-derived ids, so a union -- not a
    sum -- is what "how many violations does this repository hold" means. This is
    the same rule ``VideoLibraryService`` applies per video.
    """

    return frozenset(
        event_id
        for job in jobs
        if job.status is JobStatus.SUCCEEDED
        for event_id in job.event_ids
    )


def _representative_jobs(jobs: Sequence[JobRecord]) -> tuple[JobRecord, ...]:
    """The newest succeeded run per video -- the run that describes that video now.

    Jobs arrive in insertion order, so the last succeeded match wins, matching
    ``VideoLibraryService``'s choice of the run a client opens.
    """

    newest: dict[str, JobRecord] = {}
    for job in jobs:
        if job.status is JobStatus.SUCCEEDED:
            newest[job.video_id] = job
    return tuple(newest.values())


def _latest_metrics(jobs: Sequence[JobRecord]) -> EngineMetrics | None:
    """The most recent run's engine metrics, or ``None`` when no run recorded any."""

    for job in reversed(jobs):
        metrics = job.metrics()
        if metrics is not None:
            return metrics
    return None



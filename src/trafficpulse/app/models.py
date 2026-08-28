"""Pydantic request/response models -- the HTTP API's typed contract (H7A).

These models are the *only* shapes a client sees. Where a full-fidelity view is
wanted the frozen domain contracts are reused verbatim (a ``ConfirmedEvent`` is
returned as the event detail, an ``EvidenceManifest`` as the evidence response,
an ``EngineMetrics`` nested in the metrics response) -- the API neither redefines
nor duplicates them. API-specific shapes (health, upload receipt, job status,
event summary, paginated list, error envelope) are defined here because they are
presentation concerns with no domain contract.

Every model carries field descriptions so the auto-generated OpenAPI schema is
self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ConfidenceBreakdown, ReviewCase, ReviewEntry
from ..contracts.enums import ReviewAction, ReviewStatus, ViolationType
from ..contracts.scene import CalibrationStatus, SceneStatus
from ..engine import EngineMetrics, RuleConfig
from .registry import EvidenceStatus, JobStatus, OverlayStatus, VideoRecord


class _ApiModel(BaseModel):
    """Strict base for API models (unknown fields rejected on input)."""

    model_config = ConfigDict(extra="forbid")


# --- health --------------------------------------------------------------------
class HealthResponse(_ApiModel):
    """Liveness + version + readiness summary.

    The three original fields are unchanged, so every existing client keeps
    working. H16 adds strictly additive readiness detail, because "the process is
    alive" and "the process can actually do its job" are different questions and a
    container orchestrator needs both.
    """

    status: str = Field(description="Overall service status; 'ok' when serving.")
    version: str = Field(description="TrafficPulse package version.")
    engine: str = Field(
        description="Engine readiness: 'ready' when a backend is available, "
        "else 'unconfigured'."
    )
    repository: str = Field(
        default="ready",
        description="Repository readiness: 'ready' when the storage root is present "
        "and writable, else 'unavailable'. A repository that cannot be written "
        "still serves reads, so this is reported rather than turned into a 503.",
    )
    inference_available: bool = Field(
        default=False,
        description="Whether a processing job can actually run. False means every "
        "read endpoint works but POST /api/process returns 503 engine_unavailable.",
    )
    scene_configured: bool = Field(
        default=False,
        description="Whether the server has a fallback scene. Uncalibrated videos "
        "cannot be processed without one.",
    )


# --- upload --------------------------------------------------------------------
class VideoUploadResponse(_ApiModel):
    """Receipt for a stored upload."""

    video_id: str = Field(description="Content-derived id addressing the stored video.")
    filename: str = Field(description="The client-supplied original filename.")
    status: str = Field(description="Upload outcome; 'stored' on success.")
    size_bytes: int = Field(description="Stored file size in bytes.")
    width: int | None = Field(default=None, description="Decoded frame width, if known.")
    height: int | None = Field(default=None, description="Decoded frame height, if known.")
    fps: float | None = Field(default=None, description="Reported average FPS, if known.")
    frame_count: int | None = Field(
        default=None, description="Reported frame count, if the container exposes it."
    )
    duration_seconds: float | None = Field(
        default=None, description="Reported duration in seconds, if known."
    )
    codec: str = Field(description="Decoded video codec name.")

    @classmethod
    def from_record(cls, record: VideoRecord) -> VideoUploadResponse:
        """Present a stored :class:`VideoRecord` as the upload receipt."""

        return cls(
            video_id=record.video_id,
            filename=record.filename,
            status="stored",
            size_bytes=record.size_bytes,
            width=record.width,
            height=record.height,
            fps=record.fps,
            frame_count=record.frame_count,
            duration_seconds=record.duration_seconds,
            codec=record.codec,
        )


# --- video library (H11) ---------------------------------------------------------
class VideoSort(StrEnum):
    """Deterministic video orderings for the library endpoint.

    ``-uploaded_at`` is the default because browsing a repository means "what did I
    work on last". A video with no recorded upload instant sorts last rather than
    being dropped or dated arbitrarily.
    """

    UPLOADED_AT_ASC = "uploaded_at"
    UPLOADED_AT_DESC = "-uploaded_at"
    FILENAME_ASC = "filename"
    FILENAME_DESC = "-filename"


class VideoSummary(_ApiModel):
    """One row of the historical video library (H11): metadata for *browsing*.

    Deliberately not a container for a video's analysis. It carries what a list
    needs to render and to decide what to open -- identity, provenance, the state
    of the work -- and nothing that would make listing cost what opening costs. No
    event payload, no evidence manifest, no overlay metadata: those are fetched per
    video, after a selection, by the endpoints that already serve them.

    A recovered video and a freshly-uploaded one produce the same shape by
    construction: every field is read from the registries, which H10 rebuilds from
    disk, so there is no field a restart could leave un-fillable and therefore no
    way for a client to tell the two apart.
    """

    video_id: str = Field(description="Content-derived id addressing the stored video.")
    filename: str = Field(description="The client-supplied original filename.")
    uploaded_at: datetime | None = Field(
        default=None,
        description="When the upload was accepted, or null when unknown (a video "
        "stored before this was recorded, whose file modification time was also "
        "unreadable). Never fabricated.",
    )
    size_bytes: int = Field(description="Stored file size in bytes.")
    width: int = Field(description="Decoded frame width.")
    height: int = Field(description="Decoded frame height.")
    fps: float | None = Field(default=None, description="Reported average FPS, if known.")
    duration_seconds: float | None = Field(
        default=None, description="Reported duration in seconds, if known."
    )
    codec: str = Field(description="Decoded video codec name.")
    job_id: str | None = Field(
        default=None,
        description="The job a client should open for this video: its most recent "
        "succeeded run if there is one, else its most recent run of any status. "
        "Null when the video has never been processed.",
    )
    status: JobStatus | None = Field(
        default=None,
        description="Status of the job named by job_id, or null when the video has "
        "never been processed. This is the video's processing state.",
    )
    job_count: int = Field(
        default=0, description="How many processing runs exist for this video."
    )
    event_count: int = Field(
        default=0,
        description="Confirmed events across the video's succeeded runs, deduplicated "
        "by event id -- the same set GET /api/events?video_id=... returns.",
    )
    events_reviewed: int = Field(
        default=0,
        description="How many of those events an analyst has acted on (the event has "
        "a review journal). Deliberately 'acted on', not 'decided': opening a case "
        "counts. Fetch GET /api/events?video_id=... for per-event review status.",
    )
    overlay_available: bool = Field(
        default=False,
        description="True when a rendered overlay (annotated) video is ready for "
        "job_id at GET /api/process/{job_id}/overlay.",
    )
    media_available: bool = Field(
        default=False,
        description="True when the stored source video is still on disk and "
        "streamable at GET /api/videos/{video_id}/media. This is also the "
        "thumbnail-availability signal: thumbnails are captured from the played "
        "video, so a streamable source is exactly what makes one possible.",
    )
    scene_hash: str | None = Field(
        default=None,
        description="The calibrated scene revision bound to this video (H12), or "
        "null when it has not been calibrated. Fetch it at "
        "GET /api/videos/{video_id}/scene.",
    )
    supported_violations: tuple[ViolationType, ...] = Field(
        default=(),
        description="Violations that can be reasoned about for this video, given "
        "its bound scene (or the server's configured fallback scene when it has "
        "none). Calibrating a video is what adds the geometry-dependent ones. "
        "Empty when no scene resolves at all.",
    )


class VideoListResponse(_ApiModel):
    """A page of video summaries plus paging metadata (H11).

    The same envelope as :class:`EventListResponse` rather than a bare array, so
    paging is already in the contract on the day a repository outgrows one page.
    """

    items: tuple[VideoSummary, ...] = Field(description="The page of summaries.")
    total: int = Field(description="Total videos in the repository (before paging).")
    limit: int = Field(description="Applied page size.")
    offset: int = Field(description="Applied page offset.")


# --- scenes (H12) ----------------------------------------------------------------
class SceneSummary(_ApiModel):
    """A stored scene revision, described for browsing and binding.

    Not a parallel scene model: the scene itself is returned verbatim as a
    ``SceneConfig`` by ``GET /api/scenes/{scene_hash}``. This is the *summary*
    view -- the same relationship :class:`EventSummary` has to ``ConfirmedEvent``
    -- carrying what a client needs to render calibration state and decide what to
    run, without parsing forty nested models to find out.
    """

    scene_hash: str = Field(
        description="The revision's address: the scene's own deterministic "
        "scene_config_hash. Editing a scene mints a new one, so this is exactly "
        "the value stamped into every ConfirmedEvent reasoned under it."
    )
    scene_id: str = Field(
        description="The scene's logical id, stable across edits (unlike scene_hash)."
    )
    scene_name: str = Field(description="Human-readable name for this site.")
    camera_id: str = Field(description="Camera this scene describes.")
    site_id: str = Field(description="Site this camera belongs to.")
    status: SceneStatus = Field(
        description="Scene lifecycle state. Analyst-drawn scenes are 'draft': the "
        "geometry has not been verified against ground truth."
    )
    calibration_status: CalibrationStatus = Field(
        description="Metric-calibration state. 'absent' means no world/homography "
        "calibration was solved -- correct for image-space reasoning, which is what "
        "wrong-way and illegal-stopping use."
    )
    frame_width: int = Field(description="Reference frame width the geometry is drawn in.")
    frame_height: int = Field(description="Reference frame height the geometry is drawn in.")
    zone_count: int = Field(description="Zones declared.")
    has_legal_direction: bool = Field(
        description="Whether a legal travel direction is declared (wrong-way needs one)."
    )
    derived: bool = Field(
        default=False,
        description="True when this scene was derived from the clip's own observed "
        "motion rather than drawn by an analyst. A derived scene measures the frame "
        "and estimates the legal direction; it never claims a no-stopping zone, stop "
        "line or signal timing, because none of those is observable from footage. "
        "Review it before relying on wrong-way results.",
    )
    has_no_stopping_zone: bool = Field(
        description="Whether an enabled no-stopping zone is declared "
        "(illegal-stopping needs one)."
    )
    supported_violations: tuple[ViolationType, ...] = Field(
        default=(),
        description="Violations this scene can actually reason about, probed against "
        "the shipped rules themselves. A violation absent here would fail fast if "
        "requested, so a client should offer exactly this set.",
    )


class SceneValidationResponse(_ApiModel):
    """The result of checking a draft without storing it.

    Lets a calibration surface tell an analyst *why* a drawing is not yet usable
    -- and which rules it would unlock -- before they commit to saving it.
    """

    valid: bool = Field(description="Whether the draft forms a valid scene.")
    errors: tuple[str, ...] = Field(
        default=(), description="Human-readable validation failures; empty when valid."
    )
    supported_violations: tuple[ViolationType, ...] = Field(
        default=(),
        description="Violations the draft would unlock if saved. Empty when invalid.",
    )
    scene_hash: str | None = Field(
        default=None,
        description="The address this draft would be stored under, or null when "
        "invalid. Equal to the current binding's hash when nothing has changed.",
    )


# --- processing ----------------------------------------------------------------
class ProcessRequest(_ApiModel):
    """Request to process one uploaded video.

    ``rules`` is the H6 rule declaration set (reused verbatim). When omitted the
    server's configured ``default_rules`` apply, and when it has none the rule set
    is **derived from the scene resolved for this video**: every shipped rule that
    scene can legitimately support, run together. No engine/detector object is ever
    named -- only which shipped rules to run and their options.
    """

    video_id: str = Field(description="Id of a previously uploaded video.")
    rules: tuple[RuleConfig, ...] | None = Field(
        default=None,
        description="Rules to run. Omit (or send an empty list) to use the server's "
        "configured rule set, or -- when it has none -- every shipped rule the "
        "video's resolved scene supports. Rules named here are run verbatim, and "
        "one the scene cannot satisfy is refused rather than silently dropped.",
    )


class ProcessResponse(_ApiModel):
    """Receipt for a created processing job."""

    job_id: str = Field(description="Id addressing the created job.")
    video_id: str = Field(description="The video this job processes.")
    status: JobStatus = Field(description="Job status at creation time.")


class JobStatusResponse(_ApiModel):
    """Live status of a processing job.

    Fields that cannot be known truthfully are ``null`` rather than fabricated:
    ``progress`` is ``null`` while running when the total frame count is unknown,
    ``fps`` is ``null`` before two frames are processed, and
    ``estimated_remaining_seconds`` is ``null`` unless a wall-clock rate and a
    frame total are both available.
    """

    job_id: str = Field(description="The job id.")
    video_id: str = Field(description="The processed video id.")
    status: JobStatus = Field(description="pending | running | succeeded | failed.")
    progress: float | None = Field(
        default=None, description="Fraction complete in [0, 1], or null if unknown."
    )
    frames_processed: int = Field(description="Frames processed so far.")
    frames_total: int | None = Field(
        default=None, description="Total frames if the source reports it, else null."
    )
    fps: float | None = Field(
        default=None, description="Media-time processing rate (PTS-derived), or null."
    )
    estimated_remaining_seconds: float | None = Field(
        default=None, description="Estimated wall-clock seconds remaining, or null."
    )
    event_count: int = Field(description="Confirmed events produced so far.")
    error: str | None = Field(
        default=None, description="Failure message when status is 'failed', else null."
    )
    overlay_available: bool = Field(
        default=False,
        description="True when a rendered overlay (annotated) video is ready to play "
        "at GET /api/process/{job_id}/overlay.",
    )
    overlay_status: OverlayStatus = Field(
        default=OverlayStatus.NONE,
        description="Lifecycle of the annotated (overlay) video, which is rendered "
        "after inference finishes and therefore resolves *later* than the job "
        "status: 'pending' means a render is in flight (keep polling); 'ready' "
        "means it is playable at GET /api/process/{job_id}/overlay; 'none' means "
        "the run produced no overlay metadata; 'failed' means a render was "
        "attempted and did not complete. Events and evidence are queryable as soon "
        "as the job succeeds, regardless of this field.",
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.NONE,
        description="Lifecycle of this run's rendered evidence frames (H16). "
        "'pending' means a render is in flight; 'ready' means every event that "
        "could be rendered was; 'none' means there was nothing to render; 'failed' "
        "means a render was attempted and did not complete, so the stored artifacts "
        "may be partial -- POST /api/process/{job_id}/evidence/repair re-renders "
        "what is missing. A run interrupted by a restart always settles to 'failed' "
        "rather than passing for complete.",
    )


# --- events --------------------------------------------------------------------
class EventSort(StrEnum):
    """Deterministic event orderings for the list endpoint."""

    TRIGGER_AT_ASC = "trigger_at"
    TRIGGER_AT_DESC = "-trigger_at"
    EVENT_ID_ASC = "event_id"
    EVENT_ID_DESC = "-event_id"


class EventSummary(_ApiModel):
    """Compact event view for list responses (detail is the full contract).

    ``start_at`` and ``confidence`` are carried here, not just on the detail, so a
    review list can show each event's observation window and evidential strength
    without fetching every event individually. Both are read straight off the
    already-loaded :class:`~trafficpulse.contracts.ConfirmedEvent`, so the list
    costs exactly what it did before.
    """

    event_id: str = Field(description="Confirmed-event id.")
    video_id: str = Field(description="The video the event was found in.")
    job_id: str = Field(description="The job that produced the event.")
    violation_type: ViolationType = Field(description="The confirmed violation type.")
    camera_id: str = Field(description="Camera id.")
    track_ids: tuple[str, ...] = Field(description="Track ids implicated in the event.")
    start_at: datetime = Field(
        description="Media-time instant support for the violation began accruing. "
        "With trigger_at this gives the observation window the reasoner sustained."
    )
    trigger_at: datetime = Field(description="Media-time instant the violation triggered.")
    rule_id: str = Field(description="Rule that confirmed the event.")
    confidence: ConfidenceBreakdown = Field(
        description="The event's typed confidence components, verbatim. Components "
        "are published separately and any the rule did not measure are null; in "
        "particular 'aggregate' is deliberately null unless calibration has been "
        "demonstrated, so a client must not treat a missing value as zero."
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING,
        description="Current analyst-review state, folded from the event's review "
        "journal. 'pending' means no analyst has acted on it yet.",
    )


class ReviewDecisionRequest(_ApiModel):
    """One analyst action to record against an event (H9).

    ``reviewer`` is an opaque identifier supplied by the client: TrafficPulse has
    no authentication layer (architecture-review §21), so the API records *who the
    client says acted* and never implies it verified them. When identity becomes
    authenticated this field is where it lands, unchanged.
    """

    action: ReviewAction = Field(description="The action the analyst performed.")
    reviewer: str = Field(
        default="analyst",
        min_length=1,
        max_length=128,
        description="Opaque reviewer identifier; not authenticated.",
    )
    note: str | None = Field(
        default=None,
        max_length=4000,
        description="Free-text analyst note recorded with this action.",
    )
    reason: str | None = Field(
        default=None,
        max_length=1000,
        description="Short justification for a decision (optional).",
    )


class ReviewResponse(_ApiModel):
    """An event's current review case plus its full audit history (H9).

    Both are returned together because the case is a *fold over* the history --
    serving them from separate endpoints would let a client render a status and a
    log that disagree, which is precisely what the append-only design exists to
    make impossible.
    """

    case: ReviewCase = Field(description="Current review state, derived from the history.")
    history: tuple[ReviewEntry, ...] = Field(
        description="Every recorded analyst action, oldest first. Append-only."
    )


class EventListResponse(_ApiModel):
    """A page of event summaries plus paging metadata."""

    items: tuple[EventSummary, ...] = Field(description="The page of summaries.")
    total: int = Field(description="Total events matching the query (before paging).")
    limit: int = Field(description="Applied page size.")
    offset: int = Field(description="Applied page offset.")


# --- metrics -------------------------------------------------------------------
class MetricsResponse(_ApiModel):
    """Aggregate job counts plus the latest run's H6 metrics (reused verbatim)."""

    jobs_total: int = Field(description="All jobs ever submitted.")
    jobs_pending: int = Field(description="Jobs not yet started.")
    jobs_running: int = Field(description="Jobs currently processing.")
    jobs_succeeded: int = Field(description="Jobs that completed successfully.")
    jobs_failed: int = Field(description="Jobs that failed.")
    jobs_cancelled: int = Field(default=0, description="Jobs cancelled on request.")
    events_total: int = Field(description="Confirmed events across all succeeded jobs.")
    latest: EngineMetrics | None = Field(
        default=None,
        description="H6 EngineMetrics of the most recent job with metrics, or null.",
    )


class EvidenceRepairResponse(_ApiModel):
    """Outcome of re-rendering a run's missing evidence frames (H16)."""

    job_id: str = Field(description="The run whose evidence was repaired.")
    events_repaired: int = Field(
        description="Events that had no rendered artifact and now have one."
    )
    artifacts_written: int = Field(description="Rendered artifacts stored by the repair.")
    evidence_status: EvidenceStatus = Field(
        description="The run's evidence state after the repair. 'ready' when every "
        "event now has artifacts; 'failed' when some still do not."
    )


# --- analytics -----------------------------------------------------------------
class ViolationCount(_ApiModel):
    """Confirmed events of one violation type across the repository."""

    violation_type: str = Field(description="Violation type slug, e.g. 'no_helmet'.")
    count: int = Field(description="Confirmed events of this type, deduplicated by event id.")


class RepositoryOverview(_ApiModel):
    """What this repository holds."""

    videos_total: int = Field(description="Stored videos, including recovered ones.")
    videos_processed: int = Field(description="Videos with at least one succeeded run.")
    videos_unprocessed: int = Field(description="Videos no succeeded run covers yet.")
    videos_calibrated: int = Field(description="Videos bound to a calibrated scene (H12).")
    footage_seconds: float | None = Field(
        default=None,
        description="Total duration of stored footage, or null when no video reports "
        "a duration. Videos with an unknown duration are excluded, never counted as 0.",
    )
    storage_bytes: int = Field(description="Bytes of stored source video.")


class ProcessingStats(_ApiModel):
    """Run outcomes and the wall-clock cost of producing them."""

    jobs_total: int = Field(description="All runs ever submitted.")
    jobs_pending: int = Field(description="Runs not yet started.")
    jobs_running: int = Field(description="Runs currently processing.")
    jobs_succeeded: int = Field(description="Runs that completed successfully.")
    jobs_failed: int = Field(description="Runs that failed.")
    jobs_cancelled: int = Field(description="Runs cancelled on request.")
    average_duration_seconds: float | None = Field(
        default=None,
        description="Mean wall-clock duration of runs that recorded both a start and "
        "a finish. Null when none did -- runs recovered from a pre-H15 snapshot have "
        "no timing, and are excluded rather than assumed instantaneous.",
    )
    timed_jobs: int = Field(
        default=0,
        description="How many runs the average is computed from, so a client can tell "
        "a repository-wide mean from a single sample.",
    )
    frames_processed: int = Field(
        default=0, description="Frames processed across every run that recorded metrics."
    )


class ViolationStats(_ApiModel):
    """Confirmed violations across the repository."""

    events_total: int = Field(
        description="Confirmed events across the newest succeeded run of each video, "
        "deduplicated by event id. Scoped to exactly the runs by_type is built from, "
        "so the two agree: when uncounted_jobs is 0 this equals the sum of by_type."
    )
    by_type: tuple[ViolationCount, ...] = Field(
        default=(), description="Counts per violation type, most frequent first."
    )
    counted_jobs: int = Field(
        default=0,
        description="Succeeded runs whose per-type histogram was available. A run "
        "recovered from a pre-H15 snapshot contributes to events_total but not to "
        "by_type, so a client can tell a partial breakdown from a complete one.",
    )
    uncounted_jobs: int = Field(
        default=0, description="Succeeded runs with events but no recorded histogram."
    )


class EvidenceStats(_ApiModel):
    """How much confirmed evidence has actually been rendered (H14)."""

    events_total: int = Field(description="Confirmed events in the repository.")
    events_with_artifacts: int = Field(
        description="Events with at least one rendered evidence artifact."
    )
    artifacts_total: int = Field(description="Stored rendered artifacts (content-addressed).")
    artifact_bytes: int = Field(description="Bytes of stored rendered artifacts.")
    overlays_available: int = Field(description="Runs with a rendered annotated video.")


class ReviewStats(_ApiModel):
    """Analyst progress over the confirmed events."""

    events_total: int = Field(description="Confirmed events in the repository.")
    events_reviewed: int = Field(
        description="Events an analyst has acted on (the event has a review journal). "
        "'Acted on', not 'decided' -- opening a case counts, matching VideoSummary."
    )
    events_pending: int = Field(description="Events with no review activity yet.")


class ActivityEntry(_ApiModel):
    """One dated thing that happened, for the recent-activity feed.

    Deliberately a small, uniform shape: the feed mixes uploads, runs, and review
    actions, and every entry is anchored to a **wall-clock** instant the system
    actually recorded. Nothing derived from media time appears here.
    """

    kind: str = Field(description="'upload', 'run', or 'review'.")
    at: datetime = Field(description="Wall-clock instant the activity happened (UTC).")
    subject_id: str = Field(description="video_id, job_id, or event_id, per kind.")
    summary: str = Field(description="Short human-readable description.")
    status: str | None = Field(default=None, description="Outcome slug, when the kind has one.")


class RepositoryHealth(_ApiModel):
    """Operational signals an operator should see without digging."""

    engine: str = Field(description="Engine readiness: 'ready' or 'unconfigured'.")
    version: str = Field(description="TrafficPulse package version.")
    failed_jobs: int = Field(description="Runs that failed.")
    videos_missing_media: int = Field(
        description="Videos whose stored file is no longer on disk."
    )
    videos_uncalibrated: int = Field(
        description="Videos with no bound scene, which limits which rules can run."
    )
    runs_without_timing: int = Field(
        description="Runs recovered from a snapshot predating lifecycle timing (H15)."
    )


class AnalyticsSummary(_ApiModel):
    """The complete dashboard payload, composed server-side in one pass.

    One response rather than a family of widget endpoints: every section is derived
    from the same registry read, so splitting them would multiply round-trips and
    let sections disagree with each other mid-refresh.
    """

    repository: RepositoryOverview
    processing: ProcessingStats
    violations: ViolationStats
    evidence: EvidenceStats
    review: ReviewStats
    health: RepositoryHealth
    recent_activity: tuple[ActivityEntry, ...] = Field(
        default=(), description="Most recent activity first, newest-limited."
    )
    latest_run: EngineMetrics | None = Field(
        default=None, description="The most recent run's H6 EngineMetrics, verbatim."
    )


# --- errors --------------------------------------------------------------------
class ErrorDetail(_ApiModel):
    """The body of an error envelope.

    ``video_id`` is populated only for a ``duplicate_video`` conflict, so a
    client can offer to open the already-uploaded video instead of dead-ending
    the upload. It is ``null`` for every other error (additive; existing clients
    that read only ``type``/``message`` are unaffected).
    """

    type: str = Field(description="Stable machine-readable error slug.")
    message: str = Field(description="Human-readable, client-safe explanation.")
    video_id: str | None = Field(
        default=None,
        description="Existing video id for a duplicate_video conflict, else null.",
    )


class ErrorResponse(_ApiModel):
    """The uniform error envelope returned for every non-2xx application error."""

    error: ErrorDetail

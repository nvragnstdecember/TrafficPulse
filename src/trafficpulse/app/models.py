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

from ..contracts.enums import ViolationType
from ..engine import EngineMetrics, RuleConfig
from .registry import JobStatus, VideoRecord


class _ApiModel(BaseModel):
    """Strict base for API models (unknown fields rejected on input)."""

    model_config = ConfigDict(extra="forbid")


# --- health --------------------------------------------------------------------
class HealthResponse(_ApiModel):
    """Liveness + version + engine-readiness summary."""

    status: str = Field(description="Overall service status; 'ok' when serving.")
    version: str = Field(description="TrafficPulse package version.")
    engine: str = Field(
        description="Engine readiness: 'ready' when a backend is available, "
        "else 'unconfigured'."
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


# --- processing ----------------------------------------------------------------
class ProcessRequest(_ApiModel):
    """Request to process one uploaded video.

    ``rules`` is the H6 rule declaration set (reused verbatim); when omitted the
    server's configured ``default_rules`` apply. No engine/detector object is ever
    named -- only which shipped rules to run and their options.
    """

    video_id: str = Field(description="Id of a previously uploaded video.")
    rules: tuple[RuleConfig, ...] | None = Field(
        default=None,
        description="Rules to run; defaults to the server's configured rule set.",
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


# --- events --------------------------------------------------------------------
class EventSort(StrEnum):
    """Deterministic event orderings for the list endpoint."""

    TRIGGER_AT_ASC = "trigger_at"
    TRIGGER_AT_DESC = "-trigger_at"
    EVENT_ID_ASC = "event_id"
    EVENT_ID_DESC = "-event_id"


class EventSummary(_ApiModel):
    """Compact event view for list responses (detail is the full contract)."""

    event_id: str = Field(description="Confirmed-event id.")
    video_id: str = Field(description="The video the event was found in.")
    job_id: str = Field(description="The job that produced the event.")
    violation_type: ViolationType = Field(description="The confirmed violation type.")
    camera_id: str = Field(description="Camera id.")
    track_ids: tuple[str, ...] = Field(description="Track ids implicated in the event.")
    trigger_at: datetime = Field(description="Media-time instant the violation triggered.")
    rule_id: str = Field(description="Rule that confirmed the event.")


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
    events_total: int = Field(description="Confirmed events across all succeeded jobs.")
    latest: EngineMetrics | None = Field(
        default=None,
        description="H6 EngineMetrics of the most recent job with metrics, or null.",
    )


# --- errors --------------------------------------------------------------------
class ErrorDetail(_ApiModel):
    """The body of an error envelope."""

    type: str = Field(description="Stable machine-readable error slug.")
    message: str = Field(description="Human-readable, client-safe explanation.")


class ErrorResponse(_ApiModel):
    """The uniform error envelope returned for every non-2xx application error."""

    error: ErrorDetail

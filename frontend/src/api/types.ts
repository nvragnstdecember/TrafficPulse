/**
 * API contract types (H7B) mirroring the H7A FastAPI response models.
 *
 * These are the wire types the typed client returns. They are intentionally a
 * faithful mirror of `src/trafficpulse/app/models.py`; keeping them here (rather
 * than importing anything backend) is exactly the boundary H7A defines — the
 * frontend depends only on JSON shapes.
 */

export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/**
 * Lifecycle of a job's annotated (overlay) video — a separate axis from `JobStatus`.
 *
 * The overlay is rendered *after* inference finishes, so it resolves later than the
 * job itself: a job can be `succeeded` (events queryable, review can start) while
 * its overlay is still `pending`. `none` = the run produced no overlay metadata,
 * `failed` = a render was attempted and did not complete; both mean "play the
 * original upload".
 */
export type OverlayStatus = 'none' | 'pending' | 'ready' | 'failed';

export type ViolationType =
  | 'no_helmet'
  | 'triple_riding'
  | 'red_light_jumping'
  | 'wrong_way'
  | 'illegal_stopping'
  | 'speeding';

export type EventSort = 'trigger_at' | '-trigger_at' | 'event_id' | '-event_id';

export interface HealthResponse {
  status: string;
  version: string;
  engine: string;
}

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  status: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
  fps: number | null;
  frame_count: number | null;
  duration_seconds: number | null;
  codec: string;
}

/** Traffic-signal states the scene contract models. */
export type SignalState = 'red' | 'amber' | 'green' | 'off' | 'unknown';

/** Scene lifecycle. Analyst-drawn scenes are `draft` — geometry, not ground truth. */
export type SceneStatus =
  | 'draft'
  | 'calibration_pending'
  | 'validation_pending'
  | 'validated'
  | 'archived';

/** Metric-calibration state. `absent` = image-space only, which is what we solve. */
export type CalibrationStatus =
  | 'absent'
  | 'provisional'
  | 'unverified'
  | 'validated'
  | 'rejected';

/** The zone kinds an analyst can draw. Mirrors the scene contract's vocabulary. */
export type ZoneType =
  | 'lane'
  | 'approach'
  | 'exit'
  | 'intersection'
  | 'no_stopping'
  | 'speed_measurement'
  | 'signal_controlled_region'
  | 'roi';

/** A point in the video's own pixel space (origin top-left, +x right, +y down). */
export type ScenePoint = [number, number];

export interface ZoneDraft {
  zone_id: string;
  zone_type: ZoneType;
  polygon: ScenePoint[];
  description?: string | null;
}

export interface DirectionDraft {
  direction_id?: string;
  dx: number;
  dy: number;
  zone_id: string;
  description?: string;
}

export interface StopLineDraft {
  stop_line_id: string;
  a: ScenePoint;
  b: ScenePoint;
  crossing_dx: number;
  crossing_dy: number;
  signal_group_id: string;
  zone_ids?: string[];
}

export interface SignalGroupDraft {
  signal_group_id: string;
  roi_polygon: ScenePoint[];
  zone_ids?: string[];
}

export interface RuleTuning {
  heading_deviation_max_degrees?: number | null;
  wrong_way_min_persistence_seconds?: number | null;
  stationary_duration_seconds?: number | null;
  red_light_min_persistence_seconds?: number | null;
}

/**
 * The minimal analyst-authorable description of a site.
 *
 * Deliberately *not* a parallel scene model: the backend expands this into the
 * frozen `SceneConfig` and always returns that. Everything omitted here —
 * provenance, statuses, schema versions, calibration blocks — is bookkeeping that
 * must be correct rather than chosen, so the client never authors it.
 */
export interface SceneDraft {
  scene_name: string;
  camera_id: string;
  site_id?: string;
  description?: string;
  frame_width: number;
  frame_height: number;
  zones: ZoneDraft[];
  direction?: DirectionDraft | null;
  stop_lines?: StopLineDraft[];
  signal_groups?: SignalGroupDraft[];
  tuning?: RuleTuning;
}

/** A stored scene revision, summarised for browsing and binding. */
export interface SceneSummary {
  /** The revision's address: the scene's own content hash, as stamped on events. */
  scene_hash: string;
  /** The logical id, stable across edits (unlike `scene_hash`). */
  scene_id: string;
  scene_name: string;
  camera_id: string;
  site_id: string;
  status: SceneStatus;
  calibration_status: CalibrationStatus;
  frame_width: number;
  frame_height: number;
  zone_count: number;
  has_legal_direction: boolean;
  has_no_stopping_zone: boolean;
  /** What this scene can actually reason about, probed against the shipped rules. */
  supported_violations: ViolationType[];
}

export interface SceneValidationResponse {
  valid: boolean;
  errors: string[];
  supported_violations: ViolationType[];
  /** The address this draft would be stored under, or null when invalid. */
  scene_hash: string | null;
}

/** One declared phase of a run's signal schedule (H13), in media seconds. */
export interface SignalPhaseSpec {
  at_seconds: number;
  state: SignalState;
}

export type VideoSort = 'uploaded_at' | '-uploaded_at' | 'filename' | '-filename';

/**
 * One row of the historical video library (H11).
 *
 * Browsing metadata only — deliberately *not* a video's analysis. It carries what a
 * list needs to render and what a client needs to decide what to open; events,
 * evidence, review history, and the annotated overlay keep their own endpoints and
 * are fetched after a selection. A recovered video and a freshly uploaded one
 * produce an identical shape, so nothing here distinguishes them.
 */
export interface VideoSummary {
  video_id: string;
  filename: string;
  /** When the upload was accepted, or null when unknown. Never fabricated. */
  uploaded_at: string | null;
  size_bytes: number;
  width: number;
  height: number;
  fps: number | null;
  duration_seconds: number | null;
  codec: string;
  /** The job to open: the latest succeeded run, else the latest run, else null. */
  job_id: string | null;
  /** That job's status, or null when the video has never been processed. */
  status: JobStatus | null;
  job_count: number;
  /** Confirmed events across succeeded runs, deduplicated by event id. */
  event_count: number;
  /** How many of those an analyst has acted on (opened counts, not only decided). */
  events_reviewed: number;
  overlay_available: boolean;
  /** Whether the stored source file is still streamable (and thumbnail-able). */
  media_available: boolean;
}

export interface VideoListResponse {
  items: VideoSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ProcessResponse {
  job_id: string;
  video_id: string;
  status: JobStatus;
}

export interface JobStatusResponse {
  job_id: string;
  video_id: string;
  status: JobStatus;
  progress: number | null;
  frames_processed: number;
  frames_total: number | null;
  fps: number | null;
  estimated_remaining_seconds: number | null;
  event_count: number;
  error: string | null;
  /** True once a rendered overlay (annotated) video is ready at the overlay endpoint. */
  overlay_available: boolean;
  /** How far the annotated-video render has got; `pending` means keep polling. */
  overlay_status: OverlayStatus;
}

/**
 * The typed confidence components a rule publishes for an event.
 *
 * Deliberately *not* one number. Every component is optional and a missing one
 * means "not measured", never zero — `aggregate` in particular stays null unless
 * calibration has been demonstrated, because an uncalibrated blend read as a
 * probability of guilt is exactly what the rule layer refuses to mint.
 */
export interface ConfidenceBreakdown {
  detector?: number | null;
  classifier?: number | null;
  association?: number | null;
  temporal_consistency?: number | null;
  track_continuity?: number | null;
  geometric_margin?: number | null;
  calibration_quality?: number | null;
  aggregate?: number | null;
}

/**
 * Analyst review lifecycle states (H9).
 *
 * `false_positive` is deliberately distinct from `rejected`: rejecting says the
 * offence is not worth pursuing, marking a false positive says the system was
 * wrong. Collapsing them would destroy the only detector-quality signal review
 * produces.
 */
export type ReviewStatus =
  | 'pending'
  | 'in_review'
  | 'approved'
  | 'rejected'
  | 'false_positive'
  | 'needs_more_evidence';

/** What an analyst did. `note` and `export` are activity, not decisions. */
export type ReviewAction =
  | 'open'
  | 'note'
  | 'approve'
  | 'reject'
  | 'false_positive'
  | 'needs_more_evidence'
  | 'reopen'
  | 'export';

/** One immutable entry in an event's append-only review journal. */
export interface ReviewEntry {
  entry_id: string;
  event_id: string;
  action: ReviewAction;
  status_before: ReviewStatus;
  status_after: ReviewStatus;
  reviewer: string;
  /** Wall-clock instant of the human action (not media time). */
  at: string;
  note: string | null;
  reason: string | null;
}

/** The current review state — derived from the history, never stored beside it. */
export interface ReviewCase {
  review_case_id: string;
  evidence_package_id: string;
  event_id: string | null;
  status: ReviewStatus;
  reviewer_id: string | null;
  decided_at: string | null;
  note: string | null;
  reason: string | null;
  updated_at: string | null;
  audit_ref: string | null;
  created_at: string;
}

export interface ReviewResponse {
  case: ReviewCase;
  history: ReviewEntry[];
}

export interface ReviewDecisionRequest {
  action: ReviewAction;
  reviewer?: string;
  note?: string | null;
  reason?: string | null;
}

export interface EventSummary {
  event_id: string;
  video_id: string;
  job_id: string;
  violation_type: ViolationType;
  camera_id: string;
  track_ids: string[];
  /** Media-time instant support began accruing (with trigger_at: the window). */
  start_at: string;
  trigger_at: string;
  rule_id: string;
  confidence: ConfidenceBreakdown;
  /** Current analyst-review state, folded from the event's review journal. */
  review_status: ReviewStatus;
}

export interface EventListResponse {
  items: EventSummary[];
  total: number;
  limit: number;
  offset: number;
}

/** The full confirmed-event contract (returned verbatim by the detail endpoint). */
export interface ConfirmedEvent {
  event_id: string;
  violation_type: ViolationType;
  camera_id: string;
  track_ids: string[];
  start_at: string;
  trigger_at: string;
  end_at: string | null;
  rule_id: string;
  rule_version: string | null;
  scene_config_hash: string | null;
  code_version: string | null;
  source_hypothesis_id: string | null;
  created_at: string;
  measurements: MeasuredValue[];
  thresholds: MeasuredValue[];
  models: ModelRef[];
  confidence: ConfidenceBreakdown;
}

export interface MeasuredValue {
  name: string;
  value: number;
  unit: string | null;
}

export interface ModelRef {
  name: string;
  version: string;
  weights_hash: string | null;
}

export interface ArtifactReference {
  kind: string;
  locator: string;
  sha256: string | null;
  media_type: string | null;
}

export interface RuleTraceStep {
  index: number;
  label: string;
  note: string | null;
  measurements: MeasuredValue[];
}

export interface EvidenceManifest {
  evidence_package_id: string;
  event_id: string;
  before_frame: ArtifactReference | null;
  trigger_frame: ArtifactReference | null;
  after_frame: ArtifactReference | null;
  clip: ArtifactReference | null;
  trajectory: ArtifactReference | null;
  plate_crop: ArtifactReference | null;
  /**
   * Every rendered artifact for the event (H14), including ones with no typed slot.
   * Present on manifests the backend has merged rendered references into.
   */
  additional_artifacts?: ArtifactReference[];
  rule_trace: RuleTraceStep[];
  models: ModelRef[];
  code_version: string | null;
  scene_config_hash: string | null;
  created_at: string;
}

export interface EngineMetrics {
  frames_read: number;
  frames_skipped_stride: number;
  frames_skipped_fps: number;
  frames_dropped_backpressure: number;
  frames_admitted: number;
  frames_processed: number;
  batches_processed: number;
  detections: number;
  track_states: number;
  events_confirmed: number;
  queue_peak: number;
  media_fps: number | null;
  wall_fps: number | null;
  memory_bytes_current: number | null;
  memory_bytes_peak: number | null;
  gpu_memory_bytes_current: number | null;
  gpu_memory_bytes_peak: number | null;
}

export interface MetricsResponse {
  jobs_total: number;
  jobs_pending: number;
  jobs_running: number;
  jobs_succeeded: number;
  jobs_failed: number;
  /** Jobs cancelled on request (H7D). Optional so an older backend still types. */
  jobs_cancelled?: number;
  events_total: number;
  latest: EngineMetrics | null;
}

/** The uniform error envelope every non-2xx response carries. */
export interface ApiErrorBody {
  error: {
    type: string;
    message: string;
    /** Present only on a `duplicate_video` conflict: the existing video id (H7D). */
    video_id?: string;
  };
}

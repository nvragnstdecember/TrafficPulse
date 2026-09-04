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

/**
 * Lifecycle of a run's rendered evidence frames (H16) — the same axis as
 * `OverlayStatus`, for the evidence stills rather than the annotated video.
 *
 * `failed` means the stored artifacts may be partial: a render interrupted by a
 * restart settles here rather than passing for complete, and the repair endpoint
 * re-renders what is missing.
 */
export type EvidenceStatus = 'none' | 'pending' | 'ready' | 'failed';

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
  /**
   * Readiness detail (H16), additive so pre-H16 backends still type. `status`
   * answers "is the process alive"; these answer "can it do its job".
   */
  repository?: string;
  inference_available?: boolean;
  scene_configured?: boolean;
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
  /**
   * True when the scene was derived from the clip's own motion rather than drawn.
   * A derived scene measures the frame and estimates the legal direction; it never
   * claims a no-stopping zone, stop line or signal timing.
   */
  derived: boolean;
  /** What this scene can actually reason about, probed against the shipped rules. */
  supported_violations: ViolationType[];
}

/**
 * A stored scene revision, as much of it as the calibration surface needs.
 *
 * Deliberately a **subset** of the backend's frozen `SceneConfig`, not a mirror of
 * it: the client reads this to redraw geometry it previously authored and to show
 * the thresholds in force, and it authors a `SceneDraft` — never a `SceneConfig`.
 * Mirroring the whole contract here would invite exactly the "the browser writes its
 * own provenance and status" mistake the draft vocabulary exists to prevent.
 */
export interface StoredScene {
  scene: {
    scene_id: string;
    scene_name: string;
    description: string;
    status: SceneStatus;
    camera_id: string;
  };
  frame: { reference_width: number; reference_height: number };
  zones: Array<{
    zone_id: string;
    zone_type: ZoneType;
    enabled: boolean;
    polygon: ScenePoint[];
  }>;
  stop_lines: Array<{
    stop_line_id: string;
    enabled: boolean;
    endpoints: { a: ScenePoint; b: ScenePoint };
    crossing_direction: { dx: number; dy: number };
  }>;
  legal_directions: Array<{
    direction_id: string;
    description: string;
    vector: { dx: number; dy: number };
    zone_ids: string[];
  }>;
  signal_groups: Array<{
    signal_group_id: string;
    roi: { shape: string; polygon: ScenePoint[] | null };
  }>;
  rule_parameters: Array<{
    violation_type: ViolationType;
    parameters: Array<{
      id: string;
      value: number | null;
      unit: string;
      status: string;
      note: string | null;
    }>;
  }>;
  calibration: { source: string; type: string; status: CalibrationStatus };
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

// --- controlled demonstration: declared expectations --------------------------------
/**
 * What a controlled clip was **built** to contain, declared before it is run.
 *
 * Ground truth for a *demonstration*, and nothing else. It never reaches the engine,
 * never becomes an event, and never appears in an event listing — the backend stores
 * it in a separate place no rule or reasoner can read. Declaring it on real footage
 * would be a claim about that footage's ground truth, which this project does not
 * have for any real clip.
 */
export interface ExpectationDeclaration {
  expected_violations: ViolationType[];
  notes: string;
  declared_by: string;
}

/** A stored declaration, with the video it describes and when it was made. */
export interface ExpectationRecord extends ExpectationDeclaration {
  video_id: string;
  /** Wall-clock instant the declaration was recorded — bookkeeping, not media time. */
  declared_at: string;
}

/** How one family's expectation and detection line up. */
export type ExpectationOutcome = 'matched' | 'missing' | 'unexpected';

export interface ExpectationRow {
  violation_type: ViolationType;
  expected: boolean;
  detected_count: number;
  /** The confirmed events behind `detected_count`, so every number can be opened. */
  event_ids: string[];
  outcome: ExpectationOutcome;
}

/**
 * Declared expectations beside independently confirmed events.
 *
 * Deliberately carries **no accuracy metric**. Precision or recall over one
 * hand-authored clip would be arithmetic against ground truth the same person wrote;
 * the comparison reports matched / missing / unexpected and stops there.
 */
export interface ExpectationComparison {
  video_id: string;
  job_id: string | null;
  /** Null when nothing was declared — every detected family is then `unexpected`. */
  expectation: ExpectationRecord | null;
  rows: ExpectationRow[];
  expected_count: number;
  detected_event_count: number;
  matched_count: number;
  missing_count: number;
  unexpected_count: number;
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
  /**
   * How far the evidence-frame render has got (H16). `failed` means the stored
   * artifacts may be partial — a run interrupted by a restart settles here rather
   * than passing for complete. Optional so a pre-H16 backend still types.
   */
  evidence_status?: EvidenceStatus;
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

// --- analytics (H15) ---------------------------------------------------------
/**
 * The complete dashboard payload, aggregated server-side.
 *
 * The client renders these figures and derives none of its own: `AnalyticsService`
 * is the single aggregation layer, so any number shown on the dashboard is one the
 * backend computed from the repository.
 */
export interface ViolationCount {
  violation_type: string;
  count: number;
}

export interface RepositoryOverview {
  videos_total: number;
  videos_processed: number;
  videos_unprocessed: number;
  videos_calibrated: number;
  /** Null when no stored video declares a duration — never treat as 0. */
  footage_seconds: number | null;
  storage_bytes: number;
}

export interface ProcessingStats {
  jobs_total: number;
  jobs_pending: number;
  jobs_running: number;
  jobs_succeeded: number;
  jobs_failed: number;
  jobs_cancelled: number;
  /** Null when no run recorded both a start and a finish (pre-H15 repositories). */
  average_duration_seconds: number | null;
  timed_jobs: number;
  frames_processed: number;
}

export interface ViolationStats {
  events_total: number;
  by_type: ViolationCount[];
  counted_jobs: number;
  /** Succeeded runs with events but no recorded histogram — the breakdown is partial. */
  uncounted_jobs: number;
}

export interface EvidenceStats {
  events_total: number;
  events_with_artifacts: number;
  artifacts_total: number;
  artifact_bytes: number;
  overlays_available: number;
}

export interface ReviewProgress {
  events_total: number;
  events_reviewed: number;
  events_pending: number;
}

export interface ActivityEntry {
  kind: string;
  /** A wall-clock instant the backend recorded — never media time. */
  at: string;
  subject_id: string;
  summary: string;
  status: string | null;
}

export interface RepositoryHealth {
  engine: string;
  version: string;
  failed_jobs: number;
  videos_missing_media: number;
  videos_uncalibrated: number;
  runs_without_timing: number;
}

export interface AnalyticsSummary {
  repository: RepositoryOverview;
  processing: ProcessingStats;
  violations: ViolationStats;
  evidence: EvidenceStats;
  review: ReviewProgress;
  health: RepositoryHealth;
  recent_activity: ActivityEntry[];
  latest_run: EngineMetrics | null;
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

/**
 * How far a capability can be relied on (`/api/system/posture`).
 *
 * `limited` and `experimental` are deliberately different: `limited` works inside a
 * stated boundary, `experimental` runs but is not something the evidence supports
 * acting on. A surface that collapsed them would let unvalidated output read as
 * merely-caveated output.
 */
export type PostureState =
  | 'active'
  | 'limited'
  | 'experimental'
  | 'disabled'
  | 'unavailable';

export interface PostureComponent {
  component_id: string;
  label: string;
  state: PostureState;
  /** A complete sentence, not a metric — this is what a person must read before claiming anything. */
  detail: string;
}

/**
 * What the deployment can honestly claim. Distinct from health: a perfectly healthy
 * service can still be unable to enforce a helmet violation.
 */
export interface SystemPosture {
  components: PostureComponent[];
  helmet_backend: string | null;
  /** Exactly what the configured backend declares it can emit; empty when undeclared. */
  helmet_backend_labels: string[];
  turban_capable: boolean;
  /** Never `active`: no configuration of this system currently earns that word. */
  helmet_enforcement: PostureState;
}

/** Why a rider's helmet reading is, or is not, something a violation rule could act on. */
export type RiderEnforcementStatus =
  | 'eligible'
  | 'multi_rider_unresolved'
  | 'classification_abstained'
  | 'unstable';

export interface RiderAnalysis {
  rider_track_id: string;
  motorcycle_track_id: string | null;
  rider_count: number;
  multi_rider: boolean;
  samples: number;
  first_frame: number;
  last_frame: number;
  /** The stabilized label: helmet | no_helmet | turban | uncertain. Never a violation. */
  helmet_state: string;
  /** `null` means "not measured" — never coerce it to 0. */
  confidence: number | null;
  agreement: number;
  settled: boolean;
  /** Observed per-frame instability on this run, before smoothing. */
  raw_label_flips: number;
  stabilized_label_flips: number;
  median_head_height_px: number | null;
  enforcement: RiderEnforcementStatus;
}

export interface AnalysisLabelCount {
  label: string;
  riders: number;
}

/**
 * A finished run's helmet analysis. **No field here is a violation count** — an
 * analysis mints no event, and `/api/events` remains the only source of confirmed
 * violations.
 */
export interface HelmetAnalysis {
  job_id: string;
  enforcement: PostureState;
  frames_observed: number;
  riders_observed: number;
  motorcycles_associated: number;
  multi_rider_riders: number;
  multi_rider_motorcycles: number;
  eligible_riders: number;
  unresolved_riders: number;
  abstained_riders: number;
  unstable_riders: number;
  gate_abstentions: number;
  label_counts: AnalysisLabelCount[];
  enforcement_counts: AnalysisLabelCount[];
  riders: RiderAnalysis[];
}

// --- live camera monitoring -------------------------------------------------
/**
 * Whether this deployment can monitor a live camera, asked *before* the browser
 * requests camera permission — so a deployment that cannot monitor live says so
 * instead of failing after a person has granted access to their camera.
 */
export interface LiveReadiness {
  ready: boolean;
  detail: string;
  active_sessions: number;
  max_sessions: number;
  inference_configured: boolean;
  drawing_backend_available: boolean;
  helmet_classifier_configured: boolean;
}

export interface LiveSessionSummary {
  session_id: string;
  camera_id: string;
  width: number;
  height: number;
  scene_calibrated: boolean;
  frames_processed: number;
  uptime_seconds: number;
}

export interface LiveSessionListResponse {
  sessions: LiveSessionSummary[];
  max_sessions: number;
}

export interface LiveUnavailableViolation {
  violation_type: ViolationType;
  /** A complete sentence naming the missing evidence. Shown verbatim. */
  reason: string;
}

/** Sent once when a live session opens, before any frame. */
export interface LiveSessionMessage {
  type: 'session';
  session_id: string;
  camera_id: string;
  width: number;
  height: number;
  scene_hash: string | null;
  scene_calibrated: boolean;
  running_violations: ViolationType[];
  unavailable_violations: LiveUnavailableViolation[];
  window_frames: number;
}

export interface LiveTrack {
  track_id: string;
  object_class: string;
  status: string;
  /** `[x1, y1, x2, y2]` in the camera's own pixel space. */
  bbox: [number, number, number, number];
  confidence: number | null;
}

export interface LiveMotorcycle {
  motorcycle_track_id: string;
  rider_count: number;
  /**
   * False whenever the motorcycle carries more than one rider. The tracker
   * supplies no velocity, so no layer of this system can say which rider is
   * driving — and none of them guesses.
   */
  driver_resolved: boolean;
}

export interface LiveRider {
  rider_track_id: string;
  motorcycle_track_id: string;
  rider_count: number;
  driver_resolved: boolean;
  helmet_label: string | null;
  helmet_confidence: number | null;
  /** The crop was refused by the quality gate, so nothing was classified. */
  helmet_gated: boolean;
}

/** Every value here is counted or timed by the server. None is estimated. */
export interface LiveStats {
  frames_received: number;
  frames_dropped: number;
  frames_processed: number;
  frames_rejected: number;
  frames_out_of_order: number;
  active_tracks: number;
  events_emitted: number;
  windows_completed: number;
  window_frames_processed: number;
  uptime_seconds: number;
  /**
   * Throughput: frames completed per wall second. Deliberately not `1 / latency` —
   * two frames may be in flight, so inverting latency would report about half the
   * frames per second the server genuinely completes.
   */
  inference_fps: number | null;
  /** What one frame costs inside the pipeline, excluding any wait behind another. */
  processing_ms_mean: number | null;
  /** End-to-end, including that wait: what a viewer perceives as the delay. */
  latency_ms_mean: number | null;
  latency_ms_last: number | null;
}

export interface LiveResultMessage {
  type: 'result';
  frame_index: number;
  sequence: number;
  capture_seconds: number;
  tracks: LiveTrack[];
  motorcycles: LiveMotorcycle[];
  riders: LiveRider[];
  /** Base64 JPEG of the annotated frame, or null when there was nothing to draw. */
  annotated: string | null;
  window_rolled_over: boolean;
  stats: LiveStats;
}

export interface LiveEventsMessage {
  type: 'events';
  /** The very `ConfirmedEvent`s the rules minted — not a reduced live shape. */
  events: ConfirmedEvent[];
}

export interface LiveWarningMessage {
  type: 'warning';
  code: string;
  message: string;
}

export interface LiveErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

export interface LiveStoppedMessage {
  type: 'stopped';
  session_id: string;
  stats: LiveStats;
}

export type LiveServerMessage =
  | LiveSessionMessage
  | LiveResultMessage
  | LiveEventsMessage
  | LiveWarningMessage
  | LiveErrorMessage
  | LiveStoppedMessage;

/**
 * API contract types (H7B) mirroring the H7A FastAPI response models.
 *
 * These are the wire types the typed client returns. They are intentionally a
 * faithful mirror of `src/trafficpulse/app/models.py`; keeping them here (rather
 * than importing anything backend) is exactly the boundary H7A defines — the
 * frontend depends only on JSON shapes.
 */

export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

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
}

export interface EventSummary {
  event_id: string;
  video_id: string;
  job_id: string;
  violation_type: ViolationType;
  camera_id: string;
  track_ids: string[];
  trigger_at: string;
  rule_id: string;
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
  confidence: Record<string, unknown>;
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

import { vi } from 'vitest';

import {
  type AnalyticsSummary,
  type ConfirmedEvent,
  type EngineMetrics,
  type EventSummary,
  type EvidenceManifest,
  type JobStatusResponse,
  type ReviewCase,
  type ReviewEntry,
  type ReviewResponse,
  type ExpectationComparison,
  type ExpectationRecord,
  type SceneSummary,
  type StoredScene,
  type VideoSummary,
  type VideoUploadResponse,
  type ViolationType,
} from '@/api/types';
import { type ProcessingActions, type ProcessingController } from '@/hooks/use-processing';
import { type WorkspaceEvent, toWorkspaceEvent } from '@/lib/workspace';

/**
 * Wire-shaped fixtures for H7C tests.
 *
 * Media time is anchored at the Unix epoch (as the backend does), so
 * `mediaSeconds(12.5)` produces a `trigger_at` that maps to 12.5s on the video's
 * own timeline.
 */
export function mediaSeconds(seconds: number): string {
  return new Date(seconds * 1000).toISOString();
}

export function makeEventSummary(overrides: Partial<EventSummary> = {}): EventSummary {
  return {
    event_id: 'evt-1',
    video_id: 'vid-1',
    job_id: 'job-1',
    violation_type: 'wrong_way' as ViolationType,
    camera_id: 'cam-north',
    track_ids: ['t-1'],
    start_at: mediaSeconds(9),
    trigger_at: mediaSeconds(10),
    rule_id: 'wrong-way-v1',
    confidence: { classifier: 0.91, temporal_consistency: 0.8 },
    review_status: 'pending',
    ...overrides,
  };
}

export function makeWorkspaceEvent(overrides: Partial<EventSummary> = {}): WorkspaceEvent {
  return toWorkspaceEvent(makeEventSummary(overrides));
}

export function makeConfirmedEvent(overrides: Partial<ConfirmedEvent> = {}): ConfirmedEvent {
  return {
    event_id: 'evt-1',
    violation_type: 'wrong_way',
    camera_id: 'cam-north',
    track_ids: ['t-1'],
    start_at: mediaSeconds(9),
    trigger_at: mediaSeconds(10),
    end_at: mediaSeconds(11),
    rule_id: 'wrong-way-v1',
    rule_version: '1.0.0',
    scene_config_hash: 'scene-hash',
    code_version: 'abc1234',
    source_hypothesis_id: 'hyp-1',
    created_at: mediaSeconds(12),
    measurements: [{ name: 'heading_deviation_deg', value: 172, unit: 'deg' }],
    thresholds: [{ name: 'min_heading_deviation_deg', value: 150, unit: 'deg' }],
    models: [{ name: 'rtdetr', version: '1.0', weights_hash: null }],
    confidence: { classifier: 0.91, temporal_consistency: 0.8 },
    ...overrides,
  };
}

export function makeReviewEntry(overrides: Partial<ReviewEntry> = {}): ReviewEntry {
  return {
    entry_id: 'rev-1',
    event_id: 'evt-1',
    action: 'open',
    status_before: 'pending',
    status_after: 'in_review',
    reviewer: 'analyst',
    at: '2026-07-29T12:00:00Z',
    note: null,
    reason: null,
    ...overrides,
  };
}

export function makeReviewCase(overrides: Partial<ReviewCase> = {}): ReviewCase {
  return {
    review_case_id: 'case-evt-1',
    evidence_package_id: 'pkg-1',
    event_id: 'evt-1',
    status: 'pending',
    reviewer_id: null,
    decided_at: null,
    note: null,
    reason: null,
    updated_at: null,
    audit_ref: null,
    created_at: '1970-01-01T00:00:00Z',
    ...overrides,
  };
}

export function makeReview(overrides: Partial<ReviewResponse> = {}): ReviewResponse {
  return { case: makeReviewCase(), history: [], ...overrides };
}

export function makeEvidence(overrides: Partial<EvidenceManifest> = {}): EvidenceManifest {
  return {
    evidence_package_id: 'pkg-1',
    event_id: 'evt-1',
    before_frame: {
      kind: 'frame',
      locator: 'frames/evt-1-before.jpg',
      sha256: null,
      media_type: 'image/jpeg',
    },
    trigger_frame: null,
    after_frame: null,
    clip: null,
    trajectory: null,
    plate_crop: null,
    rule_trace: [
      {
        index: 0,
        label: 'Heading compared to legal direction',
        note: 'reversed',
        measurements: [],
      },
    ],
    models: [{ name: 'rtdetr', version: '1.0', weights_hash: null }],
    code_version: 'abc1234',
    scene_config_hash: 'scene-hash',
    created_at: mediaSeconds(12),
    ...overrides,
  };
}

/**
 * An H14 manifest: rendered frames, each carrying the content hash that makes it
 * fetchable. `makeEvidence` remains the pre-H14 shape (references without hashes),
 * so both eras stay covered.
 */
export function makeRenderedEvidence(
  overrides: Partial<EvidenceManifest> = {},
): EvidenceManifest {
  const frame = (kind: string, digest: string) => ({
    kind,
    locator: `artifacts/${digest.slice(0, 2)}/${digest}.png`,
    sha256: digest,
    media_type: 'image/png',
  });
  const before = frame('before_frame', 'a'.repeat(64));
  const trigger = frame('trigger_frame', 'b'.repeat(64));
  const after = frame('after_frame', 'c'.repeat(64));
  return makeEvidence({
    before_frame: before,
    trigger_frame: trigger,
    after_frame: after,
    additional_artifacts: [before, trigger, after],
    ...overrides,
  });
}

export function makeVideo(overrides: Partial<VideoUploadResponse> = {}): VideoUploadResponse {
  return {
    video_id: 'vid-1',
    filename: 'junction.mp4',
    status: 'uploaded',
    size_bytes: 1024 * 1024,
    width: 1920,
    height: 1080,
    fps: 25,
    frame_count: 750,
    duration_seconds: 30,
    codec: 'h264',
    ...overrides,
  };
}

/** A calibrated scene summary (H12/H13). */
export function makeSceneSummary(overrides: Partial<SceneSummary> = {}): SceneSummary {
  return {
    scene_hash: 'a'.repeat(64),
    scene_id: 'scene-vid-1',
    scene_name: 'Scene for vid-1',
    camera_id: 'cam-vid-1',
    site_id: 'site-default',
    status: 'draft',
    calibration_status: 'absent',
    frame_width: 320,
    frame_height: 240,
    zone_count: 1,
    has_legal_direction: false,
    has_no_stopping_zone: false,
    // Analyst-drawn by default; a derived scene is the exception a test opts into.
    derived: false,
    supported_violations: ['no_helmet', 'triple_riding'],
    ...overrides,
  };
}

/**
 * A stored scene revision, using the ids `lib/calibration` authors.
 *
 * Shaped so `sceneToShapes` can round-trip it: a saved calibration that cannot be
 * read back onto the drawing surface is the bug this fixture exists to catch.
 */
export function makeStoredScene(overrides: Partial<StoredScene> = {}): StoredScene {
  return {
    scene: {
      scene_id: 'scene-vid-1',
      scene_name: 'Scene for vid-1',
      description: 'Controlled demonstration intersection.',
      status: 'draft',
      camera_id: 'cam-vid-1',
    },
    frame: { reference_width: 320, reference_height: 240 },
    zones: [
      {
        zone_id: 'zone-lane',
        zone_type: 'lane',
        enabled: true,
        polygon: [
          [10, 10],
          [300, 10],
          [300, 200],
        ],
      },
      {
        zone_id: 'zone-no-stopping',
        zone_type: 'no_stopping',
        enabled: true,
        polygon: [
          [20, 120],
          [120, 120],
          [120, 200],
        ],
      },
    ],
    stop_lines: [],
    legal_directions: [
      {
        direction_id: 'dir-legal',
        description: 'Legal travel direction',
        vector: { dx: 1, dy: 0 },
        zone_ids: ['zone-lane'],
      },
    ],
    signal_groups: [],
    rule_parameters: [
      {
        violation_type: 'illegal_stopping',
        parameters: [
          {
            id: 'stationary_duration',
            value: 7,
            unit: 'seconds',
            status: 'provisional',
            note: null,
          },
        ],
      },
    ],
    calibration: { source: 'analyst_calibration', type: 'none', status: 'absent' },
    ...overrides,
  };
}

/** A declared controlled-demo expectation. Ground truth, never a detection. */
export function makeExpectation(
  overrides: Partial<ExpectationRecord> = {},
): ExpectationRecord {
  return {
    video_id: 'vid-1',
    expected_violations: ['wrong_way', 'triple_riding'],
    notes: 'Controlled demo clip; the no-stopping zone is artificially designated.',
    declared_by: 'analyst',
    declared_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

/** An expected-vs-detected comparison; defaults to one matched and one missing. */
export function makeComparison(
  overrides: Partial<ExpectationComparison> = {},
): ExpectationComparison {
  return {
    video_id: 'vid-1',
    job_id: 'job-1',
    expectation: makeExpectation(),
    rows: [
      {
        violation_type: 'triple_riding',
        expected: true,
        detected_count: 1,
        event_ids: ['evt-1'],
        outcome: 'matched',
      },
      {
        violation_type: 'wrong_way',
        expected: true,
        detected_count: 0,
        event_ids: [],
        outcome: 'missing',
      },
    ],
    expected_count: 2,
    detected_event_count: 1,
    matched_count: 1,
    missing_count: 1,
    unexpected_count: 0,
    ...overrides,
  };
}

/** A historical-library row (H11); defaults to an analysed video with events. */
export function makeVideoSummary(overrides: Partial<VideoSummary> = {}): VideoSummary {
  return {
    video_id: 'vid-1',
    filename: 'junction.mp4',
    uploaded_at: '2026-07-30T09:00:00Z',
    size_bytes: 1024 * 1024,
    width: 1920,
    height: 1080,
    fps: 25,
    duration_seconds: 30,
    codec: 'h264',
    job_id: 'job-1',
    status: 'succeeded',
    job_count: 1,
    event_count: 2,
    events_reviewed: 0,
    overlay_available: false,
    media_available: true,
    ...overrides,
  };
}

export function makeJob(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  return {
    job_id: 'job-1',
    video_id: 'vid-1',
    status: 'running',
    progress: 0.5,
    frames_processed: 375,
    frames_total: 750,
    fps: 12.5,
    estimated_remaining_seconds: 30,
    event_count: 0,
    error: null,
    overlay_available: false,
    overlay_status: 'none',
    ...overrides,
  };
}

/** A `File` that reports a size (jsdom's Blob size is derived from its parts). */
export function makeEngineMetrics(overrides: Partial<EngineMetrics> = {}): EngineMetrics {
  return {
    frames_read: 300,
    frames_skipped_stride: 0,
    frames_skipped_fps: 0,
    frames_dropped_backpressure: 0,
    frames_admitted: 300,
    frames_processed: 300,
    batches_processed: 10,
    detections: 120,
    track_states: 260,
    events_confirmed: 3,
    queue_peak: 4,
    media_fps: 25,
    wall_fps: 40,
    latencies: {},
    memory_bytes_current: null,
    memory_bytes_peak: null,
    gpu_memory_bytes_current: null,
    gpu_memory_bytes_peak: null,
    ...overrides,
  } as EngineMetrics;
}

/**
 * A populated analytics summary (H15).
 *
 * `makeEmptyAnalytics` is its counterpart for the empty-repository path, which is
 * the state a dashboard most often breaks on.
 */
export function makeAnalytics(overrides: Partial<AnalyticsSummary> = {}): AnalyticsSummary {
  return {
    repository: {
      videos_total: 4,
      videos_processed: 3,
      videos_unprocessed: 1,
      videos_calibrated: 2,
      footage_seconds: 620,
      storage_bytes: 5_242_880,
    },
    processing: {
      jobs_total: 6,
      jobs_pending: 0,
      jobs_running: 1,
      jobs_succeeded: 4,
      jobs_failed: 1,
      jobs_cancelled: 0,
      average_duration_seconds: 12.5,
      timed_jobs: 4,
      frames_processed: 1200,
    },
    violations: {
      events_total: 9,
      by_type: [
        { violation_type: 'no_helmet', count: 5 },
        { violation_type: 'wrong_way', count: 3 },
        { violation_type: 'red_light_jumping', count: 1 },
      ],
      counted_jobs: 4,
      uncounted_jobs: 0,
    },
    evidence: {
      events_total: 9,
      events_with_artifacts: 6,
      artifacts_total: 18,
      artifact_bytes: 262_144,
      overlays_available: 3,
    },
    review: { events_total: 9, events_reviewed: 3, events_pending: 6 },
    health: {
      engine: 'ready',
      version: '1.0.0',
      failed_jobs: 1,
      videos_missing_media: 0,
      videos_uncalibrated: 2,
      runs_without_timing: 0,
    },
    recent_activity: [
      {
        kind: 'run',
        at: '2026-08-03T12:05:00Z',
        subject_id: 'job-1',
        summary: 'Run succeeded with 3 violation(s)',
        status: 'succeeded',
      },
      {
        kind: 'upload',
        at: '2026-08-03T12:00:00Z',
        subject_id: 'vid-1',
        summary: 'Uploaded clip.mp4',
        status: null,
      },
    ],
    latest_run: makeEngineMetrics(),
    ...overrides,
  };
}

export function makeEmptyAnalytics(): AnalyticsSummary {
  return makeAnalytics({
    repository: {
      videos_total: 0,
      videos_processed: 0,
      videos_unprocessed: 0,
      videos_calibrated: 0,
      footage_seconds: null,
      storage_bytes: 0,
    },
    processing: {
      jobs_total: 0,
      jobs_pending: 0,
      jobs_running: 0,
      jobs_succeeded: 0,
      jobs_failed: 0,
      jobs_cancelled: 0,
      average_duration_seconds: null,
      timed_jobs: 0,
      frames_processed: 0,
    },
    violations: { events_total: 0, by_type: [], counted_jobs: 0, uncounted_jobs: 0 },
    evidence: {
      events_total: 0,
      events_with_artifacts: 0,
      artifacts_total: 0,
      artifact_bytes: 0,
      overlays_available: 0,
    },
    review: { events_total: 0, events_reviewed: 0, events_pending: 0 },
    health: {
      engine: 'ready',
      version: '1.0.0',
      failed_jobs: 0,
      videos_missing_media: 0,
      videos_uncalibrated: 0,
      runs_without_timing: 0,
    },
    recent_activity: [],
    latest_run: null,
  });
}

export function makeFile(name: string, sizeBytes = 1024): File {
  return new File([new Uint8Array(sizeBytes)], name, { type: 'video/mp4' });
}

/** A fully-populated {@link ProcessingController} for view tests (H7D). */
export function makeProcessingController(
  overrides: Partial<ProcessingController> = {},
): ProcessingController {
  const actions: ProcessingActions = {
    selectAndUpload: vi.fn(),
    openVideo: vi.fn(),
    startProcessing: vi.fn(),
    reprocessWith: vi.fn(),
    cancel: vi.fn(),
    cancelUpload: vi.fn(),
    retry: vi.fn(),
    remove: vi.fn(),
    replace: vi.fn(),
    reconnect: vi.fn(),
    ...overrides.actions,
  };
  return {
    phase: 'running',
    job: makeJob(),
    // The run the workspace is scoped to (R7); matches makeJob()'s id so a fixture
    // controller and its job describe the same run.
    jobId: 'job-1',
    video: makeVideo(),
    progressRatio: 0.5,
    elapsedSeconds: 10,
    etaSeconds: 20,
    logs: [],
    error: null,
    isBusy: true,
    isCancelling: false,
    connectionError: null,
    ...overrides,
    actions,
  };
}

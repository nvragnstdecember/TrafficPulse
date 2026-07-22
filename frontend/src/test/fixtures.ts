import { vi } from 'vitest';

import {
  type ConfirmedEvent,
  type EventSummary,
  type EvidenceManifest,
  type JobStatusResponse,
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
    trigger_at: mediaSeconds(10),
    rule_id: 'wrong-way-v1',
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
    confidence: { overall: 0.91 },
    ...overrides,
  };
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
    ...overrides,
  };
}

/** A `File` that reports a size (jsdom's Blob size is derived from its parts). */
export function makeFile(name: string, sizeBytes = 1024): File {
  return new File([new Uint8Array(sizeBytes)], name, { type: 'video/mp4' });
}

/** A fully-populated {@link ProcessingController} for view tests (H7D). */
export function makeProcessingController(
  overrides: Partial<ProcessingController> = {},
): ProcessingController {
  const actions: ProcessingActions = {
    selectAndUpload: vi.fn(),
    startProcessing: vi.fn(),
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

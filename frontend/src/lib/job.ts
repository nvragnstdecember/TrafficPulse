import { type JobStatus, type JobStatusResponse } from '@/api/types';

import { type ViolationTone } from './workspace';

/**
 * Job lifecycle model (H7C).
 *
 * The workspace surfaces a slightly richer lifecycle than the backend's four job
 * statuses: it adds `idle` (nothing submitted) and `uploading` (client-side,
 * before a job exists). These pure mappers keep phase/label/tone/progress logic
 * out of components and fully testable.
 *
 * ```text
 * idle → uploading → queued → running → completed
 *                                    ↘ failed
 * ```
 */
export type ProcessingPhase = 'idle' | 'uploading' | 'queued' | 'running' | 'completed' | 'failed';

export function jobStatusToPhase(status: JobStatus): ProcessingPhase {
  switch (status) {
    case 'pending':
      return 'queued';
    case 'running':
      return 'running';
    case 'succeeded':
      return 'completed';
    case 'failed':
      return 'failed';
  }
}

export function isTerminalPhase(phase: ProcessingPhase): boolean {
  return phase === 'completed' || phase === 'failed';
}

export function isActivePhase(phase: ProcessingPhase): boolean {
  return phase === 'uploading' || phase === 'queued' || phase === 'running';
}

const PHASE_LABELS: Record<ProcessingPhase, string> = {
  idle: 'Idle',
  uploading: 'Uploading',
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
};

const PHASE_TONES: Record<ProcessingPhase, ViolationTone> = {
  idle: 'neutral',
  uploading: 'info',
  queued: 'info',
  running: 'info',
  completed: 'success',
  failed: 'error',
};

export function phaseLabel(phase: ProcessingPhase): string {
  return PHASE_LABELS[phase];
}

export function phaseTone(phase: ProcessingPhase): ViolationTone {
  return PHASE_TONES[phase];
}

/** Progress as a 0..1 ratio: prefer the server's `progress`, else frames. */
export function jobProgressRatio(job: JobStatusResponse | undefined | null): number | null {
  if (!job) return null;
  if (typeof job.progress === 'number' && Number.isFinite(job.progress)) {
    return Math.min(1, Math.max(0, job.progress));
  }
  if (job.frames_total && job.frames_total > 0) {
    return Math.min(1, Math.max(0, job.frames_processed / job.frames_total));
  }
  return null;
}

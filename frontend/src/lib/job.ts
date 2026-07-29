import { type JobStatus, type JobStatusResponse } from '@/api/types';

import { type ViolationTone } from './workspace';

/**
 * Job lifecycle model (H7C, extended in H7D).
 *
 * The workspace surfaces a richer lifecycle than the backend's job statuses. It
 * adds two client-side phases — `idle` (nothing submitted) and `uploading`
 * (before a job exists) — and derives two finer sub-phases of the backend's
 * `running` state from truthful frame signals: `initializing` (running, but no
 * frame processed yet — the engine is opening the source and building) and
 * `finalizing` (every known frame processed, but not yet flipped to succeeded —
 * evidence build + persistence). `cancelled` mirrors the backend's terminal
 * cancelled state. These pure mappers keep phase/label/tone/progress logic out of
 * components and fully testable.
 *
 * ```text
 * idle → uploading → queued → initializing → running → finalizing → completed
 *                                                                 ↘ failed
 *                                                                 ↘ cancelled
 * ```
 */
export type ProcessingPhase =
  | 'idle'
  | 'uploading'
  | 'queued'
  | 'initializing'
  | 'running'
  | 'finalizing'
  | 'completed'
  | 'failed'
  | 'cancelled';

/** Status-only mapping (no frame context); coarse but stable. */
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
    case 'cancelled':
      return 'cancelled';
  }
}

/**
 * Derive the fine-grained phase from a live job status (H7D).
 *
 * Sub-phases of `running` are read from the job's own truthful counters — never
 * fabricated: no frames processed yet ⇒ `initializing`; all known frames
 * processed but still `running` ⇒ `finalizing`; otherwise `running`.
 */
export function derivePhase(job: JobStatusResponse | undefined | null): ProcessingPhase {
  if (!job) return 'idle';
  if (job.status !== 'running') return jobStatusToPhase(job.status);
  if (job.frames_processed <= 0) return 'initializing';
  if (job.frames_total && job.frames_total > 0 && job.frames_processed >= job.frames_total) {
    return 'finalizing';
  }
  return 'running';
}

/**
 * Whether the client should keep polling this job.
 *
 * Not simply "the job is not terminal". The annotated (overlay) video is rendered
 * *after* inference finishes, so a job reaches `succeeded` while its overlay is
 * still `pending`. Stopping at the terminal job status is exactly what stranded the
 * workspace on the raw upload: the render completed after the last poll and nothing
 * ever asked again. Polling therefore continues until **both** axes have settled.
 */
export function shouldPollJob(job: JobStatusResponse | undefined | null): boolean {
  if (!job) return false;
  if (job.status === 'pending' || job.status === 'running') return true;
  return job.overlay_status === 'pending';
}

/** Whether an annotated video is still being rendered for a finished job. */
export function isOverlayRendering(job: JobStatusResponse | undefined | null): boolean {
  return job?.status === 'succeeded' && job.overlay_status === 'pending';
}

export function isTerminalPhase(phase: ProcessingPhase): boolean {
  return phase === 'completed' || phase === 'failed' || phase === 'cancelled';
}

export function isActivePhase(phase: ProcessingPhase): boolean {
  return (
    phase === 'uploading' ||
    phase === 'queued' ||
    phase === 'initializing' ||
    phase === 'running' ||
    phase === 'finalizing'
  );
}

/**
 * Whether a *server-side* job in this phase can be cancelled via the API. The
 * client-side `uploading` phase is aborted locally instead (no job exists yet).
 */
export function isCancellablePhase(phase: ProcessingPhase): boolean {
  return (
    phase === 'queued' || phase === 'initializing' || phase === 'running' || phase === 'finalizing'
  );
}

const PHASE_LABELS: Record<ProcessingPhase, string> = {
  idle: 'Idle',
  uploading: 'Uploading',
  queued: 'Queued',
  initializing: 'Initializing',
  running: 'Running',
  finalizing: 'Finalizing',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const PHASE_TONES: Record<ProcessingPhase, ViolationTone> = {
  idle: 'neutral',
  uploading: 'info',
  queued: 'info',
  initializing: 'info',
  running: 'info',
  finalizing: 'info',
  completed: 'success',
  failed: 'error',
  cancelled: 'neutral',
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

// --- review workflow stages -----------------------------------------------------
export type WorkflowStage = 'upload' | 'processing' | 'review' | 'evidence';
export type StageState = 'done' | 'current' | 'todo';

/**
 * Where the analyst is in upload → processing → review → evidence.
 *
 * Derived entirely from the processing phase and whether an event is selected, so
 * the stepper can never disagree with the workspace it labels. A stepper that owns
 * its own "current stage" always eventually drifts out of sync with the state it
 * is describing; this one has nothing to drift.
 */
export function stageStates(
  phase: ProcessingPhase,
  hasSelection: boolean,
): Record<WorkflowStage, StageState> {
  if (phase === 'idle' || phase === 'uploading') {
    return { upload: 'current', processing: 'todo', review: 'todo', evidence: 'todo' };
  }
  if (phase !== 'completed') {
    // Includes failed/cancelled: the run is where the story stopped.
    return { upload: 'done', processing: 'current', review: 'todo', evidence: 'todo' };
  }
  return {
    upload: 'done',
    processing: 'done',
    review: hasSelection ? 'done' : 'current',
    evidence: hasSelection ? 'current' : 'todo',
  };
}

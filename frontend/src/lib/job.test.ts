import { describe, expect, it } from 'vitest';

import { makeJob } from '@/test/fixtures';

import {
  derivePhase,
  isActivePhase,
  isCancellablePhase,
  isTerminalPhase,
  jobProgressRatio,
  jobStatusToPhase,
  phaseLabel,
  phaseTone,
} from './job';

describe('jobStatusToPhase', () => {
  it('maps every backend status onto a workspace phase', () => {
    expect(jobStatusToPhase('pending')).toBe('queued');
    expect(jobStatusToPhase('running')).toBe('running');
    expect(jobStatusToPhase('succeeded')).toBe('completed');
    expect(jobStatusToPhase('failed')).toBe('failed');
    expect(jobStatusToPhase('cancelled')).toBe('cancelled');
  });
});

describe('derivePhase (H7D)', () => {
  it('returns idle without a job', () => {
    expect(derivePhase(undefined)).toBe('idle');
    expect(derivePhase(null)).toBe('idle');
  });

  it('maps terminal statuses directly', () => {
    expect(derivePhase(makeJob({ status: 'succeeded' }))).toBe('completed');
    expect(derivePhase(makeJob({ status: 'failed' }))).toBe('failed');
    expect(derivePhase(makeJob({ status: 'cancelled' }))).toBe('cancelled');
    expect(derivePhase(makeJob({ status: 'pending' }))).toBe('queued');
  });

  it('derives running sub-phases from the frame counters', () => {
    expect(derivePhase(makeJob({ status: 'running', frames_processed: 0 }))).toBe('initializing');
    expect(
      derivePhase(makeJob({ status: 'running', frames_processed: 40, frames_total: 100 })),
    ).toBe('running');
    expect(
      derivePhase(makeJob({ status: 'running', frames_processed: 100, frames_total: 100 })),
    ).toBe('finalizing');
  });

  it('stays running when the total is unknown', () => {
    expect(
      derivePhase(makeJob({ status: 'running', frames_processed: 5, frames_total: null })),
    ).toBe('running');
  });
});

describe('phase predicates', () => {
  it('classifies active and terminal phases', () => {
    expect(isActivePhase('uploading')).toBe(true);
    expect(isActivePhase('queued')).toBe(true);
    expect(isActivePhase('initializing')).toBe(true);
    expect(isActivePhase('running')).toBe(true);
    expect(isActivePhase('finalizing')).toBe(true);
    expect(isActivePhase('completed')).toBe(false);
    expect(isActivePhase('cancelled')).toBe(false);
    expect(isTerminalPhase('completed')).toBe(true);
    expect(isTerminalPhase('failed')).toBe(true);
    expect(isTerminalPhase('cancelled')).toBe(true);
    expect(isTerminalPhase('idle')).toBe(false);
  });

  it('classifies which phases can be cancelled server-side', () => {
    expect(isCancellablePhase('queued')).toBe(true);
    expect(isCancellablePhase('initializing')).toBe(true);
    expect(isCancellablePhase('running')).toBe(true);
    expect(isCancellablePhase('finalizing')).toBe(true);
    // Uploading is aborted locally, not cancelled via the API.
    expect(isCancellablePhase('uploading')).toBe(false);
    expect(isCancellablePhase('completed')).toBe(false);
  });

  it('labels and tones every phase', () => {
    expect(phaseLabel('running')).toBe('Running');
    expect(phaseLabel('initializing')).toBe('Initializing');
    expect(phaseLabel('finalizing')).toBe('Finalizing');
    expect(phaseLabel('cancelled')).toBe('Cancelled');
    expect(phaseTone('completed')).toBe('success');
    expect(phaseTone('failed')).toBe('error');
    expect(phaseTone('idle')).toBe('neutral');
    expect(phaseTone('cancelled')).toBe('neutral');
    expect(phaseTone('initializing')).toBe('info');
  });
});

describe('jobProgressRatio', () => {
  it('prefers the reported progress, clamped to 0..1', () => {
    expect(jobProgressRatio(makeJob({ progress: 0.25 }))).toBe(0.25);
    expect(jobProgressRatio(makeJob({ progress: 1.4 }))).toBe(1);
    expect(jobProgressRatio(makeJob({ progress: -1 }))).toBe(0);
  });

  it('falls back to the frame counts', () => {
    expect(
      jobProgressRatio(makeJob({ progress: null, frames_processed: 30, frames_total: 120 })),
    ).toBe(0.25);
  });

  it('returns null when progress is unknowable', () => {
    expect(jobProgressRatio(undefined)).toBeNull();
    expect(jobProgressRatio(makeJob({ progress: null, frames_total: null }))).toBeNull();
  });
});

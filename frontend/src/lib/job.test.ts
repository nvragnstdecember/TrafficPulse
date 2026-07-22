import { describe, expect, it } from 'vitest';

import { makeJob } from '@/test/fixtures';

import {
  isActivePhase,
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
  });
});

describe('phase predicates', () => {
  it('classifies active and terminal phases', () => {
    expect(isActivePhase('uploading')).toBe(true);
    expect(isActivePhase('queued')).toBe(true);
    expect(isActivePhase('running')).toBe(true);
    expect(isActivePhase('completed')).toBe(false);
    expect(isTerminalPhase('completed')).toBe(true);
    expect(isTerminalPhase('failed')).toBe(true);
    expect(isTerminalPhase('idle')).toBe(false);
  });

  it('labels and tones every phase', () => {
    expect(phaseLabel('running')).toBe('Running');
    expect(phaseTone('completed')).toBe('success');
    expect(phaseTone('failed')).toBe('error');
    expect(phaseTone('idle')).toBe('neutral');
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

import { describe, expect, it } from 'vitest';

import { makeAnalytics, makeEmptyAnalytics } from '@/test/fixtures';

import {
  activityKindLabel,
  completionFraction,
  evidenceStats,
  hasPartialBreakdown,
  healthStats,
  isEmptyRepository,
  processingStats,
  repositoryStats,
  violationBars,
} from './analytics';

function statValue(stats: ReturnType<typeof repositoryStats>, key: string) {
  return stats.find((stat) => stat.key === key)?.value;
}

describe('violationBars (H15)', () => {
  it('normalises against the largest count, widest first', () => {
    const bars = violationBars(makeAnalytics().violations.by_type);
    expect(bars.map((bar) => bar.key)).toEqual([
      'no_helmet',
      'wrong_way',
      'red_light_jumping',
    ]);
    expect(bars[0].fraction).toBe(1); // the largest is the reference, not the total
    expect(bars[1].fraction).toBeCloseTo(3 / 5);
  });

  it('labels each row with the violation display name', () => {
    const [first] = violationBars([{ violation_type: 'no_helmet', count: 2 }]);
    expect(first.label).toBe('No helmet');
  });

  it('returns nothing for an empty or absent breakdown', () => {
    expect(violationBars([])).toEqual([]);
    expect(violationBars(undefined)).toEqual([]);
    expect(violationBars(null)).toEqual([]);
  });

  it('does not divide by zero when every count is zero', () => {
    const bars = violationBars([{ violation_type: 'no_helmet', count: 0 }]);
    expect(bars[0].fraction).toBe(0);
  });
});

describe('completionFraction', () => {
  it('is a clamped ratio', () => {
    expect(completionFraction(3, 12)).toBe(0.25);
    expect(completionFraction(20, 10)).toBe(1);
  });

  it('is null when there is nothing to complete', () => {
    // Not 0 — "no events to review" is not "0% reviewed".
    expect(completionFraction(0, 0)).toBeNull();
  });
});

describe('stat mapping', () => {
  it('maps a populated repository', () => {
    const stats = repositoryStats(makeAnalytics());
    expect(statValue(stats, 'videos')).toBe('4');
    expect(statValue(stats, 'violations')).toBe('9');
    expect(statValue(stats, 'reviewed')).toBe('33%');
    expect(statValue(stats, 'footage')).toBe('10m 20s');
  });

  it('reports an unmeasured value as null, never as zero', () => {
    const empty = makeEmptyAnalytics();
    // Footage: no video declares a duration.
    expect(statValue(repositoryStats(empty), 'footage')).toBeNull();
    // Review completion: nothing to review.
    expect(statValue(repositoryStats(empty), 'reviewed')).toBeNull();
    // Average run: no run recorded timing.
    expect(statValue(processingStats(empty), 'duration')).toBeNull();
    // Evidence coverage: no events.
    expect(statValue(evidenceStats(empty), 'coverage')).toBeNull();
  });

  it('still reports genuine zero counts as zero', () => {
    const stats = repositoryStats(makeEmptyAnalytics());
    expect(statValue(stats, 'videos')).toBe('0');
    expect(statValue(stats, 'violations')).toBe('0');
  });

  it('maps processing outcomes', () => {
    const stats = processingStats(makeAnalytics());
    expect(statValue(stats, 'succeeded')).toBe('4');
    expect(statValue(stats, 'failed')).toBe('1');
    expect(statValue(stats, 'duration')).toBe('12.5s');
  });

  it('maps evidence coverage', () => {
    const stats = evidenceStats(makeAnalytics());
    expect(statValue(stats, 'coverage')).toBe('67%');
    expect(statValue(stats, 'artifacts')).toBe('18');
  });

  it('maps health signals', () => {
    const stats = healthStats(makeAnalytics());
    expect(statValue(stats, 'engine')).toBe('ready');
    expect(statValue(stats, 'failed')).toBe('1');
    expect(statValue(stats, 'calibrated')).toBe('2/4');
  });
});

describe('summary predicates', () => {
  it('detects an empty repository', () => {
    expect(isEmptyRepository(makeEmptyAnalytics())).toBe(true);
    expect(isEmptyRepository(makeAnalytics())).toBe(false);
  });

  it('flags a breakdown the backend declared incomplete', () => {
    expect(hasPartialBreakdown(makeAnalytics())).toBe(false);
    const partial = makeAnalytics({
      violations: {
        events_total: 9,
        by_type: [{ violation_type: 'no_helmet', count: 5 }],
        counted_jobs: 1,
        uncounted_jobs: 2,
      },
    });
    expect(hasPartialBreakdown(partial)).toBe(true);
  });

  it('labels activity kinds', () => {
    expect(activityKindLabel('upload')).toBe('Upload');
    expect(activityKindLabel('run')).toBe('Run');
    expect(activityKindLabel('mystery')).toBe('mystery');
  });
});

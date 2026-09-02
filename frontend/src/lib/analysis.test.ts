import { describe, expect, it } from 'vitest';

import {
  type HelmetAnalysis,
  type PostureState,
  type RiderAnalysis,
  type SystemPosture,
} from '@/api/types';
import {
  analysisStats,
  enforcementNote,
  enforcementTone,
  flipRate,
  helmetTone,
  percent,
  postureTone,
  sortRiders,
} from '@/lib/analysis';

function rider(overrides: Partial<RiderAnalysis> = {}): RiderAnalysis {
  return {
    rider_track_id: 'r-1',
    motorcycle_track_id: 'm-1',
    rider_count: 1,
    multi_rider: false,
    samples: 10,
    first_frame: 0,
    last_frame: 9,
    helmet_state: 'helmet',
    confidence: 0.91,
    agreement: 1,
    settled: true,
    raw_label_flips: 0,
    stabilized_label_flips: 0,
    median_head_height_px: 42,
    enforcement: 'eligible',
    ...overrides,
  };
}

function analysis(overrides: Partial<HelmetAnalysis> = {}): HelmetAnalysis {
  return {
    job_id: 'job-1',
    enforcement: 'disabled',
    frames_observed: 30,
    riders_observed: 2,
    motorcycles_associated: 1,
    multi_rider_riders: 1,
    multi_rider_motorcycles: 1,
    eligible_riders: 1,
    unresolved_riders: 1,
    abstained_riders: 0,
    unstable_riders: 0,
    gate_abstentions: 3,
    label_counts: [
      { label: 'helmet', riders: 1 },
      { label: 'no_helmet', riders: 1 },
    ],
    enforcement_counts: [
      { label: 'eligible', riders: 1 },
      { label: 'multi_rider_unresolved', riders: 1 },
    ],
    riders: [rider(), rider({ rider_track_id: 'r-2', helmet_state: 'no_helmet' })],
    ...overrides,
  };
}

describe('posture presentation', () => {
  it('keeps experimental visually distinct from limited', () => {
    // The load-bearing distinction: "works within a boundary" must not look like
    // "runs but is not safe to act on".
    expect(postureTone('limited')).not.toBe(postureTone('experimental'));
    expect(postureTone('experimental')).toBe('warning');
  });

  it('does not dress a disabled capability as a failure', () => {
    // Enforcement being off is the guard working, not the system breaking.
    expect(postureTone('disabled')).toBe('neutral');
  });
});

describe('helmet label presentation', () => {
  it('never renders a no-helmet reading in the confirmed-violation register', () => {
    // Red means "a violation was confirmed" in this design system. An analysis
    // confirms nothing, so no reading may claim that colour.
    expect(helmetTone('no_helmet')).toBe('warning');
    expect(helmetTone('no_helmet')).not.toBe('error');
  });

  it('treats an unmapped backend label as neutral rather than guessing', () => {
    expect(helmetTone('balaclava')).toBe('neutral');
  });
});

describe('percent', () => {
  it('preserves an unmeasured value as null rather than zero', () => {
    // The whole point of the null-vs-0.0 convention: "not measured" must never be
    // presented as "measured, and it was nothing".
    expect(percent(null)).toBeNull();
    expect(percent(0)).toBe('0%');
  });
});

describe('enforcement presentation', () => {
  it('marks an unresolved rider as a warning, not a success', () => {
    expect(enforcementTone('multi_rider_unresolved')).toBe('warning');
    expect(enforcementTone('eligible')).toBe('success');
  });
});

describe('analysisStats', () => {
  it('counts motorcycles that carried riders, not motorcycles detected', () => {
    const stats = analysisStats(analysis());
    const bikes = stats.find((stat) => stat.key === 'motorcycles');
    expect(bikes?.label).toBe('Motorcycles with riders');
    expect(bikes?.value).toBe('1');
  });

  it('reports no-helmet readings without calling them violations', () => {
    const stats = analysisStats(analysis());
    const noHelmet = stats.find((stat) => stat.key === 'no-helmet');
    expect(noHelmet?.value).toBe('1');
    expect(noHelmet?.hint).toContain('No violation is confirmed');
    expect(stats.map((stat) => stat.label).join(' ')).not.toMatch(/violation/i);
  });

  it('folds unsettled riders into the abstained figure', () => {
    const stats = analysisStats(analysis({ abstained_riders: 2, unstable_riders: 3 }));
    expect(stats.find((stat) => stat.key === 'abstained')?.value).toBe('5');
  });
});

describe('sortRiders', () => {
  it('surfaces unresolved riders first, then no-helmet readings', () => {
    const sorted = sortRiders([
      rider({ rider_track_id: 'c', helmet_state: 'helmet' }),
      rider({ rider_track_id: 'b', helmet_state: 'no_helmet' }),
      rider({
        rider_track_id: 'a',
        multi_rider: true,
        rider_count: 2,
        enforcement: 'multi_rider_unresolved',
      }),
    ]);
    expect(sorted.map((entry) => entry.rider_track_id)).toEqual(['a', 'b', 'c']);
  });

  it('is stable and does not mutate its input', () => {
    const input = [rider({ rider_track_id: 'z' }), rider({ rider_track_id: 'y' })];
    const first = sortRiders(input).map((entry) => entry.rider_track_id);
    expect(sortRiders(input).map((entry) => entry.rider_track_id)).toEqual(first);
    expect(input.map((entry) => entry.rider_track_id)).toEqual(['z', 'y']);
  });
});

describe('flipRate', () => {
  it('reports the P4-U10 shape: tracks that flipped, out of tracks seen', () => {
    expect(
      flipRate([rider({ raw_label_flips: 3 }), rider({ raw_label_flips: 0 })]),
    ).toEqual({ flipped: 1, total: 2 });
  });

  it('reports null rather than a reassuring 0% when nothing was observed', () => {
    expect(flipRate([])).toBeNull();
  });
});

describe('enforcementNote', () => {
  const posture = (state: PostureState): SystemPosture => ({
    components: [],
    helmet_backend: 'ResNetHelmetConfig',
    helmet_backend_labels: ['helmet', 'no_helmet'],
    turban_capable: false,
    helmet_enforcement: state,
  });

  it('qualifies an empty event list when the helmet rule was never allowed to run', () => {
    // The single most consequential wrong sentence this UI could show: an empty list
    // reading as "we looked and found nothing" when the rule was switched off.
    const note = enforcementNote(posture('disabled'));
    expect(note).toContain('Helmet violation enforcement is off');
    expect(note).toContain('Other violation families ran normally');
  });

  it('also qualifies it when no backend makes the rule available at all', () => {
    expect(enforcementNote(posture('unavailable'))).not.toBeNull();
  });

  it('adds nothing when the rule could actually run', () => {
    // Where the wording is already honest it must be left exactly as it was.
    expect(enforcementNote(posture('experimental'))).toBeNull();
    expect(enforcementNote(posture('active'))).toBeNull();
  });

  it('adds nothing before the posture has loaded', () => {
    expect(enforcementNote(undefined)).toBeNull();
  });
});

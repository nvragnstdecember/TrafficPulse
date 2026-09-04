import { describe, expect, it } from 'vitest';

import { makeComparison, makeExpectation } from '@/test/fixtures';

import {
  DEMO_SELECTABLE_VIOLATIONS,
  OUTCOME_LABEL,
  OUTCOME_MEANING,
  OUTCOME_TONE,
  comparisonSummary,
  isAsDeclared,
} from './demo';

describe('controlled-demo presentation', () => {
  it('reports counts and never derives an accuracy from them', () => {
    const summary = comparisonSummary(makeComparison());

    expect(summary).toContain('1 of 2 declared families matched');
    expect(summary).toContain('1 event confirmed');
    expect(summary).toContain('1 not detected');
    // "50%" over a sample of two, against ground truth the same person wrote, is the
    // exact number this project refuses to state.
    expect(summary).not.toMatch(/%|precision|recall|accuracy/i);
  });

  it('describes a clip nobody declared anything about', () => {
    expect(comparisonSummary(makeComparison({ expectation: null, detected_event_count: 0 })))
      .toBe('Nothing declared for this clip, and nothing confirmed.');
    expect(
      comparisonSummary(
        makeComparison({ expectation: null, detected_event_count: 3, unexpected_count: 2 }),
      ),
    ).toBe('Nothing declared for this clip; 3 events confirmed.');
  });

  it('has no comparison to describe before one exists', () => {
    expect(comparisonSummary(null)).toBe('No comparison yet.');
    expect(comparisonSummary(undefined)).toBe('No comparison yet.');
  });

  it('singularises one event', () => {
    const summary = comparisonSummary(
      makeComparison({ detected_event_count: 1, missing_count: 0, matched_count: 2, rows: [] }),
    );
    expect(summary).toContain('1 event confirmed');
    expect(summary).not.toContain('1 events');
  });

  it('treats a declaration as met only when nothing is missing or unexpected', () => {
    expect(isAsDeclared(makeComparison({ missing_count: 0, unexpected_count: 0 }))).toBe(true);
    expect(isAsDeclared(makeComparison({ missing_count: 1, unexpected_count: 0 }))).toBe(false);
    expect(isAsDeclared(makeComparison({ missing_count: 0, unexpected_count: 1 }))).toBe(false);
  });

  it('never calls an undeclared clip "as declared"', () => {
    expect(
      isAsDeclared(makeComparison({ expectation: null, missing_count: 0, unexpected_count: 0 })),
    ).toBe(false);
  });

  it('colours a not-detected family as a warning, not an error', () => {
    // A refusal to confirm is the temporal guarantee working; painting it red would
    // invite a reader to treat it as a defect.
    expect(OUTCOME_TONE.missing).toBe('warning');
    expect(OUTCOME_TONE.matched).toBe('success');
    expect(OUTCOME_TONE.unexpected).toBe('info');
  });

  it('spells out what each outcome means rather than relying on colour', () => {
    for (const outcome of ['matched', 'missing', 'unexpected'] as const) {
      expect(OUTCOME_LABEL[outcome].length).toBeGreaterThan(0);
      expect(OUTCOME_MEANING[outcome].length).toBeGreaterThan(20);
    }
    expect(OUTCOME_MEANING.missing).toMatch(/not a defect/i);
  });

  it('offers only families that have a shipped reasoner', () => {
    // `speeding` has none, so declaring it could only ever be permanently missing.
    expect(DEMO_SELECTABLE_VIOLATIONS).not.toContain('speeding');
    expect(DEMO_SELECTABLE_VIOLATIONS).toContain('red_light_jumping');
    expect(new Set(DEMO_SELECTABLE_VIOLATIONS).size).toBe(DEMO_SELECTABLE_VIOLATIONS.length);
  });

  it('keeps a declaration and a detection structurally separate', () => {
    // The expectation carries no event ids, and a row's detected side carries no
    // expectation — they meet only in a comparison the backend computed.
    const expectation = makeExpectation();
    expect(Object.keys(expectation)).not.toContain('event_ids');
    expect(Object.keys(expectation)).not.toContain('detected_count');
  });
});

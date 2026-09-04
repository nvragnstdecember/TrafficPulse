import { type ExpectationComparison, type ExpectationOutcome, type ViolationType } from '@/api/types';
import { type StatusTone } from '@/components/common/status-chip';

/**
 * Presentation logic for the controlled demonstration (pure).
 *
 * How a declared expectation and an independently confirmed event set are described
 * to a reader — and, just as importantly, how they are kept apart. Kept out of the
 * components so the wording of each outcome is testable without rendering, the same
 * split `lib/calibration.ts` uses for the drawing rules.
 *
 * Nothing here computes an accuracy. The backend deliberately reports no precision,
 * recall or F1 for a hand-authored clip, and inventing one in the client would be
 * exactly the same mistake one layer further out.
 */

/**
 * The families a controlled clip can be declared to contain, in a fixed order.
 *
 * Every one has a shipped reasoner. `speeding` is absent because none exists, so
 * declaring it could only ever produce a permanently missing row. `no_helmet` is
 * present but is additionally gated by the deployment's classifier posture — it can
 * be declared and simply reported as not detected, which is the honest outcome when
 * the capability guard refuses the rule.
 */
export const DEMO_SELECTABLE_VIOLATIONS: ViolationType[] = [
  'wrong_way',
  'illegal_stopping',
  'red_light_jumping',
  'triple_riding',
  'no_helmet',
];

export const OUTCOME_LABEL: Record<ExpectationOutcome, string> = {
  matched: 'Matched',
  missing: 'Not detected',
  unexpected: 'Unexpected',
};

/**
 * The tone each outcome reads in.
 *
 * `missing` is a **warning**, not an error: a declared family the reasoners declined
 * to confirm is the single most informative row in the table, and colouring it as a
 * failure would invite the reader to treat the refusal as a bug rather than as the
 * temporal guarantee doing its job.
 */
export const OUTCOME_TONE: Record<ExpectationOutcome, StatusTone> = {
  matched: 'success',
  missing: 'warning',
  unexpected: 'info',
};

/** What each outcome actually means, spelled out rather than left to the colour. */
export const OUTCOME_MEANING: Record<ExpectationOutcome, string> = {
  matched: 'Declared for this clip, and the reasoners independently confirmed it.',
  missing:
    'Declared for this clip, and nothing was confirmed. Either the evidence never met the '
    + 'rule’s threshold, or the rule was not run — not a defect on its own.',
  unexpected:
    'Not declared, but confirmed anyway. Worth opening: either the scenario contains more '
    + 'than was written down, or the geometry is catching something it should not.',
};

/**
 * One line summarising a comparison, for a heading.
 *
 * Deliberately reports counts and nothing derived from them. "3 of 4 matched" is a
 * fact; "75% accurate" is a measurement this project does not have.
 */
export function comparisonSummary(comparison: ExpectationComparison | null | undefined): string {
  if (!comparison) return 'No comparison yet.';
  if (comparison.expectation === null) {
    return comparison.detected_event_count === 0
      ? 'Nothing declared for this clip, and nothing confirmed.'
      : `Nothing declared for this clip; ${plural(comparison.detected_event_count, 'event')} confirmed.`;
  }
  const parts = [
    `${comparison.matched_count} of ${comparison.expected_count} declared families matched`,
    `${plural(comparison.detected_event_count, 'event')} confirmed`,
  ];
  if (comparison.missing_count > 0) parts.push(`${comparison.missing_count} not detected`);
  if (comparison.unexpected_count > 0) parts.push(`${comparison.unexpected_count} unexpected`);
  return `${parts.join(' · ')}.`;
}

/**
 * Whether the demonstration came out as declared.
 *
 * Used only to pick a chip tone. It is emphatically **not** a pass/fail verdict on
 * the system: a controlled clip that confirms everything declared shows that the
 * reasoners ran correctly over declared context, and nothing about real-world
 * performance.
 */
export function isAsDeclared(comparison: ExpectationComparison | null | undefined): boolean {
  if (!comparison?.expectation) return false;
  return comparison.missing_count === 0 && comparison.unexpected_count === 0;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

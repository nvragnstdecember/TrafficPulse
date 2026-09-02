import {
  type HelmetAnalysis,
  type PostureState,
  type RiderAnalysis,
  type RiderEnforcementStatus,
  type SystemPosture,
} from '@/api/types';
import { type StatusTone } from '@/components/common/status-chip';
import { type StatValue } from '@/lib/review-stats';

/**
 * Pure presentation logic for helmet analysis and deployment posture.
 *
 * Kept out of the components for the same reason the rest of `lib/` is: this is the
 * layer that decides what a reader is told, and it must be testable without mounting
 * anything. Two rules run through all of it.
 *
 * **A classification is never rendered as a violation.** The label and the enforcement
 * status are separate values with separate vocabularies, and no function here maps one
 * to the other. A rider can read `no_helmet` and still be unattributable; that is the
 * normal case in dense traffic, not an edge case to smooth over.
 *
 * **An unmeasured value is never a zero.** `confidence` is `null` when the classifier
 * never scored the crop, and every formatter here preserves that as an em dash.
 */

/** Tone for a capability state. `experimental` and `limited` are visually distinct. */
export function postureTone(state: PostureState): StatusTone {
  switch (state) {
    case 'active':
      return 'success';
    case 'limited':
      return 'info';
    case 'experimental':
      return 'warning';
    case 'disabled':
      return 'neutral';
    case 'unavailable':
      return 'neutral';
  }
}

const HELMET_LABEL: Record<string, string> = {
  helmet: 'Helmet',
  no_helmet: 'No helmet',
  turban: 'Turban',
  uncertain: 'Uncertain',
};

/** Display form of a helmet label; an unknown backend label passes through verbatim. */
export function helmetLabel(state: string): string {
  return HELMET_LABEL[state] ?? state;
}

/**
 * Tone for a helmet label.
 *
 * `no_helmet` is `warning`, never `error`. Red is this design system's confirmed-
 * violation colour, and nothing in an analysis is confirmed — a red chip would say
 * "violation" to every viewer regardless of the words next to it.
 */
export function helmetTone(state: string): StatusTone {
  switch (state) {
    case 'no_helmet':
      return 'warning';
    case 'helmet':
      return 'success';
    default:
      return 'neutral';
  }
}

const ENFORCEMENT_LABEL: Record<RiderEnforcementStatus, string> = {
  eligible: 'Eligible for analysis',
  multi_rider_unresolved: 'Multi-rider — driver unresolved',
  classification_abstained: 'Classification abstained',
  unstable: 'Stabilizing',
};

export function enforcementLabel(status: RiderEnforcementStatus): string {
  return ENFORCEMENT_LABEL[status] ?? status;
}

const ENFORCEMENT_HINT: Record<RiderEnforcementStatus, string> = {
  eligible:
    'One rider on this motorcycle and a settled reading. No known blocker applies — ' +
    'which is not the same as a violation having been confirmed.',
  multi_rider_unresolved:
    'Two or more riders share this motorcycle. The driver cannot be identified: the ' +
    'tracker supplies no velocity, so which end of the bike is the front is unknown. ' +
    'No driver is guessed at.',
  classification_abstained:
    'The classifier declined to call this crop — too small, off-frame, or below its ' +
    'confidence floor. An abstention is an outcome, not a missing value.',
  unstable:
    'Too few frames so far for the per-track window to settle. The reading is shown, ' +
    'but it is not yet supported by enough samples to lean on.',
};

export function enforcementHint(status: RiderEnforcementStatus): string {
  return ENFORCEMENT_HINT[status] ?? '';
}

export function enforcementTone(status: RiderEnforcementStatus): StatusTone {
  switch (status) {
    case 'eligible':
      return 'success';
    case 'multi_rider_unresolved':
      return 'warning';
    default:
      return 'neutral';
  }
}

function integer(value: number): string {
  return Math.round(value).toLocaleString();
}

/** A percentage, or an em dash when the value was never measured. */
export function percent(value: number | null): string | null {
  return value === null ? null : `${Math.round(value * 100)}%`;
}

function countFor(counts: { label: string; riders: number }[], label: string): number {
  return counts.find((entry) => entry.label === label)?.riders ?? 0;
}

/**
 * The summary tiles.
 *
 * Every label says what is actually counted. "Riders detected" rather than
 * "violations", "Motorcycles with riders" rather than "motorcycles" — the analysis
 * only ever saw bikes that carried an associated rider, and claiming a detector total
 * it does not hold would be a fabrication in the direction that flatters the demo.
 */
export function analysisStats(analysis: HelmetAnalysis): StatValue[] {
  return [
    {
      key: 'frames',
      label: 'Frames with riders',
      value: integer(analysis.frames_observed),
      hint: 'Frames on which at least one rider was associated to a motorcycle and classified.',
    },
    {
      key: 'motorcycles',
      label: 'Motorcycles with riders',
      value: integer(analysis.motorcycles_associated),
      hint: 'Motorcycle tracks that carried at least one associated rider — not every motorcycle the detector saw.',
    },
    {
      key: 'riders',
      label: 'Riders detected',
      value: integer(analysis.riders_observed),
      hint: 'Distinct rider tracks that reached the classifier.',
    },
    {
      key: 'helmet',
      label: 'Helmet',
      value: integer(countFor(analysis.label_counts, 'helmet')),
      hint: 'Rider tracks whose stabilized reading is "helmet". A classification, not a clearance.',
    },
    {
      key: 'no-helmet',
      label: 'No helmet',
      value: integer(countFor(analysis.label_counts, 'no_helmet')),
      hint: 'Rider tracks whose stabilized reading is "no helmet". No violation is confirmed from this.',
    },
    {
      key: 'multi-rider',
      label: 'Multi-rider',
      value: integer(analysis.multi_rider_riders),
      hint: 'Riders sharing a motorcycle with someone else, so their role is unresolved.',
    },
    {
      key: 'unresolved',
      label: 'Driver unresolved',
      value: integer(analysis.unresolved_riders),
      hint: 'Riders for whom no driver attribution exists, by design.',
    },
    {
      key: 'abstained',
      label: 'Abstained',
      value: integer(analysis.abstained_riders + analysis.unstable_riders),
      hint: 'Riders whose reading is uncertain or not yet settled. Reported, never rounded to a call.',
    },
  ];
}

/**
 * Rider ordering for the table: what a reviewer needs to look at first.
 *
 * Unresolved riders lead, because they are the cases where the system is declining to
 * answer and that is the thing worth seeing. Then `no_helmet` readings, then everything
 * else, and ties break on track id so the order is stable between renders.
 */
export function sortRiders(riders: RiderAnalysis[]): RiderAnalysis[] {
  const rank = (rider: RiderAnalysis): number => {
    if (rider.enforcement === 'multi_rider_unresolved') return 0;
    if (rider.helmet_state === 'no_helmet') return 1;
    if (rider.enforcement === 'eligible') return 2;
    return 3;
  };
  return [...riders].sort(
    (a, b) => rank(a) - rank(b) || a.rider_track_id.localeCompare(b.rider_track_id),
  );
}

/**
 * Why an empty event list may understate what was checked, or `null` when it does not.
 *
 * The situation this exists for is the demo's normal one. With helmet enforcement
 * disabled, a clip of bare-headed riders confirms nothing, and an event list that then
 * reads "No violations detected" states a conclusion the system was never permitted to
 * reach. That is the most consequential wrong sentence this UI could show a reviewer:
 * it converts a deliberate abstention into a finding of compliance.
 *
 * Returns `null` when helmet enforcement is `active` or `experimental` — i.e. when the
 * rule could actually run — so the original wording is untouched wherever it is honest.
 * It deliberately says the *other* families still ran, because they did, and implying
 * the whole run was inert would be its own overstatement.
 */
export function enforcementNote(posture: SystemPosture | undefined): string | null {
  if (!posture) return null;
  if (posture.helmet_enforcement === 'active' || posture.helmet_enforcement === 'experimental') {
    return null;
  }
  return (
    'Helmet violation enforcement is off in this deployment, so no helmet violation ' +
    'could be confirmed here — see the helmet analysis panel for what the classifier ' +
    'actually read. Other violation families ran normally.'
  );
}

/**
 * The instability disclosure: how much the raw per-frame label moved on this run.
 *
 * Reported as a share of *tracks that flipped at all*, which is the shape P4-U10 used
 * ("7 of 11 tracks"), so the demo's own figure is directly comparable to the one in the
 * evaluation. `null` when there are no riders — never 0%, which would read as stability.
 */
export function flipRate(riders: RiderAnalysis[]): { flipped: number; total: number } | null {
  if (riders.length === 0) return null;
  return {
    flipped: riders.filter((rider) => rider.raw_label_flips > 0).length,
    total: riders.length,
  };
}

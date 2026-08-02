import { type ReviewAction, type ReviewEntry, type ReviewStatus } from '@/api/types';

import { type ViolationTone } from './workspace';

/**
 * Review-lifecycle presentation (H9).
 *
 * Labels, tones, and the client's copy of which actions are offered from which
 * state. Pure — no React, no network.
 *
 * The client mirror of the transition table is a **UX affordance, not a
 * safeguard**: it exists so the panel can grey out an action rather than let an
 * analyst click something that will 409. The backend's table remains the only
 * authority, and a disagreement between them shows up as a rejected request, which
 * is exactly the right failure mode — the server never trusts this list.
 */

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  pending: 'Pending',
  in_review: 'In review',
  approved: 'Approved',
  rejected: 'Rejected',
  false_positive: 'False positive',
  needs_more_evidence: 'Needs evidence',
};

const REVIEW_STATUS_TONES: Record<ReviewStatus, ViolationTone> = {
  pending: 'neutral',
  in_review: 'info',
  approved: 'success',
  rejected: 'warning',
  false_positive: 'error',
  needs_more_evidence: 'warning',
};

export const ALL_REVIEW_STATUSES: ReviewStatus[] = [
  'pending',
  'in_review',
  'approved',
  'rejected',
  'false_positive',
  'needs_more_evidence',
];

export function reviewStatusLabel(status: ReviewStatus): string {
  return REVIEW_STATUS_LABELS[status] ?? status;
}

export function reviewStatusTone(status: ReviewStatus): ViolationTone {
  return REVIEW_STATUS_TONES[status] ?? 'neutral';
}

/** Whether a status records a completed decision (mirrors the backend enum). */
export function isDecided(status: ReviewStatus): boolean {
  return status === 'approved' || status === 'rejected' || status === 'false_positive';
}

export const REVIEW_ACTION_LABELS: Record<ReviewAction, string> = {
  open: 'Start review',
  note: 'Note added',
  approve: 'Approve',
  reject: 'Reject',
  false_positive: 'False positive',
  needs_more_evidence: 'Needs evidence',
  reopen: 'Reopen',
  export: 'Exported',
};

/** Past-tense phrasing for the audit history, where every entry already happened. */
const REVIEW_ACTION_HISTORY_LABELS: Record<ReviewAction, string> = {
  open: 'Review opened',
  note: 'Note added',
  approve: 'Approved',
  reject: 'Rejected',
  false_positive: 'Marked false positive',
  needs_more_evidence: 'Flagged for more evidence',
  reopen: 'Reopened',
  export: 'Exported',
};

export function reviewActionLabel(action: ReviewAction): string {
  return REVIEW_ACTION_LABELS[action] ?? action;
}

export function historyLabel(entry: ReviewEntry): string {
  return REVIEW_ACTION_HISTORY_LABELS[entry.action] ?? entry.action;
}

/** The client's mirror of the backend transition table (see the module note). */
const TRANSITIONS: Record<ReviewStatus, ReviewAction[]> = {
  pending: ['open', 'note', 'export'],
  in_review: ['note', 'approve', 'reject', 'false_positive', 'needs_more_evidence', 'export'],
  needs_more_evidence: [
    'note',
    'reopen',
    'approve',
    'reject',
    'false_positive',
    'export',
  ],
  approved: ['note', 'reopen', 'export'],
  rejected: ['note', 'reopen', 'export'],
  false_positive: ['note', 'reopen', 'export'],
};

export function canTransition(status: ReviewStatus, action: ReviewAction): boolean {
  return (TRANSITIONS[status] ?? []).includes(action);
}

/**
 * The decision buttons to offer for a status, in the order they should appear.
 *
 * `note` and `export` are excluded: notes have their own editor and export has its
 * own control, so surfacing them again as decision buttons would imply they change
 * the case's state, which they deliberately do not.
 */
export function availableDecisions(status: ReviewStatus): ReviewAction[] {
  const order: ReviewAction[] = [
    'open',
    'approve',
    'reject',
    'false_positive',
    'needs_more_evidence',
    'reopen',
  ];
  return order.filter((action) => canTransition(status, action));
}

import { describe, expect, it } from 'vitest';

import { DEFAULT_EVENT_FILTERS, filterWorkspaceEvents } from '@/lib/workspace';
import { makeWorkspaceEvent } from '@/test/fixtures';

import {
  ALL_REVIEW_STATUSES,
  availableDecisions,
  canTransition,
  isDecided,
  reviewStatusLabel,
  reviewStatusTone,
} from './review';

describe('review transition mirror', () => {
  it('matches the backend on the happy path', () => {
    expect(canTransition('pending', 'open')).toBe(true);
    expect(canTransition('in_review', 'approve')).toBe(true);
    expect(canTransition('in_review', 'false_positive')).toBe(true);
  });

  it('refuses to offer a decision the backend would reject', () => {
    // The panel greys these out so an analyst never clicks something that 409s.
    expect(canTransition('pending', 'approve')).toBe(false);
    expect(canTransition('approved', 'reject')).toBe(false);
    expect(canTransition('rejected', 'approve')).toBe(false);
  });

  it('allows a decision to be corrected by reopening', () => {
    expect(canTransition('approved', 'reopen')).toBe(true);
    expect(canTransition('false_positive', 'reopen')).toBe(true);
  });

  it('allows notes and exports from every state', () => {
    for (const status of ALL_REVIEW_STATUSES) {
      expect(canTransition(status, 'note')).toBe(true);
      expect(canTransition(status, 'export')).toBe(true);
    }
  });
});

describe('availableDecisions', () => {
  it('offers only starting a review on a pending case', () => {
    expect(availableDecisions('pending')).toEqual(['open']);
  });

  it('offers the full decision set once in review', () => {
    expect(availableDecisions('in_review')).toEqual([
      'approve',
      'reject',
      'false_positive',
      'needs_more_evidence',
    ]);
  });

  it('offers only reopening on a decided case', () => {
    expect(availableDecisions('approved')).toEqual(['reopen']);
  });

  it('never offers note or export as decision buttons', () => {
    // Both have their own controls, and surfacing them here would imply they
    // change the case's state — which they deliberately do not.
    for (const status of ALL_REVIEW_STATUSES) {
      expect(availableDecisions(status)).not.toContain('note');
      expect(availableDecisions(status)).not.toContain('export');
    }
  });
});

describe('review presentation', () => {
  it('labels and tones every status', () => {
    for (const status of ALL_REVIEW_STATUSES) {
      expect(reviewStatusLabel(status)).toBeTruthy();
      expect(reviewStatusTone(status)).toBeTruthy();
    }
  });

  it('separates false positive from rejected', () => {
    expect(reviewStatusLabel('false_positive')).toBe('False positive');
    expect(reviewStatusTone('false_positive')).not.toBe(reviewStatusTone('rejected'));
  });

  it('reports only decisions as decided', () => {
    expect(ALL_REVIEW_STATUSES.filter(isDecided)).toEqual([
      'approved',
      'rejected',
      'false_positive',
    ]);
  });
});

describe('filtering by review state', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', review_status: 'pending' }),
    makeWorkspaceEvent({ event_id: 'b', review_status: 'approved' }),
    makeWorkspaceEvent({ event_id: 'c', review_status: 'false_positive' }),
  ];

  it('includes everything when no state is selected', () => {
    expect(filterWorkspaceEvents(events, DEFAULT_EVENT_FILTERS)).toHaveLength(3);
  });

  it('narrows to the selected states', () => {
    const filtered = filterWorkspaceEvents(events, {
      ...DEFAULT_EVENT_FILTERS,
      reviewStatuses: ['approved', 'false_positive'],
    });
    expect(filtered.map((event) => event.id)).toEqual(['b', 'c']);
  });

  it('keeps working alongside the search box', () => {
    const filtered = filterWorkspaceEvents(events, {
      ...DEFAULT_EVENT_FILTERS,
      reviewStatuses: ['approved'],
      query: 'wrong way',
    });
    expect(filtered.map((event) => event.id)).toEqual(['b']);
  });

  it('is searchable by state name', () => {
    const filtered = filterWorkspaceEvents(events, {
      ...DEFAULT_EVENT_FILTERS,
      query: 'false_positive',
    });
    expect(filtered.map((event) => event.id)).toEqual(['c']);
  });
});

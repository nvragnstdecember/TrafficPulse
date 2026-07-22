import { describe, expect, it } from 'vitest';

import {
  makeConfirmedEvent,
  makeEventSummary,
  makeWorkspaceEvent,
  mediaSeconds,
} from '@/test/fixtures';

import {
  DEFAULT_EVENT_FILTERS,
  buildTimelineMarkers,
  clampRatio,
  clusterMarkers,
  eventMediaSeconds,
  filterWorkspaceEvents,
  formatClock,
  hasActiveFilters,
  mergeWorkspaceEvents,
  sortWorkspaceEvents,
  timelineDuration,
  toWorkspaceEvent,
  violationLabel,
  violationTone,
  workspaceEventsEqual,
} from './workspace';

describe('eventMediaSeconds', () => {
  it('maps an epoch-anchored timestamp onto the video timeline', () => {
    expect(eventMediaSeconds(mediaSeconds(12.5))).toBe(12.5);
  });

  it('is safe for unparseable input', () => {
    expect(eventMediaSeconds('not-a-time')).toBe(0);
  });
});

describe('violation presentation', () => {
  it('labels and tones known violations', () => {
    expect(violationLabel('no_helmet')).toBe('No helmet');
    expect(violationTone('no_helmet')).toBe('error');
    expect(violationTone('wrong_way')).toBe('warning');
  });

  it('degrades gracefully for an unknown violation', () => {
    expect(violationLabel('some_new_rule')).toBe('Some new rule');
    expect(violationTone('some_new_rule')).toBe('neutral');
  });
});

describe('toWorkspaceEvent', () => {
  it('derives the media position from the summary', () => {
    const event = toWorkspaceEvent(makeEventSummary({ trigger_at: mediaSeconds(42) }));
    expect(event.mediaSeconds).toBe(42);
    expect(event.confidence).toBeNull();
    expect(event.lane).toBeNull();
  });

  it('enriches confidence and lane from the detail record', () => {
    const event = toWorkspaceEvent(
      makeEventSummary(),
      makeConfirmedEvent({
        confidence: { overall: 0.8 },
        measurements: [{ name: 'lane_index', value: 3, unit: null }],
      }),
    );
    expect(event.confidence).toBe(0.8);
    expect(event.lane).toBe('3');
  });
});

describe('timeline building', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(5) }),
    makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(50) }),
  ];

  it('clamps ratios into 0..1', () => {
    expect(clampRatio(-2)).toBe(0);
    expect(clampRatio(4)).toBe(1);
    expect(clampRatio(Number.NaN)).toBe(0);
  });

  it('prefers the player duration and falls back to the latest event', () => {
    expect(timelineDuration(100, events)).toBe(100);
    expect(timelineDuration(null, events)).toBe(50);
    expect(timelineDuration(null, [])).toBe(0);
  });

  it('positions markers along the timeline', () => {
    const markers = buildTimelineMarkers(events, 100);
    expect(markers.map((marker) => marker.positionRatio)).toEqual([0.05, 0.5]);
  });

  it('positions markers at zero when the duration is unknown', () => {
    expect(buildTimelineMarkers(events, 0).every((m) => m.positionRatio === 0)).toBe(true);
  });

  it('clusters markers that would visually overlap', () => {
    const dense = buildTimelineMarkers(
      [
        makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(10) }),
        makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(10.5) }),
        makeWorkspaceEvent({ event_id: 'c', trigger_at: mediaSeconds(80) }),
      ],
      100,
    );
    const clusters = clusterMarkers(dense);
    expect(clusters).toHaveLength(2);
    expect(clusters[0].markers).toHaveLength(2);
    expect(clusters[1].markers).toHaveLength(1);
  });
});

describe('filtering', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', violation_type: 'wrong_way', camera_id: 'cam-north' }),
    makeWorkspaceEvent({ event_id: 'b', violation_type: 'no_helmet', camera_id: 'cam-south' }),
  ];

  it('reports whether any filter is active', () => {
    expect(hasActiveFilters(DEFAULT_EVENT_FILTERS)).toBe(false);
    expect(hasActiveFilters({ ...DEFAULT_EVENT_FILTERS, query: ' ' })).toBe(false);
    expect(hasActiveFilters({ ...DEFAULT_EVENT_FILTERS, minConfidence: 0.2 })).toBe(true);
  });

  it('matches the query across id, camera, rule, label, and tracks', () => {
    expect(
      filterWorkspaceEvents(events, { ...DEFAULT_EVENT_FILTERS, query: 'cam-south' }),
    ).toHaveLength(1);
    expect(
      filterWorkspaceEvents(events, { ...DEFAULT_EVENT_FILTERS, query: 'No helmet' }),
    ).toHaveLength(1);
    expect(filterWorkspaceEvents(events, DEFAULT_EVENT_FILTERS)).toHaveLength(2);
  });

  it('filters by violation type', () => {
    const filtered = filterWorkspaceEvents(events, {
      ...DEFAULT_EVENT_FILTERS,
      violationTypes: ['no_helmet'],
    });
    expect(filtered.map((event) => event.id)).toEqual(['b']);
  });

  it('drops events without a known confidence once a threshold is set', () => {
    const withConfidence = { ...events[0], confidence: 0.9 };
    const filtered = filterWorkspaceEvents([withConfidence, events[1]], {
      ...DEFAULT_EVENT_FILTERS,
      minConfidence: 0.5,
    });
    expect(filtered.map((event) => event.id)).toEqual(['a']);
  });
});

describe('sorting', () => {
  const early = makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(5) });
  const late = makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(50) });
  const events = [late, early];

  it('sorts by time in both directions', () => {
    expect(sortWorkspaceEvents(events, 'time-asc').map((e) => e.id)).toEqual(['a', 'b']);
    expect(sortWorkspaceEvents(events, 'time-desc').map((e) => e.id)).toEqual(['b', 'a']);
  });

  it('sorts unknown confidence last', () => {
    const scored = [{ ...early, confidence: 0.4 }, late];
    expect(sortWorkspaceEvents(scored, 'confidence-desc').map((e) => e.id)).toEqual(['a', 'b']);
  });

  it('sorts by violation label, then time', () => {
    const helmet = makeWorkspaceEvent({
      event_id: 'c',
      violation_type: 'no_helmet',
      trigger_at: mediaSeconds(90),
    });
    expect(sortWorkspaceEvents([late, helmet], 'violation').map((e) => e.id)).toEqual(['c', 'b']);
  });

  it('does not mutate the input array', () => {
    const input = [...events];
    sortWorkspaceEvents(input, 'time-asc');
    expect(input.map((e) => e.id)).toEqual(['b', 'a']);
  });
});

describe('mergeWorkspaceEvents (H7D)', () => {
  it('returns the same array reference when nothing changed', () => {
    const previous = [makeWorkspaceEvent({ event_id: 'a' })];
    const incoming = [makeWorkspaceEvent({ event_id: 'a' })];
    expect(mergeWorkspaceEvents(previous, incoming)).toBe(previous);
  });

  it('preserves the reference of unchanged events while appending new ones', () => {
    const a = makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(5) });
    const previous = [a];
    const incoming = [
      makeWorkspaceEvent({ event_id: 'a', trigger_at: mediaSeconds(5) }),
      makeWorkspaceEvent({ event_id: 'b', trigger_at: mediaSeconds(9) }),
    ];
    const merged = mergeWorkspaceEvents(previous, incoming);
    expect(merged).not.toBe(previous);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toBe(a); // unchanged event keeps its identity
    expect(merged[1].id).toBe('b');
  });

  it('replaces the reference of an event whose content changed', () => {
    const a = makeWorkspaceEvent({ event_id: 'a' });
    const previous = [a];
    const enriched = { ...a, confidence: 0.9 };
    const merged = mergeWorkspaceEvents(previous, [enriched]);
    expect(merged[0]).not.toBe(a);
    expect(merged[0].confidence).toBe(0.9);
  });

  it('from an empty previous set, adopts the incoming events by reference', () => {
    const incoming = [makeWorkspaceEvent({ event_id: 'a' })];
    const merged = mergeWorkspaceEvents([], incoming);
    expect(merged).toEqual(incoming);
    expect(merged[0]).toBe(incoming[0]); // element identity preserved
  });
});

describe('workspaceEventsEqual', () => {
  it('is true for identical view-models and false when a field differs', () => {
    const a = makeWorkspaceEvent({ event_id: 'a' });
    expect(workspaceEventsEqual(a, { ...a })).toBe(true);
    expect(workspaceEventsEqual(a, { ...a, confidence: 0.5 })).toBe(false);
    expect(workspaceEventsEqual(a, { ...a, trackIds: ['x', 'y'] })).toBe(false);
  });
});

describe('formatClock', () => {
  it('formats minutes and hours', () => {
    expect(formatClock(0)).toBe('0:00');
    expect(formatClock(65)).toBe('1:05');
    expect(formatClock(3725)).toBe('1:02:05');
  });

  it('is safe for missing or invalid values', () => {
    expect(formatClock(null)).toBe('0:00');
    expect(formatClock(-4)).toBe('0:00');
    expect(formatClock(Number.NaN)).toBe('0:00');
  });
});

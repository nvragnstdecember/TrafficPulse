import { describe, expect, it } from 'vitest';

import { makeWorkspaceEvent, mediaSeconds } from '@/test/fixtures';

import {
  averageConfidence,
  averageObservationSeconds,
  distinctTrackCount,
  processingSummary,
  reviewStats,
  violationBreakdown,
} from './review-stats';
import {
  REVIEW_LEAD_IN_SECONDS,
  REVIEW_TAIL_SECONDS,
  buildEventNarrative,
  confidenceComponents,
  headlineConfidence,
  parseClockQuery,
  reviewWindow,
} from './workspace';

describe('confidence', () => {
  it('headlines the strongest measured component and names it', () => {
    const headline = headlineConfidence({ classifier: 0.9, temporal_consistency: 0.4 });
    expect(headline).toEqual({ value: 0.9, source: 'classifier' });
  });

  it('prefers a calibrated aggregate when one exists', () => {
    // The system publishes null today, but a calibrated backend would be the
    // correct single number and must win.
    expect(headlineConfidence({ aggregate: 0.7, classifier: 0.99 })?.source).toBe('aggregate');
  });

  it('returns null rather than inventing a score', () => {
    expect(headlineConfidence({})).toBeNull();
    expect(headlineConfidence(undefined)).toBeNull();
    expect(headlineConfidence({ classifier: null })).toBeNull();
  });

  it('lists every measured component in a stable order, skipping absent ones', () => {
    const components = confidenceComponents({
      association: 0.5,
      classifier: 0.9,
      detector: null,
    });
    expect(components.map((component) => component.key)).toEqual(['classifier', 'association']);
    expect(components[0].label).toBe('Classifier');
  });

  it('clamps a component into 0..1', () => {
    expect(headlineConfidence({ classifier: 1.4 })?.value).toBe(1);
    expect(headlineConfidence({ classifier: -0.2 })?.value).toBe(0);
  });
});

describe('parseClockQuery', () => {
  it('reads the clock formats an analyst would type', () => {
    expect(parseClockQuery('90')).toBe(90);
    expect(parseClockQuery('1:23')).toBe(83);
    expect(parseClockQuery('01:23')).toBe(83);
    expect(parseClockQuery('1:02:03')).toBe(3723);
  });

  it('rejects anything that is not a clock', () => {
    expect(parseClockQuery('iou-7')).toBeNull();
    expect(parseClockQuery('no helmet')).toBeNull();
    expect(parseClockQuery('')).toBeNull();
  });
});

describe('reviewWindow', () => {
  const event = makeWorkspaceEvent({
    start_at: mediaSeconds(30),
    trigger_at: mediaSeconds(34),
  });

  it('starts before support began and ends after the trigger', () => {
    // Seeking to the trigger alone would show the aftermath, not the behaviour.
    const window = reviewWindow(event, 120);
    expect(window.from).toBeCloseTo(30 - REVIEW_LEAD_IN_SECONDS);
    expect(window.to).toBeCloseTo(34 + REVIEW_TAIL_SECONDS);
  });

  it('never seeks before the start of the media', () => {
    const early = makeWorkspaceEvent({
      start_at: mediaSeconds(0.2),
      trigger_at: mediaSeconds(0.5),
    });
    expect(reviewWindow(early, 60).from).toBe(0);
  });

  it('clamps the tail to the media duration', () => {
    expect(reviewWindow(event, 34.5).to).toBe(34.5);
  });

  it('tolerates an unknown duration', () => {
    expect(reviewWindow(event, null).to).toBeCloseTo(36);
  });
});

describe('buildEventNarrative', () => {
  const event = makeWorkspaceEvent({
    start_at: mediaSeconds(20),
    trigger_at: mediaSeconds(24),
  });

  it('tells the story from observation to confirmation', () => {
    const steps = buildEventNarrative(event, { thresholdSeconds: 3, hasEvidence: true });
    expect(steps.map((step) => step.stage)).toEqual([
      'observation',
      'accumulation',
      'confirmation',
      'evidence',
    ]);
    expect(steps[0].seconds).toBe(20);
    expect(steps[2].seconds).toBe(24);
  });

  it('states accumulation against the rule’s own threshold', () => {
    const steps = buildEventNarrative(event, { thresholdSeconds: 3 });
    expect(steps[1].detail).toContain('of 3.00s required');
  });

  it('omits the evidence step when no manifest exists', () => {
    const steps = buildEventNarrative(event, { hasEvidence: false });
    expect(steps.map((step) => step.key)).not.toContain('evidence');
  });

  it('omits the accumulation step for an instantaneous window', () => {
    // Padding the story with a step that describes nothing would be fabrication.
    const instant = makeWorkspaceEvent({
      start_at: mediaSeconds(10),
      trigger_at: mediaSeconds(10),
    });
    expect(buildEventNarrative(instant).map((step) => step.stage)).toEqual([
      'observation',
      'confirmation',
    ]);
  });

  it('never fabricates a threshold it was not given', () => {
    const steps = buildEventNarrative(event, { thresholdSeconds: null });
    expect(steps[1].detail).not.toContain('required');
  });
});

describe('review statistics', () => {
  const events = [
    makeWorkspaceEvent({ event_id: 'a', violation_type: 'no_helmet', track_ids: ['t1', 't2'] }),
    makeWorkspaceEvent({ event_id: 'b', violation_type: 'no_helmet', track_ids: ['t2'] }),
    makeWorkspaceEvent({ event_id: 'c', violation_type: 'wrong_way', track_ids: ['t9'] }),
  ];

  it('averages only the events that carry a confidence', () => {
    const mixed = [{ ...events[0], confidence: 0.8 }, { ...events[1], confidence: null }];
    expect(averageConfidence(mixed)).toBeCloseTo(0.8);
  });

  it('returns null when nothing was measured, never zero', () => {
    expect(averageConfidence([{ ...events[0], confidence: null }])).toBeNull();
    expect(averageConfidence([])).toBeNull();
    expect(averageObservationSeconds([])).toBeNull();
  });

  it('counts distinct tracks, not track mentions', () => {
    expect(distinctTrackCount(events)).toBe(3);
  });

  it('breaks violations down, most frequent first', () => {
    expect(violationBreakdown(events).map((entry) => [entry.type, entry.count])).toEqual([
      ['no_helmet', 2],
      ['wrong_way', 1],
    ]);
  });

  it('renders an unmeasured engine metric as null rather than 0', () => {
    const stats = reviewStats(events, null);
    expect(stats.find((stat) => stat.key === 'detections')?.value).toBeNull();
    expect(stats.find((stat) => stat.key === 'violations')?.value).toBe('3');
  });

  it('summarizes processing from the engine snapshot', () => {
    const summary = processingSummary(null, {
      frames_read: 300,
      frames_skipped_stride: 0,
      frames_skipped_fps: 0,
      frames_dropped_backpressure: 0,
      frames_admitted: 300,
      frames_processed: 300,
      batches_processed: 10,
      detections: 1200,
      track_states: 900,
      events_confirmed: 3,
      queue_peak: 2,
      media_fps: 30,
      wall_fps: 1.25,
      memory_bytes_current: null,
      memory_bytes_peak: null,
      gpu_memory_bytes_current: null,
      gpu_memory_bytes_peak: null,
    }, 240);
    const byKey = Object.fromEntries(summary.map((stat) => [stat.key, stat.value]));
    expect(byKey.frames).toBe('300');
    expect(byKey.fps).toBe('1.3');
    expect(byKey.events).toBe('3');
    expect(byKey.duration).toBe('240.0s');
  });
});

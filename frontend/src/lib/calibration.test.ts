import { describe, expect, it } from 'vitest';

import { type SignalPhaseSpec } from '@/api/types';

import {
  type CalibrationShapes,
  EMPTY_SHAPES,
  ZONE_IDS,
  buildSceneDraft,
  canSubmit,
  centroid,
  isScheduleUsable,
  isSegmentComplete,
  perpendicular,
  previewUnlocked,
  rulesForRun,
  segmentVector,
  sortSchedule,
} from './calibration';

const LANE: CalibrationShapes['lane'] = [
  [0, 0],
  [320, 0],
  [320, 240],
  [0, 240],
];
const JUNCTION: CalibrationShapes['junction'] = [
  [100, 150],
  [220, 150],
  [220, 235],
  [100, 235],
];

function shapes(overrides: Partial<CalibrationShapes> = {}): CalibrationShapes {
  return { ...EMPTY_SHAPES, ...overrides };
}

function draft(overrides: Partial<CalibrationShapes> = {}) {
  return buildSceneDraft({
    shapes: shapes(overrides),
    frameWidth: 320,
    frameHeight: 240,
    cameraId: 'cam-1',
    sceneName: 'Scene',
  });
}

// --- geometry helpers ------------------------------------------------------------
describe('segmentVector', () => {
  it('returns a unit vector', () => {
    expect(segmentVector({ from: [0, 0], to: [0, 100] })).toEqual({ dx: 0, dy: 1 });
    expect(segmentVector({ from: [0, 0], to: [30, 40] })).toEqual({ dx: 0.6, dy: 0.8 });
  });

  it('refuses a zero-length segment', () => {
    // Rounding a zero vector would produce (0,0), which the scene contract rejects
    // as a direction -- better to report "not drawn yet".
    expect(segmentVector({ from: [5, 5], to: [5, 5] })).toBeNull();
    expect(segmentVector(null)).toBeNull();
  });

  it('rounds so the same drawing yields the same scene hash', () => {
    const vector = segmentVector({ from: [0, 0], to: [1, 3] });
    expect(vector).toEqual({ dx: 0.3162, dy: 0.9487 });
  });
});

describe('perpendicular', () => {
  it('orients the crossing direction toward the junction', () => {
    // The single easiest thing for an analyst to get backwards, so it is derived
    // rather than drawn: a stop line above the junction must point down into it.
    const line = { from: [100, 120] as [number, number], to: [220, 120] as [number, number] };

    const normal = perpendicular(line, centroid(JUNCTION));
    const vector = segmentVector(normal);

    expect(vector?.dy).toBeGreaterThan(0); // downward, into the junction
    expect(vector?.dx).toBe(0);
  });

  it('flips when the junction is on the other side', () => {
    const line = { from: [100, 200] as [number, number], to: [220, 200] as [number, number] };
    const above: [number, number][] = [
      [100, 20],
      [220, 20],
      [220, 100],
      [100, 100],
    ];

    const vector = segmentVector(perpendicular(line, centroid(above)));

    expect(vector?.dy).toBeLessThan(0);
  });

  it('falls back to the raw normal with nothing to orient against', () => {
    const line = { from: [0, 0] as [number, number], to: [100, 0] as [number, number] };
    expect(segmentVector(perpendicular(line))).toEqual({ dx: 0, dy: 1 });
  });
});

describe('centroid', () => {
  it('averages the vertices, and reports nothing for an empty ring', () => {
    expect(centroid(JUNCTION)).toEqual([160, 192.5]);
    expect(centroid([])).toBeUndefined();
  });
});

// --- draft assembly ---------------------------------------------------------------
describe('buildSceneDraft', () => {
  it('needs a lane before anything can be submitted', () => {
    expect(canSubmit(shapes())).toBe(false);
    expect(canSubmit(shapes({ lane: LANE }))).toBe(true);
  });

  it('omits incomplete shapes rather than half-submitting them', () => {
    // A two-point polygon is not a zone; sending one trades a clear client state
    // for a server validation error.
    const result = draft({ lane: LANE, junction: [[1, 1], [2, 2]] });

    expect(result.zones.map((z) => z.zone_id)).toEqual([ZONE_IDS.lane]);
    expect(result.stop_lines).toBeUndefined();
  });

  it('wires a direction to the lane it governs', () => {
    const result = draft({ lane: LANE, direction: { from: [10, 200], to: [10, 20] } });

    expect(result.direction).toEqual({
      direction_id: 'dir-legal',
      dx: 0,
      dy: -1,
      zone_id: ZONE_IDS.lane,
    });
  });

  it('emits a no-stopping zone when one is drawn', () => {
    const result = draft({ lane: LANE, noStopping: JUNCTION });

    const zone = result.zones.find((z) => z.zone_id === ZONE_IDS.noStopping);
    expect(zone?.zone_type).toBe('no_stopping');
  });

  it('emits a stop line and signal group only alongside a junction', () => {
    // A stop line without the junction it guards is a dangling reference the scene
    // contract would reject, so the two travel together.
    const withoutJunction = draft({
      lane: LANE,
      stopLine: { from: [100, 120], to: [220, 120] },
    });
    expect(withoutJunction.stop_lines).toBeUndefined();
    expect(withoutJunction.signal_groups).toBeUndefined();

    const complete = draft({
      lane: LANE,
      junction: JUNCTION,
      stopLine: { from: [100, 120], to: [220, 120] },
    });
    expect(complete.stop_lines).toHaveLength(1);
    expect(complete.stop_lines?.[0].signal_group_id).toBe('sg-1');
    expect(complete.stop_lines?.[0].zone_ids).toEqual([ZONE_IDS.junction]);
    expect(complete.stop_lines?.[0].crossing_dy).toBeGreaterThan(0);
    expect(complete.signal_groups?.[0].zone_ids).toEqual([ZONE_IDS.junction]);
  });

  it('uses the junction as the signal ROI when no head was outlined', () => {
    // The contract requires an ROI polygon; claiming a signal head we were never
    // shown would be worse than reusing a shape the analyst did draw.
    const result = draft({
      lane: LANE,
      junction: JUNCTION,
      stopLine: { from: [100, 120], to: [220, 120] },
    });

    expect(result.signal_groups?.[0].roi_polygon).toEqual(JUNCTION);
  });

  it('prefers a drawn signal head when there is one', () => {
    const head: [number, number][] = [
      [5, 5],
      [45, 5],
      [45, 60],
    ];
    const result = draft({
      lane: LANE,
      junction: JUNCTION,
      stopLine: { from: [100, 120], to: [220, 120] },
      signalRoi: head,
    });

    expect(result.signal_groups?.[0].roi_polygon).toEqual(head);
  });

  it('carries the frame size the geometry was drawn against', () => {
    const result = draft({ lane: LANE });
    expect([result.frame_width, result.frame_height]).toEqual([320, 240]);
  });
});

// --- capability preview -------------------------------------------------------------
describe('previewUnlocked', () => {
  it('always offers the geometry-free motorcycle rules', () => {
    expect(previewUnlocked(shapes({ lane: LANE }))).toEqual(['no_helmet', 'triple_riding']);
  });

  it('adds wrong way only with a lane and a direction', () => {
    const withDirection = previewUnlocked(
      shapes({ lane: LANE, direction: { from: [10, 200], to: [10, 20] } }),
    );
    expect(withDirection).toContain('wrong_way');
  });

  it('adds red light only with both a stop line and a junction', () => {
    const lineOnly = previewUnlocked(
      shapes({ lane: LANE, stopLine: { from: [100, 120], to: [220, 120] } }),
    );
    const both = previewUnlocked(
      shapes({
        lane: LANE,
        junction: JUNCTION,
        stopLine: { from: [100, 120], to: [220, 120] },
      }),
    );

    expect(lineOnly).not.toContain('red_light_jumping');
    expect(both).toContain('red_light_jumping');
  });
});

describe('isSegmentComplete', () => {
  it('treats a segment with identical ends as unfinished', () => {
    expect(isSegmentComplete({ from: [1, 1], to: [1, 1] })).toBe(false);
    expect(isSegmentComplete({ from: [1, 1], to: [2, 2] })).toBe(true);
    expect(isSegmentComplete(null)).toBe(false);
  });
});

// --- signal schedule -------------------------------------------------------------------
describe('isScheduleUsable', () => {
  it('refuses an empty schedule', () => {
    // Every instant would resolve to `unknown` and the rule could never confirm;
    // offering to run with it would promise an analysis that cannot happen.
    expect(isScheduleUsable([])).toBe(false);
  });

  it('refuses out-of-order phases', () => {
    expect(
      isScheduleUsable([
        { at_seconds: 10, state: 'red' },
        { at_seconds: 2, state: 'green' },
      ]),
    ).toBe(false);
  });

  it('accepts a non-decreasing schedule', () => {
    expect(
      isScheduleUsable([
        { at_seconds: 0, state: 'red' },
        { at_seconds: 0, state: 'amber' },
        { at_seconds: 12.5, state: 'green' },
      ]),
    ).toBe(true);
  });
});

describe('sortSchedule', () => {
  it('orders by time without mutating the input', () => {
    const input: SignalPhaseSpec[] = [
      { at_seconds: 9, state: 'green' },
      { at_seconds: 1, state: 'red' },
    ];

    expect(sortSchedule(input).map((p) => p.at_seconds)).toEqual([1, 9]);
    expect(input[0].at_seconds).toBe(9);
  });
});

// --- rules for a run ---------------------------------------------------------------------
describe('rulesForRun', () => {
  it('maps supported violations onto rule declarations', () => {
    const rules = rulesForRun(['wrong_way', 'illegal_stopping', 'triple_riding'], []);

    expect(rules).toEqual([
      { kind: 'wrong_way' },
      { kind: 'illegal_stopping' },
      { kind: 'triple_riding' },
    ]);
  });

  it('omits red light when no usable schedule has been entered', () => {
    // The backend refuses a schedule-less red-light rule; sending one would turn a
    // knowable client-side state into a failed run.
    expect(rulesForRun(['red_light_jumping'], [])).toEqual([]);
    expect(
      rulesForRun(['red_light_jumping'], [
        { at_seconds: 5, state: 'red' },
        { at_seconds: 1, state: 'green' },
      ]),
    ).toEqual([]);
  });

  it('includes red light with its sorted schedule when one exists', () => {
    const rules = rulesForRun(['red_light_jumping'], [
      { at_seconds: 0, state: 'red' },
      { at_seconds: 12, state: 'green' },
    ]);

    expect(rules).toEqual([
      {
        kind: 'red_light_jumping',
        schedule: [
          { at_seconds: 0, state: 'red' },
          { at_seconds: 12, state: 'green' },
        ],
      },
    ]);
  });

  it('never requests speeding, which has no shipped reasoner', () => {
    expect(rulesForRun(['speeding', 'no_helmet'], [])).toEqual([{ kind: 'no_helmet' }]);
  });
});

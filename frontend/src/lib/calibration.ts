import {
  type SceneDraft,
  type ScenePoint,
  type SignalPhaseSpec,
  type SignalState,
  type ViolationType,
} from '@/api/types';

/**
 * Pure calibration logic (H12 surface, H13 junction tools).
 *
 * Everything about turning drawn shapes into a `SceneDraft`, and a draft plus a
 * signal schedule into the rule declarations a processing run needs. Kept out of
 * the component so the rules are testable without rendering a canvas — the same
 * split `lib/job.ts` uses for the processing lifecycle.
 *
 * Geometry is in the **video's own pixel space** throughout. The backend rejects a
 * draft whose frame size does not match the video's decoded dimensions, so there is
 * exactly one coordinate system and no scaling to get wrong.
 */

/** The shapes an analyst can draw, and which violation each unlocks. */
export type CalibrationTool =
  | 'lane'
  | 'direction'
  | 'no-stopping'
  | 'junction'
  | 'stop-line'
  | 'signal-roi';

export interface ToolSpec {
  id: CalibrationTool;
  label: string;
  /** `polygon` collects points until closed; `segment` takes exactly two. */
  shape: 'polygon' | 'segment';
  hint: string;
  /** The violation this shape contributes to, for the UI to explain itself. */
  unlocks: ViolationType | null;
}

export const CALIBRATION_TOOLS: ToolSpec[] = [
  {
    id: 'lane',
    label: 'Lane',
    shape: 'polygon',
    hint: 'Outline the carriageway being monitored. Every scene needs one.',
    unlocks: null,
  },
  {
    id: 'direction',
    label: 'Traffic direction',
    shape: 'segment',
    hint: 'Drag along the direction traffic legally travels.',
    unlocks: 'wrong_way',
  },
  {
    id: 'no-stopping',
    label: 'No-stopping zone',
    shape: 'polygon',
    hint: 'Outline where stopping is prohibited. Not observable from footage — this is a policy fact.',
    unlocks: 'illegal_stopping',
  },
  {
    id: 'stop-line',
    label: 'Stop line',
    shape: 'segment',
    hint: 'Draw the stop line, then set which way crossing it counts as entering.',
    unlocks: 'red_light_jumping',
  },
  {
    id: 'junction',
    label: 'Junction area',
    shape: 'polygon',
    hint: 'Outline the conflict area beyond the stop line. Keep it clear of the line so a vehicle stopping just past the line is not counted.',
    unlocks: 'red_light_jumping',
  },
  {
    id: 'signal-roi',
    label: 'Signal head',
    shape: 'polygon',
    hint: 'Outline the signal head this approach obeys.',
    unlocks: null,
  },
];

export interface Segment {
  from: ScenePoint;
  to: ScenePoint;
}

/** Everything drawn on the frame, before it becomes a draft. */
export interface CalibrationShapes {
  lane: ScenePoint[];
  direction: Segment | null;
  noStopping: ScenePoint[];
  junction: ScenePoint[];
  stopLine: Segment | null;
  signalRoi: ScenePoint[];
}

export const EMPTY_SHAPES: CalibrationShapes = {
  lane: [],
  direction: null,
  noStopping: [],
  junction: [],
  stopLine: null,
  signalRoi: [],
};

export const ZONE_IDS = {
  lane: 'zone-lane',
  noStopping: 'zone-no-stopping',
  junction: 'zone-junction',
} as const;

const DIRECTION_ID = 'dir-legal';
const STOP_LINE_ID = 'sl-1';
const SIGNAL_GROUP_ID = 'sg-1';

/** A polygon needs three points; a segment needs two distinct ends. */
export function isPolygonComplete(points: ScenePoint[]): boolean {
  return points.length >= 3;
}

export function isSegmentComplete(segment: Segment | null): boolean {
  if (!segment) return false;
  return segment.from[0] !== segment.to[0] || segment.from[1] !== segment.to[1];
}

/** The unit vector from a drawn segment, or `null` when it has no length. */
export function segmentVector(segment: Segment | null): { dx: number; dy: number } | null {
  if (!isSegmentComplete(segment) || !segment) return null;
  const dx = segment.to[0] - segment.from[0];
  const dy = segment.to[1] - segment.from[1];
  const length = Math.hypot(dx, dy);
  if (length === 0) return null;
  // Rounded so the same drawing produces the same scene content — and therefore
  // the same hash — rather than a new revision per pixel of mouse jitter.
  return { dx: Number((dx / length).toFixed(4)), dy: Number((dy / length).toFixed(4)) };
}

/**
 * Whether the drawn shapes can form a scene at all.
 *
 * Only the structural minimum is checked here: the backend's `validate` endpoint is
 * the authority and reports the contract's own messages. This exists so the Save
 * button can be disabled before a pointless round trip.
 */
export function canSubmit(shapes: CalibrationShapes): boolean {
  return isPolygonComplete(shapes.lane);
}

/**
 * Which violations the drawn shapes would unlock, for live UI feedback.
 *
 * A *preview*, deliberately not the authority — the server probes the real rule
 * factories and its answer is what the UI shows once a draft has been validated.
 * The two agree on the geometry, and where they could diverge the server wins.
 */
export function previewUnlocked(shapes: CalibrationShapes): ViolationType[] {
  const unlocked: ViolationType[] = [];
  if (isPolygonComplete(shapes.lane) && isSegmentComplete(shapes.direction)) {
    unlocked.push('wrong_way');
  }
  if (isPolygonComplete(shapes.noStopping)) unlocked.push('illegal_stopping');
  if (isPolygonComplete(shapes.junction) && isSegmentComplete(shapes.stopLine)) {
    unlocked.push('red_light_jumping');
  }
  // Motorcycle rules need no geometry at all; an authored scene always carries them.
  unlocked.push('no_helmet', 'triple_riding');
  return unlocked;
}

export interface BuildDraftInput {
  shapes: CalibrationShapes;
  frameWidth: number;
  frameHeight: number;
  cameraId: string;
  sceneName: string;
}

/**
 * Assemble the `SceneDraft` the API expects from the drawn shapes.
 *
 * Shapes that are incomplete are simply omitted rather than half-submitted: a
 * two-point polygon is not a zone, and sending one would trade a clear client-side
 * state for a server validation error.
 */
export function buildSceneDraft({
  shapes,
  frameWidth,
  frameHeight,
  cameraId,
  sceneName,
}: BuildDraftInput): SceneDraft {
  const zones: SceneDraft['zones'] = [];
  if (isPolygonComplete(shapes.lane)) {
    zones.push({ zone_id: ZONE_IDS.lane, zone_type: 'lane', polygon: shapes.lane });
  }
  if (isPolygonComplete(shapes.noStopping)) {
    zones.push({
      zone_id: ZONE_IDS.noStopping,
      zone_type: 'no_stopping',
      polygon: shapes.noStopping,
    });
  }
  const hasJunction = isPolygonComplete(shapes.junction);
  if (hasJunction) {
    zones.push({
      zone_id: ZONE_IDS.junction,
      zone_type: 'intersection',
      polygon: shapes.junction,
    });
  }

  const direction = segmentVector(shapes.direction);
  // The crossing direction is derived from the stop line's normal, oriented toward
  // the junction — so the analyst draws one line instead of a line plus an arrow
  // they could point the wrong way (which would make the rule silently never fire).
  const crossing = isSegmentComplete(shapes.stopLine)
    ? segmentVector(perpendicular(shapes.stopLine!, centroid(shapes.junction)))
    : null;

  const draft: SceneDraft = {
    scene_name: sceneName,
    camera_id: cameraId,
    frame_width: frameWidth,
    frame_height: frameHeight,
    zones,
  };

  if (direction && isPolygonComplete(shapes.lane)) {
    draft.direction = {
      direction_id: DIRECTION_ID,
      dx: direction.dx,
      dy: direction.dy,
      zone_id: ZONE_IDS.lane,
    };
  }

  // A stop line without the junction it guards cannot support red-light reasoning,
  // and the contract would reject the dangling reference, so both go together.
  if (crossing && hasJunction) {
    draft.stop_lines = [
      {
        stop_line_id: STOP_LINE_ID,
        a: shapes.stopLine!.from,
        b: shapes.stopLine!.to,
        crossing_dx: crossing.dx,
        crossing_dy: crossing.dy,
        signal_group_id: SIGNAL_GROUP_ID,
        zone_ids: [ZONE_IDS.junction],
      },
    ];
    draft.signal_groups = [
      {
        signal_group_id: SIGNAL_GROUP_ID,
        // The contract requires an ROI polygon. When the analyst has not outlined
        // the signal head, the junction stands in: the group's identity and its
        // link to the stop line are what red-light reasoning uses, and claiming a
        // head we were not shown would be worse than reusing a known shape.
        roi_polygon: isPolygonComplete(shapes.signalRoi) ? shapes.signalRoi : shapes.junction,
        zone_ids: [ZONE_IDS.junction],
      },
    ];
  }

  return draft;
}

/**
 * The direction crossing a stop line counts as *entering* the junction.
 *
 * Derived rather than drawn: it is the segment's normal, oriented toward the
 * junction's centroid, so an analyst draws one line instead of a line plus an
 * arrow they could easily point the wrong way. Falls back to the raw normal when
 * there is no junction to orient against.
 */
export function perpendicular(segment: Segment, towards?: ScenePoint): Segment {
  const dx = segment.to[0] - segment.from[0];
  const dy = segment.to[1] - segment.from[1];
  const midpoint: ScenePoint = [
    (segment.from[0] + segment.to[0]) / 2,
    (segment.from[1] + segment.to[1]) / 2,
  ];
  // Left normal of (dx, dy) in an image space whose y grows downward.
  let normal: ScenePoint = [-dy, dx];
  if (towards) {
    const toTarget: ScenePoint = [towards[0] - midpoint[0], towards[1] - midpoint[1]];
    if (normal[0] * toTarget[0] + normal[1] * toTarget[1] < 0) {
      normal = [dy, -dx];
    }
  }
  return { from: midpoint, to: [midpoint[0] + normal[0], midpoint[1] + normal[1]] };
}

/** The average of a polygon's vertices — good enough to orient a normal. */
export function centroid(points: ScenePoint[]): ScenePoint | undefined {
  if (points.length === 0) return undefined;
  const sum = points.reduce<[number, number]>((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
  return [sum[0] / points.length, sum[1] / points.length];
}

// --- signal schedule ---------------------------------------------------------------
export const SIGNAL_STATES: SignalState[] = ['red', 'amber', 'green', 'off'];

/**
 * Whether a schedule can drive a red-light run.
 *
 * Non-empty and non-decreasing: an empty schedule resolves every instant to
 * `unknown` and the rule could never confirm, and out-of-order phases make the step
 * function ambiguous. The backend refuses both; this mirrors it so the UI can say so
 * before the request.
 */
export function isScheduleUsable(schedule: SignalPhaseSpec[]): boolean {
  if (schedule.length === 0) return false;
  return schedule.every(
    (phase, index) => index === 0 || phase.at_seconds >= schedule[index - 1].at_seconds,
  );
}

/** Order a schedule by time, so an analyst can add phases in any order. */
export function sortSchedule(schedule: SignalPhaseSpec[]): SignalPhaseSpec[] {
  return [...schedule].sort((a, b) => a.at_seconds - b.at_seconds);
}

/**
 * The rule declarations to send for a processing run.
 *
 * Built from what the *scene* supports (the server's answer, not the client's
 * preview) intersected with what is actually runnable: red-light additionally needs
 * the run's signal schedule, so it is omitted when none has been entered rather
 * than sent in a form the backend would refuse.
 */
export function rulesForRun(
  supported: ViolationType[],
  schedule: SignalPhaseSpec[],
): Array<Record<string, unknown>> {
  const rules: Array<Record<string, unknown>> = [];
  for (const violation of supported) {
    if (violation === 'red_light_jumping') {
      if (isScheduleUsable(schedule)) {
        rules.push({ kind: 'red_light_jumping', schedule: sortSchedule(schedule) });
      }
      continue;
    }
    if (violation === 'speeding') continue; // no shipped reasoner
    rules.push({ kind: violation });
  }
  return rules;
}

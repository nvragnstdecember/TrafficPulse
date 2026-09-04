import {
  type SceneDraft,
  type ScenePoint,
  type SceneSummary,
  type SignalPhaseSpec,
  type SignalState,
  type StoredScene,
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

/**
 * The site thresholds an analyst may set, in the units they are read in.
 *
 * Deliberately few, and deliberately the same four the backend's `RuleTuning`
 * exposes — most rule parameters are policy constants that should not vary per
 * camera. An empty field means "use the provisional default", which is why every
 * value is optional rather than pre-filled with a number the operator did not choose.
 */
export interface TuningInput {
  /** Seconds a vehicle may dwell in the no-stopping zone before it is a violation. */
  stationaryDurationSeconds?: number;
  /** Degrees off the legal heading that count as opposing traffic. */
  headingDeviationMaxDegrees?: number;
  /** Seconds of sustained opposition before wrong-way confirms. */
  wrongWayMinPersistenceSeconds?: number;
  /** Debounce after a stop-line crossing before red-light confirms. */
  redLightMinPersistenceSeconds?: number;
}

export const EMPTY_TUNING: TuningInput = {};

/**
 * The provisional defaults the backend applies when a field is left blank.
 *
 * Shown as placeholders so an analyst can see what will be used without the number
 * being submitted as though they had chosen it. Mirrors
 * `trafficpulse.scenes.builder`'s `DEFAULT_*` constants; the backend remains the
 * authority and nothing here is sent unless the analyst types it.
 */
export const TUNING_DEFAULTS = {
  stationaryDurationSeconds: 5,
  headingDeviationMaxDegrees: 120,
  wrongWayMinPersistenceSeconds: 1,
  redLightMinPersistenceSeconds: 0.4,
} as const;

/** A tuning value must be a positive number when present; blank means "default". */
export function tuningErrors(tuning: TuningInput): string[] {
  const errors: string[] = [];
  const check = (value: number | undefined, label: string, max: number): void => {
    if (value === undefined) return;
    if (!Number.isFinite(value) || value <= 0) errors.push(`${label} must be greater than 0.`);
    else if (value > max) errors.push(`${label} must be at most ${max}.`);
  };
  check(tuning.stationaryDurationSeconds, 'Stopping dwell threshold', 3600);
  check(tuning.headingDeviationMaxDegrees, 'Heading deviation', 180);
  check(tuning.wrongWayMinPersistenceSeconds, 'Wrong-way persistence', 600);
  check(tuning.redLightMinPersistenceSeconds, 'Red-light debounce', 30);
  return errors;
}

export interface BuildDraftInput {
  shapes: CalibrationShapes;
  frameWidth: number;
  frameHeight: number;
  cameraId: string;
  sceneName: string;
  /** Operator-chosen site thresholds; omitted fields take the provisional defaults. */
  tuning?: TuningInput;
  /**
   * The operator's written declaration of what this scene is.
   *
   * Stored *in the scene*, so it travels with the geometry: a reviewer resolving an
   * event's `scene_config_hash` months later reads why the zone was drawn where it
   * was, rather than having to be told.
   */
  notes?: string;
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
  tuning,
  notes,
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

  if (notes && notes.trim()) draft.description = notes.trim();
  const tuned = buildTuning(tuning);
  if (tuned) draft.tuning = tuned;

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

/**
 * The `RuleTuning` payload for the draft, or `undefined` when nothing was set.
 *
 * Blank fields are **omitted rather than defaulted**: sending the provisional value
 * would record it as an operator's choice, and a scene that says "the analyst chose
 * 5 seconds" when nobody typed anything is a small lie the hash would then preserve
 * forever.
 */
export function buildTuning(tuning: TuningInput | undefined): SceneDraft['tuning'] {
  if (!tuning) return undefined;
  const payload: NonNullable<SceneDraft['tuning']> = {};
  if (tuning.stationaryDurationSeconds !== undefined) {
    payload.stationary_duration_seconds = tuning.stationaryDurationSeconds;
  }
  if (tuning.headingDeviationMaxDegrees !== undefined) {
    payload.heading_deviation_max_degrees = tuning.headingDeviationMaxDegrees;
  }
  if (tuning.wrongWayMinPersistenceSeconds !== undefined) {
    payload.wrong_way_min_persistence_seconds = tuning.wrongWayMinPersistenceSeconds;
  }
  if (tuning.redLightMinPersistenceSeconds !== undefined) {
    payload.red_light_min_persistence_seconds = tuning.redLightMinPersistenceSeconds;
  }
  return Object.keys(payload).length > 0 ? payload : undefined;
}

// --- reloading a saved calibration --------------------------------------------------
/**
 * Rebuild the drawing surface from a stored scene revision.
 *
 * Reproducibility, not convenience. Before this, a saved calibration was invisible:
 * refresh the page and the shapes were gone, so a reviewer could not see what had
 * been drawn, could not correct one polygon without redrawing all of them, and had no
 * way to check that a stored revision matches what the demo claims. Reading the
 * geometry back from the revision the video is actually bound to closes that.
 *
 * Zones are matched by the ids this module authors. A scene drawn elsewhere (an
 * auto-derived one, or a hand-written YAML) may use other ids, and then the tools
 * simply come back empty rather than half-populated with shapes the analyst cannot
 * account for — an honest "there is nothing here I can redraw".
 */
export function sceneToShapes(scene: StoredScene | null | undefined): CalibrationShapes {
  if (!scene) return EMPTY_SHAPES;
  const polygonFor = (zoneId: string): ScenePoint[] =>
    scene.zones.find((zone) => zone.zone_id === zoneId)?.polygon ?? [];

  const direction = scene.legal_directions[0];
  const lane = polygonFor(ZONE_IDS.lane);
  const stopLine = scene.stop_lines[0];
  const roi = scene.signal_groups[0]?.roi;
  const junction = polygonFor(ZONE_IDS.junction);

  return {
    lane,
    noStopping: polygonFor(ZONE_IDS.noStopping),
    junction,
    // The saved ROI stands in for the drawn signal head only when it is genuinely a
    // separate shape: `buildSceneDraft` reuses the junction polygon when no head was
    // outlined, and echoing that back would invent a drawing nobody made.
    signalRoi:
      roi?.polygon && !samePolygon(roi.polygon, junction) ? roi.polygon : [],
    // A stored direction is a unit vector with no anchor, so the arrow is redrawn
    // from the lane's centre — the same direction, placed where it reads.
    direction: direction ? vectorSegment(centroid(lane), direction.vector) : null,
    stopLine: stopLine ? { from: stopLine.endpoints.a, to: stopLine.endpoints.b } : null,
  };
}

/** The tuning values a stored scene actually carries, for repopulating the form. */
export function sceneToTuning(scene: StoredScene | null | undefined): TuningInput {
  if (!scene) return EMPTY_TUNING;
  const value = (violation: ViolationType, id: string): number | undefined => {
    const block = scene.rule_parameters.find((b) => b.violation_type === violation);
    const parameter = block?.parameters.find((p) => p.id === id);
    return parameter?.value ?? undefined;
  };
  const tuning: TuningInput = {};
  const dwell = value('illegal_stopping', 'stationary_duration');
  if (dwell !== undefined) tuning.stationaryDurationSeconds = dwell;
  const heading = value('wrong_way', 'heading_deviation_max');
  if (heading !== undefined) tuning.headingDeviationMaxDegrees = heading;
  const wrongWay = value('wrong_way', 'min_persistence');
  if (wrongWay !== undefined) tuning.wrongWayMinPersistenceSeconds = wrongWay;
  const redLight = value('red_light_jumping', 'min_persistence');
  if (redLight !== undefined) tuning.redLightMinPersistenceSeconds = redLight;
  return tuning;
}

function samePolygon(a: ScenePoint[], b: ScenePoint[]): boolean {
  if (a.length !== b.length) return false;
  return a.every(([x, y], index) => x === b[index][0] && y === b[index][1]);
}

/** An arrow of readable length, drawn from `anchor` along a unit vector. */
function vectorSegment(
  anchor: ScenePoint | undefined,
  vector: { dx: number; dy: number },
  length = 80,
): Segment | null {
  if (!anchor) return null;
  return {
    from: anchor,
    to: [anchor[0] + vector.dx * length, anchor[1] + vector.dy * length],
  };
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

/** One constant stretch of the declared signal, in media seconds. */
export interface ScheduleSegment {
  from: number;
  to: number;
  state: SignalState;
  /** Fraction of the clip this segment covers, for laying out a timeline bar. */
  fraction: number;
}

/**
 * The declared schedule as a step function over the clip's duration.
 *
 * What a timeline bar draws. Two honest details it does not gloss over:
 *
 * - a schedule that does not begin at 0 leaves the head of the clip **unknown**,
 *   because `signal_state_at` resolves anything before the first phase to
 *   `unknown` and no rule may confirm on it. Showing that gap is the point — an
 *   operator who forgot the opening phase can see it rather than discover it as a
 *   silent non-detection;
 * - a phase declared past the end of the clip is kept but clamped, so a typo is
 *   visible instead of vanishing.
 *
 * Returns `[]` for a zero/unknown duration rather than inventing one.
 */
export function scheduleSegments(
  schedule: SignalPhaseSpec[],
  durationSeconds: number,
): ScheduleSegment[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return [];
  const ordered = sortSchedule(schedule).filter((phase) => phase.at_seconds < durationSeconds);
  const segments: ScheduleSegment[] = [];

  const push = (from: number, to: number, state: SignalState): void => {
    if (to <= from) return;
    segments.push({ from, to, state, fraction: (to - from) / durationSeconds });
  };

  if (ordered.length === 0 || ordered[0].at_seconds > 0) {
    push(0, ordered.length > 0 ? ordered[0].at_seconds : durationSeconds, 'unknown');
  }
  ordered.forEach((phase, index) => {
    const next = ordered[index + 1];
    push(
      Math.max(phase.at_seconds, 0),
      next ? next.at_seconds : durationSeconds,
      phase.state,
    );
  });
  return segments;
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


/** What to tell the analyst about a scene nobody drew. */
export interface DerivedSceneNotice {
  title: string;
  body: string;
  /** Whether a legal direction actually came out of the derivation. */
  hasDirection: boolean;
}

/**
 * The caveat a derived scene carries, or `null` for one an analyst drew.
 *
 * Automatic derivation has two honest outcomes and they must not be described the
 * same way. When the clip's traffic is coherent, a legal direction was *estimated*
 * and wrong-way runs against an estimate the analyst should check. When it is not
 * — a two-way road whose movers cancel, or too few vehicles to be evidence — the
 * derivation **abstained**, and saying a direction was estimated would claim
 * something that did not happen. `has_legal_direction` is the backend's own answer
 * to which case this is, so it is what decides the wording rather than a second
 * guess made here.
 *
 * Neither outcome ever invents a no-stopping zone, stop line or signal timing, so
 * both say so: an absent violation is a fact about the footage, not a gap.
 */
export function derivedSceneNotice(scene: SceneSummary | null | undefined): DerivedSceneNotice | null {
  if (!scene?.derived) return null;
  if (scene.has_legal_direction) {
    return {
      hasDirection: true,
      title: 'Scene derived automatically — check the lane',
      body:
        "The frame size was measured and the legal travel direction estimated from this clip's own traffic. " +
        'Nothing unobservable was assumed: there is no no-stopping zone, stop line or signal timing. ' +
        'Wrong-way results rest on that estimate — draw the lane yourself before relying on them.',
    };
  }
  return {
    hasDirection: false,
    title: 'Scene derived automatically — no legal direction could be established',
    body:
      "The frame size was measured, but this clip's traffic does not define a single legal direction " +
      '(for example a two-way road, or too few moving vehicles to be evidence), so none was inferred. ' +
      'Wrong-way detection is therefore unavailable until you draw the lane and its direction. ' +
      'No no-stopping zone, stop line or signal timing was assumed either.',
  };
}

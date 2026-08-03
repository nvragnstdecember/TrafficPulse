import {
  type ConfidenceBreakdown,
  type ConfirmedEvent,
  type EventSummary,
  type ReviewStatus,
  type ViolationType,
} from '@/api/types';

// Value-only import; `review` imports nothing but types from here, so there is
// no runtime cycle.
import { ALL_REVIEW_STATUSES } from './review';

/**
 * Workspace domain models + pure logic (H7C).
 *
 * The video workspace works with strongly-typed view-models derived from the H7A
 * wire types — never the raw summaries in the UI. Every transformation here
 * (media-time mapping, marker building, clustering, filtering, sorting, clock
 * formatting) is a pure function, so the interactive behaviour is unit-testable
 * without React or the network.
 */

// Media time: the backend anchors PTS at the Unix epoch, so an event's
// `trigger_at` (an epoch-anchored ISO timestamp) maps directly to a position on
// the uploaded video's own 0..duration timeline.
export function eventMediaSeconds(triggerAt: string): number {
  const ms = Date.parse(triggerAt);
  return Number.isNaN(ms) ? 0 : Math.max(0, ms / 1000);
}

export type ViolationTone = 'success' | 'warning' | 'error' | 'info' | 'neutral';

export const VIOLATION_LABELS: Record<ViolationType, string> = {
  no_helmet: 'No helmet',
  triple_riding: 'Triple riding',
  red_light_jumping: 'Red-light jumping',
  wrong_way: 'Wrong way',
  illegal_stopping: 'Illegal stopping',
  speeding: 'Speeding',
};

const VIOLATION_TONES: Record<ViolationType, ViolationTone> = {
  no_helmet: 'error',
  triple_riding: 'warning',
  red_light_jumping: 'error',
  wrong_way: 'warning',
  illegal_stopping: 'info',
  speeding: 'warning',
};

/** Human label for a violation, with a graceful fallback for unknown values. */
export function violationLabel(type: string): string {
  if (type in VIOLATION_LABELS) return VIOLATION_LABELS[type as ViolationType];
  return type.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

export function violationTone(type: string): ViolationTone {
  return type in VIOLATION_TONES ? VIOLATION_TONES[type as ViolationType] : 'neutral';
}

export const ALL_VIOLATION_TYPES: ViolationType[] = [
  'no_helmet',
  'triple_riding',
  'red_light_jumping',
  'wrong_way',
  'illegal_stopping',
  'speeding',
];

// --- severity (H7E) ------------------------------------------------------------
export type ViolationSeverity = 'high' | 'medium' | 'low';

// Safety-criticality ranking used for the severity badge and the severity sort.
// Higher rank = more severe (sorts first in a severity-descending order).
const VIOLATION_SEVERITY: Record<ViolationType, ViolationSeverity> = {
  no_helmet: 'high',
  red_light_jumping: 'high',
  wrong_way: 'medium',
  speeding: 'medium',
  triple_riding: 'medium',
  illegal_stopping: 'low',
};

const SEVERITY_RANK: Record<ViolationSeverity, number> = { high: 3, medium: 2, low: 1 };
const SEVERITY_LABELS: Record<ViolationSeverity, string> = {
  high: 'High',
  medium: 'Medium',
  low: 'Low',
};
const SEVERITY_TONES: Record<ViolationSeverity, ViolationTone> = {
  high: 'error',
  medium: 'warning',
  low: 'info',
};

export function violationSeverity(type: string): ViolationSeverity {
  return type in VIOLATION_SEVERITY ? VIOLATION_SEVERITY[type as ViolationType] : 'low';
}

const VIOLATION_DESCRIPTIONS: Record<ViolationType, string> = {
  no_helmet: 'A rider was detected travelling without a helmet.',
  triple_riding: 'More than two riders were detected on a single two-wheeler.',
  red_light_jumping: 'A vehicle crossed the stop line while the signal was red.',
  wrong_way: 'A vehicle travelled against the lane’s legal direction of flow.',
  illegal_stopping: 'A vehicle remained stopped inside a no-stopping zone.',
  speeding: 'A vehicle exceeded the posted speed threshold for the zone.',
};

/** A one-line, human explanation of what a violation means. */
export function violationDescription(type: string): string {
  if (type in VIOLATION_DESCRIPTIONS) return VIOLATION_DESCRIPTIONS[type as ViolationType];
  return `A ${violationLabel(type).toLowerCase()} violation was confirmed.`;
}

export function severityRank(type: string): number {
  return SEVERITY_RANK[violationSeverity(type)];
}

export function severityLabel(severity: ViolationSeverity): string {
  return SEVERITY_LABELS[severity];
}

export function severityTone(severity: ViolationSeverity): ViolationTone {
  return SEVERITY_TONES[severity];
}

export interface WorkspaceEvent {
  id: string;
  videoId: string;
  jobId: string;
  violationType: ViolationType;
  cameraId: string;
  trackIds: string[];
  startAt: string;
  triggerAt: string;
  /** Position on the video timeline, in seconds (the trigger instant). */
  mediaSeconds: number;
  /** Where support began accruing, in seconds on the same timeline. */
  startSeconds: number;
  /** How long support was sustained before confirmation, in seconds. */
  observationSeconds: number;
  ruleId: string;
  /** The headline confidence 0..1, or null when the rule measured none. */
  confidence: number | null;
  /** Which component {@link confidence} came from, for honest labelling. */
  confidenceSource: ConfidenceSource | null;
  /** Every published component, for the detail panel. */
  confidenceComponents: ConfidenceComponent[];
  /** Derived lane locator when the detail carries one, else null. */
  lane: string | null;
  /** Analyst-review state, folded server-side from the event's review journal. */
  reviewStatus: ReviewStatus;
}

// --- confidence ----------------------------------------------------------------
export type ConfidenceSource =
  | 'aggregate'
  | 'classifier'
  | 'temporal_consistency'
  | 'association'
  | 'detector'
  | 'geometric_margin';

export interface ConfidenceComponent {
  key: ConfidenceSource | 'track_continuity' | 'calibration_quality';
  label: string;
  value: number;
}

export const CONFIDENCE_LABELS: Record<ConfidenceComponent['key'], string> = {
  aggregate: 'Aggregate',
  classifier: 'Classifier',
  temporal_consistency: 'Temporal consistency',
  association: 'Association',
  detector: 'Detector',
  geometric_margin: 'Geometric margin',
  track_continuity: 'Track continuity',
  calibration_quality: 'Calibration quality',
};

/**
 * Order in which a component is promoted to *the* headline confidence.
 *
 * `aggregate` first because a backend that has demonstrated calibration would
 * publish it, and it would then be the right single number. It is null in this
 * system today, so in practice the headline is the strongest evidential component
 * the rule actually measured — for no-helmet that is the classifier's mean score
 * across the supporting observations.
 */
const CONFIDENCE_PRIORITY: ConfidenceSource[] = [
  'aggregate',
  'classifier',
  'geometric_margin',
  'temporal_consistency',
  'association',
  'detector',
];

const COMPONENT_ORDER: ConfidenceComponent['key'][] = [
  'aggregate',
  'classifier',
  'temporal_consistency',
  'association',
  'detector',
  'geometric_margin',
  'track_continuity',
  'calibration_quality',
];

function readComponent(
  breakdown: ConfidenceBreakdown | undefined,
  key: ConfidenceComponent['key'],
): number | null {
  const value = breakdown?.[key];
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : null;
}

/** Every measured component, in a stable display order. */
export function confidenceComponents(
  breakdown: ConfidenceBreakdown | undefined,
): ConfidenceComponent[] {
  const out: ConfidenceComponent[] = [];
  for (const key of COMPONENT_ORDER) {
    const value = readComponent(breakdown, key);
    if (value !== null) out.push({ key, label: CONFIDENCE_LABELS[key], value });
  }
  return out;
}

/**
 * The one component to headline, with its provenance — never a blend.
 *
 * Returns null when the rule measured nothing, so the UI can say "not measured"
 * instead of rendering a fabricated 0%.
 */
export function headlineConfidence(
  breakdown: ConfidenceBreakdown | undefined,
): { value: number; source: ConfidenceSource } | null {
  for (const source of CONFIDENCE_PRIORITY) {
    const value = readComponent(breakdown, source);
    if (value !== null) return { value, source };
  }
  return null;
}

function extractLane(detail: ConfirmedEvent | undefined): string | null {
  if (!detail) return null;
  const laneMeasure = detail.measurements.find((m) => /lane/i.test(m.name));
  return laneMeasure ? String(laneMeasure.value) : null;
}

/** Build a workspace view-model from a summary, enriched by an optional detail. */
export function toWorkspaceEvent(summary: EventSummary, detail?: ConfirmedEvent): WorkspaceEvent {
  // The summary already carries the window and the confidence components, so a
  // list row is fully informative without fetching each event's detail. The
  // detail, when present, only adds what the summary genuinely lacks (the lane).
  const breakdown = detail?.confidence ?? summary.confidence;
  const headline = headlineConfidence(breakdown);
  const startSeconds = eventMediaSeconds(summary.start_at);
  const mediaSeconds = eventMediaSeconds(summary.trigger_at);
  return {
    id: summary.event_id,
    videoId: summary.video_id,
    jobId: summary.job_id,
    violationType: summary.violation_type,
    cameraId: summary.camera_id,
    trackIds: summary.track_ids,
    startAt: summary.start_at,
    triggerAt: summary.trigger_at,
    mediaSeconds,
    startSeconds,
    observationSeconds: Math.max(0, mediaSeconds - startSeconds),
    ruleId: summary.rule_id,
    confidence: headline?.value ?? null,
    confidenceSource: headline?.source ?? null,
    confidenceComponents: confidenceComponents(breakdown),
    lane: extractLane(detail),
    reviewStatus: summary.review_status,
  };
}

/** Structural equality over the fields that affect rendering (H7D). */
export function workspaceEventsEqual(a: WorkspaceEvent, b: WorkspaceEvent): boolean {
  return (
    a.id === b.id &&
    a.violationType === b.violationType &&
    a.mediaSeconds === b.mediaSeconds &&
    a.startSeconds === b.startSeconds &&
    a.confidence === b.confidence &&
    a.confidenceSource === b.confidenceSource &&
    a.confidenceComponents.length === b.confidenceComponents.length &&
    a.confidenceComponents.every(
      (component, index) =>
        component.key === b.confidenceComponents[index].key &&
        component.value === b.confidenceComponents[index].value,
    ) &&
    a.lane === b.lane &&
    a.reviewStatus === b.reviewStatus &&
    a.cameraId === b.cameraId &&
    a.ruleId === b.ruleId &&
    a.trackIds.length === b.trackIds.length &&
    a.trackIds.every((track, index) => track === b.trackIds[index])
  );
}

/**
 * Merge a freshly-fetched event set into the previous one, preserving references
 * (H7D).
 *
 * Live polling refetches the whole list each tick; returning brand-new objects
 * every time would rerender every row and defeat memoization. This keeps each
 * prior {@link WorkspaceEvent} reference when its content is unchanged, and
 * returns the *previous array itself* when nothing changed at all — so appends
 * (new events arriving mid-processing) update only what moved, existing rows and
 * the current selection are preserved, and an identical poll causes no rerender.
 */
export function mergeWorkspaceEvents(
  previous: WorkspaceEvent[],
  incoming: WorkspaceEvent[],
): WorkspaceEvent[] {
  const priorById = new Map(previous.map((event) => [event.id, event]));
  const merged = incoming.map((event) => {
    const prior = priorById.get(event.id);
    return prior && workspaceEventsEqual(prior, event) ? prior : event;
  });
  if (merged.length === previous.length && merged.every((event, i) => event === previous[i])) {
    return previous;
  }
  return merged;
}

// --- timeline markers ----------------------------------------------------------
export interface TimelineMarker {
  id: string;
  time: number;
  /** 0..1 across the timeline for positioning. */
  positionRatio: number;
  violationType: ViolationType;
  event: WorkspaceEvent;
}

export function clampRatio(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/** Effective timeline length: the player duration, else the latest event. */
export function timelineDuration(
  playerDuration: number | null | undefined,
  events: WorkspaceEvent[],
): number {
  if (playerDuration && Number.isFinite(playerDuration) && playerDuration > 0) {
    return playerDuration;
  }
  const latest = events.reduce((max, event) => Math.max(max, event.mediaSeconds), 0);
  return latest > 0 ? latest : 0;
}

export function buildTimelineMarkers(events: WorkspaceEvent[], duration: number): TimelineMarker[] {
  return events.map((event) => ({
    id: event.id,
    time: event.mediaSeconds,
    positionRatio: duration > 0 ? clampRatio(event.mediaSeconds / duration) : 0,
    violationType: event.violationType,
    event,
  }));
}

export interface MarkerCluster {
  key: string;
  positionRatio: number;
  time: number;
  markers: TimelineMarker[];
}

/**
 * Group markers that would visually overlap (positions within `thresholdRatio`)
 * so the timeline can render a single badge for a dense cluster.
 */
export function clusterMarkers(markers: TimelineMarker[], thresholdRatio = 0.02): MarkerCluster[] {
  const sorted = [...markers].sort((a, b) => a.positionRatio - b.positionRatio);
  const clusters: MarkerCluster[] = [];
  for (const marker of sorted) {
    const last = clusters[clusters.length - 1];
    if (last && Math.abs(marker.positionRatio - last.positionRatio) <= thresholdRatio) {
      last.markers.push(marker);
    } else {
      clusters.push({
        key: marker.id,
        positionRatio: marker.positionRatio,
        time: marker.time,
        markers: [marker],
      });
    }
  }
  return clusters;
}

// --- filtering + sorting -------------------------------------------------------
export interface EventFilters {
  query: string;
  violationTypes: ViolationType[];
  /** Minimum confidence 0..1; 0 disables the filter. */
  minConfidence: number;
  /** Maximum confidence 0..1; 1 disables the filter. */
  maxConfidence: number;
  /** Earliest media-time second to include; 0 disables the filter. */
  fromSeconds: number;
  /** Latest media-time second to include; null disables the filter. */
  toSeconds: number | null;
  /** Review states to include; empty means every state (H9). */
  reviewStatuses: ReviewStatus[];
}

export const DEFAULT_EVENT_FILTERS: EventFilters = {
  query: '',
  violationTypes: [],
  minConfidence: 0,
  maxConfidence: 1,
  fromSeconds: 0,
  toSeconds: null,
  reviewStatuses: [],
};

export function hasActiveFilters(filters: EventFilters): boolean {
  return (
    filters.query.trim().length > 0 ||
    filters.violationTypes.length > 0 ||
    filters.minConfidence > 0 ||
    filters.maxConfidence < 1 ||
    filters.fromSeconds > 0 ||
    filters.toSeconds !== null ||
    filters.reviewStatuses.length > 0
  );
}

function normalizeMembers<T extends string>(value: unknown, allowed: readonly T[]): T[] {
  if (!Array.isArray(value)) return [];
  const permitted = new Set<string>(allowed);
  return value.filter((item): item is T => typeof item === 'string' && permitted.has(item));
}

function normalizeNumber(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

/**
 * Coerce an arbitrary value into filters that are safe to render.
 *
 * `EventFilters` has gained fields over time (H7C shipped `query` /
 * `violationTypes` / `minConfidence`; the review workspace added the confidence
 * ceiling and the time range; H9 added `reviewStatuses`), and these filters are
 * persisted to browser storage. A rehydrated blob is therefore whatever *some
 * past version* of the app wrote, so it cannot be assumed to have today's
 * shape — a field added after the blob was written is simply absent.
 *
 * Every field is taken only when it is present and well-typed, and falls back
 * to its default otherwise. Enum members that are no longer recognised are
 * dropped rather than kept: a stale violation type or review state is not inert
 * — a non-empty filter set narrows the list, so keeping one would silently hide
 * every event with no way for an analyst to see why.
 */
export function normalizeEventFilters(value: unknown): EventFilters {
  const raw = (typeof value === 'object' && value !== null ? value : {}) as Record<string, unknown>;
  const toSeconds =
    typeof raw.toSeconds === 'number' && Number.isFinite(raw.toSeconds)
      ? Math.max(0, raw.toSeconds)
      : DEFAULT_EVENT_FILTERS.toSeconds;
  return {
    query: typeof raw.query === 'string' ? raw.query : DEFAULT_EVENT_FILTERS.query,
    violationTypes: normalizeMembers(raw.violationTypes, ALL_VIOLATION_TYPES),
    minConfidence: normalizeNumber(raw.minConfidence, DEFAULT_EVENT_FILTERS.minConfidence, 0, 1),
    maxConfidence: normalizeNumber(raw.maxConfidence, DEFAULT_EVENT_FILTERS.maxConfidence, 0, 1),
    fromSeconds: normalizeNumber(
      raw.fromSeconds,
      DEFAULT_EVENT_FILTERS.fromSeconds,
      0,
      Number.MAX_SAFE_INTEGER,
    ),
    toSeconds,
    reviewStatuses: normalizeMembers(raw.reviewStatuses, ALL_REVIEW_STATUSES),
  };
}

/**
 * Parse a clock-ish search term into seconds, or null if it is not one.
 *
 * Accepts `1:23`, `01:23`, `1:02:03`, and a bare `90`. Lets an analyst who is
 * reading a timestamp off the video paste it straight into the search box.
 */
export function parseClockQuery(query: string): number | null {
  const trimmed = query.trim();
  if (!/^\d{1,2}(:\d{1,2}){0,2}$/.test(trimmed)) return null;
  const parts = trimmed.split(':').map(Number);
  if (parts.some((part) => !Number.isFinite(part))) return null;
  const seconds = parts.reduce((total, part) => total * 60 + part, 0);
  return Number.isFinite(seconds) ? seconds : null;
}

/** How close (seconds) a timestamp search has to be to count as a hit. */
export const CLOCK_QUERY_TOLERANCE_SECONDS = 1.5;

function matchesQuery(event: WorkspaceEvent, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;

  // A timestamp term matches the event's *window*, not just its trigger: an
  // analyst typing the moment they saw something on screen means "the violation
  // that was happening then", which starts before it is confirmed.
  const clock = parseClockQuery(needle);
  if (clock !== null) {
    const from = event.startSeconds - CLOCK_QUERY_TOLERANCE_SECONDS;
    const to = event.mediaSeconds + CLOCK_QUERY_TOLERANCE_SECONDS;
    if (clock >= from && clock <= to) return true;
    // fall through: a bare number may still be part of a track id
  }

  const haystack = [
    event.id,
    event.cameraId,
    event.ruleId,
    violationLabel(event.violationType),
    formatClock(event.mediaSeconds),
    event.reviewStatus,
    ...event.trackIds,
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(needle);
}

export function filterWorkspaceEvents(
  events: WorkspaceEvent[],
  filters: EventFilters,
): WorkspaceEvent[] {
  const violationSet = new Set(filters.violationTypes);
  const reviewSet = new Set(filters.reviewStatuses);
  return events.filter((event) => {
    if (!matchesQuery(event, filters.query)) return false;
    if (violationSet.size > 0 && !violationSet.has(event.violationType)) return false;
    if (filters.minConfidence > 0 || filters.maxConfidence < 1) {
      // An unmeasured confidence is excluded by a *narrowed* range rather than
      // treated as zero — the filter asks about evidence, and "not measured" is
      // not a low score.
      if (event.confidence === null) return false;
      if (event.confidence < filters.minConfidence) return false;
      if (event.confidence > filters.maxConfidence) return false;
    }
    if (filters.fromSeconds > 0 && event.mediaSeconds < filters.fromSeconds) return false;
    if (filters.toSeconds !== null && event.mediaSeconds > filters.toSeconds) return false;
    if (reviewSet.size > 0 && !reviewSet.has(event.reviewStatus)) return false;
    return true;
  });
}

export type WorkspaceSort =
  'time-asc' | 'time-desc' | 'confidence-desc' | 'severity-desc' | 'violation';

export const WORKSPACE_SORTS: Array<{ value: WorkspaceSort; label: string }> = [
  { value: 'time-asc', label: 'Earliest first' },
  { value: 'time-desc', label: 'Latest first' },
  { value: 'severity-desc', label: 'Severity' },
  { value: 'confidence-desc', label: 'Confidence' },
  { value: 'violation', label: 'Violation type' },
];

export const DEFAULT_WORKSPACE_SORT: WorkspaceSort = 'time-asc';

/** Coerce an arbitrary value into a sort the list (and its label) understands. */
export function normalizeWorkspaceSort(value: unknown): WorkspaceSort {
  return WORKSPACE_SORTS.some((option) => option.value === value)
    ? (value as WorkspaceSort)
    : DEFAULT_WORKSPACE_SORT;
}

export function sortWorkspaceEvents(
  events: WorkspaceEvent[],
  sort: WorkspaceSort,
): WorkspaceEvent[] {
  const copy = [...events];
  switch (sort) {
    case 'time-asc':
      return copy.sort((a, b) => a.mediaSeconds - b.mediaSeconds || a.id.localeCompare(b.id));
    case 'time-desc':
      return copy.sort((a, b) => b.mediaSeconds - a.mediaSeconds || a.id.localeCompare(b.id));
    case 'confidence-desc':
      return copy.sort(
        (a, b) => (b.confidence ?? -1) - (a.confidence ?? -1) || a.id.localeCompare(b.id),
      );
    case 'severity-desc':
      return copy.sort(
        (a, b) =>
          severityRank(b.violationType) - severityRank(a.violationType) ||
          a.mediaSeconds - b.mediaSeconds ||
          a.id.localeCompare(b.id),
      );
    case 'violation':
      return copy.sort(
        (a, b) =>
          violationLabel(a.violationType).localeCompare(violationLabel(b.violationType)) ||
          a.mediaSeconds - b.mediaSeconds,
      );
  }
}

// --- event narrative (how the violation evolved) --------------------------------
export type NarrativeStage = 'observation' | 'accumulation' | 'confirmation' | 'evidence';

export interface NarrativeStep {
  key: string;
  /** Media-time position of this step, in seconds. */
  seconds: number;
  stage: NarrativeStage;
  title: string;
  detail: string | null;
}

/**
 * The story of one violation, reconstructed from what the reasoner published.
 *
 * Reads the event's own window (`start_at` → `trigger_at`), its threshold, and its
 * evidence manifest. It derives nothing the backend did not decide: the midpoint
 * step is labelled as elapsed progress toward the rule's own bar, and the evidence
 * step appears only when a manifest actually exists. Where a value is absent the
 * step is omitted rather than invented, which is why a rule that publishes no
 * threshold yields a three-step story instead of a padded four.
 */
export function buildEventNarrative(
  event: WorkspaceEvent,
  options?: { thresholdSeconds?: number | null; hasEvidence?: boolean },
): NarrativeStep[] {
  const steps: NarrativeStep[] = [];
  const threshold = options?.thresholdSeconds ?? null;

  steps.push({
    key: 'observation',
    seconds: event.startSeconds,
    stage: 'observation',
    title: `${violationLabel(event.violationType)} observation begins`,
    detail:
      event.trackIds.length > 0 ? `Track ${event.trackIds.join(', ')} enters observation` : null,
  });

  if (event.observationSeconds > 0) {
    const midpoint = event.startSeconds + event.observationSeconds / 2;
    steps.push({
      key: 'accumulation',
      seconds: midpoint,
      stage: 'accumulation',
      title: 'Supporting evidence accumulates',
      detail:
        threshold && threshold > 0
          ? `${(event.observationSeconds / 2).toFixed(2)}s of ${threshold.toFixed(2)}s required`
          : `${(event.observationSeconds / 2).toFixed(2)}s sustained so far`,
    });
  }

  steps.push({
    key: 'confirmation',
    seconds: event.mediaSeconds,
    stage: 'confirmation',
    title: 'Violation confirmed',
    detail:
      event.confidence !== null && event.confidenceSource
        ? `${CONFIDENCE_LABELS[event.confidenceSource]} ${Math.round(event.confidence * 100)}% · sustained ${event.observationSeconds.toFixed(2)}s`
        : `Sustained ${event.observationSeconds.toFixed(2)}s`,
  });

  if (options?.hasEvidence) {
    steps.push({
      key: 'evidence',
      seconds: event.mediaSeconds,
      stage: 'evidence',
      title: 'Evidence package finalized',
      detail: 'Frames, rule trace, and model provenance recorded',
    });
  }

  return steps;
}

// --- review playback window ------------------------------------------------------
/** Seconds of lead-in before an event's observation window when auto-playing. */
export const REVIEW_LEAD_IN_SECONDS = 1.5;
/** Seconds to keep playing past the confirmation instant before stopping. */
export const REVIEW_TAIL_SECONDS = 2;

export interface PlaybackWindow {
  /** Where to seek before playing. */
  from: number;
  /** Where playback should stop. */
  to: number;
}

/**
 * The clip an analyst should see when they pick a violation.
 *
 * Starts a beat before support began accruing — the confirmation instant alone is
 * the *end* of the story, so seeking there and playing shows the aftermath rather
 * than the behaviour — and stops shortly after the trigger so review does not
 * drift into unrelated footage. Clamped to the media so a violation near either
 * end of the clip still yields a valid range.
 */
export function reviewWindow(event: WorkspaceEvent, duration?: number | null): PlaybackWindow {
  const limit = duration && Number.isFinite(duration) && duration > 0 ? duration : null;
  const from = Math.max(0, event.startSeconds - REVIEW_LEAD_IN_SECONDS);
  const rawTo = event.mediaSeconds + REVIEW_TAIL_SECONDS;
  const to = limit === null ? rawTo : Math.min(limit, rawTo);
  return { from, to: Math.max(from, to) };
}

// --- clock ---------------------------------------------------------------------
/** Format seconds as a media clock: `m:ss`, or `h:mm:ss` past an hour. */
export function formatClock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds);
  const s = total % 60;
  const m = Math.floor(total / 60) % 60;
  const h = Math.floor(total / 3600);
  const pad = (n: number) => n.toString().padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

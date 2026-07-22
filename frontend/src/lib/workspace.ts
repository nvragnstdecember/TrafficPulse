import { type ConfirmedEvent, type EventSummary, type ViolationType } from '@/api/types';

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

export interface WorkspaceEvent {
  id: string;
  videoId: string;
  jobId: string;
  violationType: ViolationType;
  cameraId: string;
  trackIds: string[];
  triggerAt: string;
  /** Position on the video timeline, in seconds. */
  mediaSeconds: number;
  ruleId: string;
  /** 0..1 when a detail record is loaded, else null. */
  confidence: number | null;
  /** Derived lane locator when the detail carries one, else null. */
  lane: string | null;
}

function extractConfidence(breakdown: Record<string, unknown> | undefined): number | null {
  if (!breakdown) return null;
  for (const key of ['overall', 'combined', 'score', 'value']) {
    const candidate = breakdown[key];
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      return Math.min(1, Math.max(0, candidate));
    }
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
  return {
    id: summary.event_id,
    videoId: summary.video_id,
    jobId: summary.job_id,
    violationType: summary.violation_type,
    cameraId: summary.camera_id,
    trackIds: summary.track_ids,
    triggerAt: summary.trigger_at,
    mediaSeconds: eventMediaSeconds(summary.trigger_at),
    ruleId: summary.rule_id,
    confidence: detail ? extractConfidence(detail.confidence) : null,
    lane: extractLane(detail),
  };
}

/** Structural equality over the fields that affect rendering (H7D). */
export function workspaceEventsEqual(a: WorkspaceEvent, b: WorkspaceEvent): boolean {
  return (
    a.id === b.id &&
    a.violationType === b.violationType &&
    a.mediaSeconds === b.mediaSeconds &&
    a.confidence === b.confidence &&
    a.lane === b.lane &&
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
}

export const DEFAULT_EVENT_FILTERS: EventFilters = {
  query: '',
  violationTypes: [],
  minConfidence: 0,
};

export function hasActiveFilters(filters: EventFilters): boolean {
  return (
    filters.query.trim().length > 0 ||
    filters.violationTypes.length > 0 ||
    filters.minConfidence > 0
  );
}

function matchesQuery(event: WorkspaceEvent, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  const haystack = [
    event.id,
    event.cameraId,
    event.ruleId,
    violationLabel(event.violationType),
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
  return events.filter((event) => {
    if (!matchesQuery(event, filters.query)) return false;
    if (violationSet.size > 0 && !violationSet.has(event.violationType)) return false;
    if (filters.minConfidence > 0) {
      if (event.confidence === null) return false;
      if (event.confidence < filters.minConfidence) return false;
    }
    return true;
  });
}

export type WorkspaceSort = 'time-asc' | 'time-desc' | 'confidence-desc' | 'violation';

export const WORKSPACE_SORTS: Array<{ value: WorkspaceSort; label: string }> = [
  { value: 'time-asc', label: 'Earliest first' },
  { value: 'time-desc', label: 'Latest first' },
  { value: 'confidence-desc', label: 'Confidence' },
  { value: 'violation', label: 'Violation type' },
];

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
    case 'violation':
      return copy.sort(
        (a, b) =>
          violationLabel(a.violationType).localeCompare(violationLabel(b.violationType)) ||
          a.mediaSeconds - b.mediaSeconds,
      );
  }
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

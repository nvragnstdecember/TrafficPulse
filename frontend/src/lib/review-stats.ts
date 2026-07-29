import { type EngineMetrics, type JobStatusResponse } from '@/api/types';

import { type WorkspaceEvent, violationLabel } from './workspace';

/**
 * Read-only summary metrics for the review workspace (Phase 2).
 *
 * Pure aggregation over data already fetched — the engine's own `EngineMetrics`
 * snapshot and the confirmed events the list endpoint returned. Nothing here
 * counts, infers, or re-derives anything the backend did not measure.
 *
 * The distinction that matters: a value the engine reports is a **measurement**; a
 * value it does not is `null`, and the UI renders "—". In particular there is no
 * "vehicles processed" counter in the engine — `track_states` is the number of
 * per-frame track observations, not distinct vehicles — so this module exposes it
 * under its real name rather than relabelling it into a number it is not.
 */

export interface StatValue {
  key: string;
  label: string;
  /** Formatted for display, or null when the value was never measured. */
  value: string | null;
  /** Short clarification of what the number actually counts. */
  hint?: string;
}

function integer(value: number | null | undefined): string | null {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.round(value).toLocaleString()
    : null;
}

function decimal(value: number | null | undefined, digits = 1): string | null {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : null;
}

function percent(value: number | null): string | null {
  return value === null ? null : `${Math.round(value * 100)}%`;
}

/** Mean of the events that actually carry a confidence, or null if none do. */
export function averageConfidence(events: WorkspaceEvent[]): number | null {
  const scored = events.filter((event) => event.confidence !== null);
  if (scored.length === 0) return null;
  return scored.reduce((total, event) => total + (event.confidence ?? 0), 0) / scored.length;
}

/** Mean observation duration across events, or null when there are none. */
export function averageObservationSeconds(events: WorkspaceEvent[]): number | null {
  if (events.length === 0) return null;
  return events.reduce((total, event) => total + event.observationSeconds, 0) / events.length;
}

/** Confirmed-event counts per violation type, most frequent first. */
export function violationBreakdown(
  events: WorkspaceEvent[],
): Array<{ type: string; label: string; count: number }> {
  const counts = new Map<string, number>();
  for (const event of events) {
    counts.set(event.violationType, (counts.get(event.violationType) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([type, count]) => ({ type, label: violationLabel(type), count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

/** Distinct tracks implicated across every confirmed event. */
export function distinctTrackCount(events: WorkspaceEvent[]): number {
  const tracks = new Set<string>();
  for (const event of events) for (const track of event.trackIds) tracks.add(track);
  return tracks.size;
}

/**
 * The review dashboard's headline statistics.
 *
 * `metrics` is the engine's snapshot for the run (null before one exists).
 */
export function reviewStats(events: WorkspaceEvent[], metrics: EngineMetrics | null): StatValue[] {
  return [
    {
      key: 'violations',
      label: 'Violations confirmed',
      value: integer(events.length),
    },
    {
      key: 'tracks',
      label: 'Tracks implicated',
      value: integer(distinctTrackCount(events)),
      hint: 'Distinct track ids named on confirmed events',
    },
    {
      key: 'detections',
      label: 'Detections',
      value: integer(metrics?.detections),
      hint: 'Objects detected across all processed frames',
    },
    {
      key: 'track-states',
      label: 'Track observations',
      value: integer(metrics?.track_states),
      hint: 'Per-frame tracked-object states, not distinct vehicles',
    },
    {
      key: 'confidence',
      label: 'Average confidence',
      value: percent(averageConfidence(events)),
      hint: 'Mean of the strongest measured component per event',
    },
    {
      key: 'observation',
      label: 'Average observation',
      value: (() => {
        const mean = averageObservationSeconds(events);
        return mean === null ? null : `${mean.toFixed(2)}s`;
      })(),
      hint: 'Mean time support was sustained before confirmation',
    },
  ];
}

/**
 * The post-run summary shown when processing completes.
 *
 * `models` come from any confirmed event's provenance — every event in a run is
 * stamped with the same run-level model set, so the first one is representative.
 */
export function processingSummary(
  job: JobStatusResponse | null | undefined,
  metrics: EngineMetrics | null,
  elapsedSeconds: number | null,
): StatValue[] {
  return [
    {
      key: 'duration',
      label: 'Duration',
      value: elapsedSeconds === null ? null : `${elapsedSeconds.toFixed(1)}s`,
    },
    {
      key: 'frames',
      label: 'Frames processed',
      value: integer(metrics?.frames_processed ?? job?.frames_processed),
    },
    {
      key: 'fps',
      label: 'Processing FPS',
      value: decimal(metrics?.wall_fps),
      hint: 'Wall-clock frames per second',
    },
    {
      key: 'media-fps',
      label: 'Media FPS',
      value: decimal(metrics?.media_fps),
      hint: 'Frame rate of the source footage',
    },
    {
      key: 'detections',
      label: 'Detections',
      value: integer(metrics?.detections),
    },
    {
      key: 'events',
      label: 'Violations detected',
      value: integer(metrics?.events_confirmed ?? job?.event_count),
    },
  ];
}

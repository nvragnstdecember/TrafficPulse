import { type WorkspaceEvent, formatClock, violationLabel, violationSeverity } from './workspace';

/**
 * Export formatting + download helpers (H7E).
 *
 * Pure serializers plus one guarded browser-download primitive, so the review
 * UI can export the events it already holds without re-implementing anything the
 * backend does. Single-event JSON / evidence-manifest exports reuse the backend's
 * own JSON verbatim (the already-fetched response); CSV and multi-event JSON are
 * a client-side presentation of the in-memory summaries — there is no backend CSV
 * to duplicate.
 */

/** RFC-4180-ish CSV cell quoting (only when needed). */
function csvCell(value: string | number | null | undefined): string {
  const text = value == null ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export const EVENT_CSV_COLUMNS = [
  'event_id',
  'violation_type',
  'severity',
  'media_time',
  'trigger_at',
  'camera_id',
  'rule_id',
  'confidence',
  'lane',
  'track_ids',
] as const;

/** One CSV row (as cells) for a workspace event. */
export function eventCsvRow(event: WorkspaceEvent): string[] {
  return [
    event.id,
    violationLabel(event.violationType),
    violationSeverity(event.violationType),
    formatClock(event.mediaSeconds),
    event.triggerAt,
    event.cameraId,
    event.ruleId,
    event.confidence == null ? '' : event.confidence.toFixed(4),
    event.lane ?? '',
    event.trackIds.join(' '),
  ].map(csvCell);
}

/** A full CSV document (header + one row per event). */
export function eventsToCsv(events: WorkspaceEvent[]): string {
  const header = EVENT_CSV_COLUMNS.join(',');
  const rows = events.map((event) => eventCsvRow(event).join(','));
  return [header, ...rows].join('\r\n');
}

/** A plain JSON view-model for multi-event JSON export (stable field order). */
export function eventsToJsonModel(events: WorkspaceEvent[]): Array<Record<string, unknown>> {
  return events.map((event) => ({
    event_id: event.id,
    video_id: event.videoId,
    job_id: event.jobId,
    violation_type: event.violationType,
    severity: violationSeverity(event.violationType),
    media_seconds: event.mediaSeconds,
    trigger_at: event.triggerAt,
    camera_id: event.cameraId,
    rule_id: event.ruleId,
    confidence: event.confidence,
    lane: event.lane,
    track_ids: event.trackIds,
  }));
}

export function jsonString(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

/** A filesystem-safe filename, e.g. `trafficpulse-events-3.csv`. */
export function exportFilename(base: string, extension: string): string {
  const safe = base.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
  return `${safe || 'trafficpulse'}.${extension}`;
}

/**
 * Trigger a browser download of text content. Guarded so it is a no-op (returning
 * false) in a non-DOM environment; the UI wires this to real export actions.
 */
export function downloadTextFile(
  filename: string,
  content: string,
  mimeType = 'application/octet-stream',
): boolean {
  if (typeof document === 'undefined' || typeof URL === 'undefined') return false;
  if (typeof URL.createObjectURL !== 'function') return false;
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = 'noopener';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    return true;
  } finally {
    URL.revokeObjectURL(url);
  }
}

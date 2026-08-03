import { type AnalyticsSummary, type ViolationCount } from '@/api/types';
import { formatBytes, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { type StatValue } from '@/lib/review-stats';
import { violationLabel } from '@/lib/workspace';

/**
 * Dashboard presentation mapping (H15) — pure, no React, no network.
 *
 * The backend's `AnalyticsService` is the only aggregation layer; this module does
 * no aggregation of its own. It turns an already-computed {@link AnalyticsSummary}
 * into the shapes the existing presentation components consume — {@link StatValue}
 * for `StatGrid`, and normalised rows for the bar chart.
 *
 * The one rule it enforces everywhere: **a value the backend reported as `null` was
 * never measured, and renders as an em dash rather than `0`.** A repository with no
 * timed runs has no average duration; showing "0.0s" would claim runs were
 * instantaneous. This is the same discipline `review-stats.ts` established.
 */

/** A single row of a horizontal bar chart, pre-normalised to a 0..1 fraction. */
export interface BarDatum {
  key: string;
  label: string;
  value: number;
  /** Share of the largest value in the set, for bar width. 0 when the set is empty. */
  fraction: number;
}

/**
 * Bar rows for the violation breakdown, widest first.
 *
 * Normalised against the **largest** count rather than the total: the chart's job
 * is comparing types to each other, and scaling to the total would leave every bar
 * a sliver whenever one type dominates.
 */
export function violationBars(counts: ViolationCount[] | undefined | null): BarDatum[] {
  if (!counts || counts.length === 0) return [];
  const largest = counts.reduce((max, item) => Math.max(max, item.count), 0);
  return counts
    .map((item) => ({
      key: item.violation_type,
      label: violationLabel(item.violation_type),
      value: item.count,
      fraction: largest > 0 ? item.count / largest : 0,
    }))
    .sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
}

/** Completion fraction 0..1, or null when there is nothing to complete. */
export function completionFraction(done: number, total: number): number | null {
  if (total <= 0) return null;
  return Math.min(1, Math.max(0, done / total));
}

/**
 * A percentage for a {@link StatValue}, or `null` when the ratio is undefined.
 *
 * `formatPercent` already renders an em dash for null, but a `StatValue` whose
 * `value` is the *string* "—" is indistinguishable from a measured one. Returning
 * `null` keeps "not measured" a property of the data, which is what `StatGrid`
 * and every test assert on.
 */
function percentOrNull(fraction: number | null): string | null {
  return fraction === null ? null : formatPercent(fraction);
}

/** The headline KPI row: what an operator should see first. */
export function repositoryStats(summary: AnalyticsSummary): StatValue[] {
  const { repository, violations, review, processing } = summary;
  return [
    {
      key: 'videos',
      label: 'Videos',
      value: formatNumber(repository.videos_total),
      hint: `${formatNumber(repository.videos_processed)} processed`,
    },
    {
      key: 'footage',
      label: 'Footage',
      value:
        repository.footage_seconds === null
          ? null
          : formatDuration(repository.footage_seconds),
      hint: 'Total duration of stored video that declares one',
    },
    {
      key: 'violations',
      label: 'Violations',
      value: formatNumber(violations.events_total),
      hint: 'Confirmed events, deduplicated across runs',
    },
    {
      key: 'reviewed',
      label: 'Reviewed',
      value: percentOrNull(completionFraction(review.events_reviewed, review.events_total)),
      hint: 'Events an analyst has acted on',
    },
    {
      key: 'runs',
      label: 'Runs',
      value: formatNumber(processing.jobs_total),
      hint: `${formatNumber(processing.jobs_succeeded)} succeeded`,
    },
    {
      key: 'storage',
      label: 'Storage',
      value: formatBytes(repository.storage_bytes),
      hint: 'Stored source video',
    },
  ];
}

/** Processing detail: run outcomes and what they cost. */
export function processingStats(summary: AnalyticsSummary): StatValue[] {
  const { processing } = summary;
  return [
    { key: 'succeeded', label: 'Succeeded', value: formatNumber(processing.jobs_succeeded) },
    { key: 'running', label: 'Running', value: formatNumber(processing.jobs_running) },
    { key: 'pending', label: 'Pending', value: formatNumber(processing.jobs_pending) },
    { key: 'failed', label: 'Failed', value: formatNumber(processing.jobs_failed) },
    { key: 'cancelled', label: 'Cancelled', value: formatNumber(processing.jobs_cancelled) },
    {
      key: 'duration',
      label: 'Average run',
      // Null for a repository whose runs predate lifecycle timing — an unmeasured
      // mean, not a zero-length run.
      value:
        processing.average_duration_seconds === null
          ? null
          : `${processing.average_duration_seconds.toFixed(1)}s`,
      hint: `Mean over ${formatNumber(processing.timed_jobs)} timed run(s)`,
    },
  ];
}

/** Evidence coverage: how much confirmed evidence has actually been rendered. */
export function evidenceStats(summary: AnalyticsSummary): StatValue[] {
  const { evidence } = summary;
  return [
    {
      key: 'coverage',
      label: 'Evidence coverage',
      value: percentOrNull(
        completionFraction(evidence.events_with_artifacts, evidence.events_total),
      ),
      hint: 'Events with at least one rendered artifact',
    },
    {
      key: 'artifacts',
      label: 'Artifacts',
      value: formatNumber(evidence.artifacts_total),
      hint: 'Content-addressed rendered files',
    },
    {
      key: 'artifact-bytes',
      label: 'Artifact storage',
      value: formatBytes(evidence.artifact_bytes),
    },
    {
      key: 'overlays',
      label: 'Annotated videos',
      value: formatNumber(evidence.overlays_available),
    },
  ];
}

/** Repository health signals, each one actionable. */
export function healthStats(summary: AnalyticsSummary): StatValue[] {
  const { health, repository } = summary;
  return [
    { key: 'engine', label: 'Engine', value: health.engine },
    { key: 'version', label: 'Version', value: health.version },
    {
      key: 'failed',
      label: 'Failed runs',
      value: formatNumber(health.failed_jobs),
    },
    {
      key: 'calibrated',
      label: 'Calibrated',
      value: `${formatNumber(repository.videos_calibrated)}/${formatNumber(
        repository.videos_total,
      )}`,
      hint: 'Videos bound to a scene; uncalibrated videos can run fewer rules',
    },
    {
      key: 'missing-media',
      label: 'Missing media',
      value: formatNumber(health.videos_missing_media),
      hint: 'Videos whose stored file is gone from disk',
    },
    {
      key: 'untimed',
      label: 'Untimed runs',
      value: formatNumber(health.runs_without_timing),
      hint: 'Runs recovered from a snapshot predating lifecycle timing',
    },
  ];
}

/** Whether the violation breakdown is known to be incomplete. */
export function hasPartialBreakdown(summary: AnalyticsSummary): boolean {
  return summary.violations.uncounted_jobs > 0;
}

/** Whether the repository holds nothing at all yet. */
export function isEmptyRepository(summary: AnalyticsSummary): boolean {
  return summary.repository.videos_total === 0 && summary.processing.jobs_total === 0;
}

const ACTIVITY_LABELS: Record<string, string> = {
  upload: 'Upload',
  run: 'Run',
  review: 'Review',
};

/** A readable label for an activity entry's kind. */
export function activityKindLabel(kind: string): string {
  return ACTIVITY_LABELS[kind] ?? kind;
}

import { BarChart3 } from 'lucide-react';

import { BarChart } from '@/components/dashboard/bar-chart';
import { ProgressMeter } from '@/components/dashboard/progress-meter';
import { EmptyState } from '@/components/common/empty-state';
import { ErrorBanner } from '@/components/common/error-banner';
import { PageHeader } from '@/components/common/page-header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { StatGrid } from '@/components/workspace/stat-grid';
import { useAnalytics } from '@/hooks/use-system';
import {
  completionFraction,
  evidenceStats,
  hasPartialBreakdown,
  isEmptyRepository,
  processingStats,
  repositoryStats,
  violationBars,
} from '@/lib/analytics';
import { formatNumber } from '@/lib/format';
import { reviewStats } from '@/lib/review-stats';

/**
 * Analytics (H15) — the detailed read of the same repository summary.
 *
 * Deliberately the *same* payload the dashboard uses, laid out for study rather
 * than for a glance: full breakdowns, coverage ratios, and the engine's own
 * metrics for the most recent run. Sharing one endpoint is what keeps the two
 * pages from ever disagreeing.
 *
 * There is no time-series chart here, and that is on purpose: run lifecycle
 * timing only began being recorded in H15, so a repository has no history to plot
 * until it accumulates one. A chart of `trigger_at` would be worse than nothing —
 * media time is anchored at a fixed 1970 epoch.
 */
export default function AnalyticsPage() {
  const query = useAnalytics();
  const summary = query.data;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={BarChart3}
        title="Analytics"
        description="Violation breakdown, throughput, and engine performance."
      />

      {query.isError ? (
        <ErrorBanner
          title="Could not load analytics"
          error={query.error}
          onRetry={() => void query.refetch()}
        />
      ) : null}

      {query.isLoading && !summary ? <Skeleton className="h-64 w-full" /> : null}

      {summary && isEmptyRepository(summary) ? (
        <EmptyState
          icon={BarChart3}
          title="No analytics yet"
          description="Process a video and its violations, throughput, and evidence coverage will be summarised here."
        />
      ) : null}

      {summary && !isEmptyRepository(summary) ? (
        <div className="space-y-4">
          <StatGrid stats={repositoryStats(summary)} className="sm:grid-cols-3 xl:grid-cols-6" />

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Violations by type</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <BarChart
                  data={violationBars(summary.violations.by_type)}
                  label="Confirmed violations by type"
                  emptyMessage="No violations have been confirmed yet."
                />
                <p className="text-2xs text-muted-foreground">
                  {formatNumber(summary.violations.events_total)} confirmed event(s) across{' '}
                  {formatNumber(summary.violations.counted_jobs)} counted run(s)
                  {hasPartialBreakdown(summary)
                    ? `; ${formatNumber(
                        summary.violations.uncounted_jobs,
                      )} run(s) predate per-type counts and are excluded from this breakdown.`
                    : '.'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Coverage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ProgressMeter
                  label="Evidence rendered"
                  fraction={completionFraction(
                    summary.evidence.events_with_artifacts,
                    summary.evidence.events_total,
                  )}
                />
                <ProgressMeter
                  label="Events reviewed"
                  fraction={completionFraction(
                    summary.review.events_reviewed,
                    summary.review.events_total,
                  )}
                />
                <ProgressMeter
                  label="Videos calibrated"
                  fraction={completionFraction(
                    summary.repository.videos_calibrated,
                    summary.repository.videos_total,
                  )}
                />
                <ProgressMeter
                  label="Videos processed"
                  fraction={completionFraction(
                    summary.repository.videos_processed,
                    summary.repository.videos_total,
                  )}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Processing</CardTitle>
              </CardHeader>
              <CardContent>
                <StatGrid stats={processingStats(summary)} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Evidence</CardTitle>
              </CardHeader>
              <CardContent>
                <StatGrid stats={evidenceStats(summary)} />
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Latest run</CardTitle>
              </CardHeader>
              <CardContent>
                {summary.latest_run ? (
                  // The engine's own snapshot, reused verbatim through the existing
                  // review-stats mapping rather than re-derived here.
                  <StatGrid stats={reviewStats([], summary.latest_run)} />
                ) : (
                  <p className="text-xs text-muted-foreground">
                    No run has recorded engine metrics yet.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      ) : null}
    </div>
  );
}

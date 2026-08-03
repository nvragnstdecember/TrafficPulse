import { LayoutDashboard, Video } from 'lucide-react';
import { Link } from 'react-router-dom';

import { ActivityFeed } from '@/components/dashboard/activity-feed';
import { BarChart } from '@/components/dashboard/bar-chart';
import { ProgressMeter } from '@/components/dashboard/progress-meter';
import { EmptyState } from '@/components/common/empty-state';
import { ErrorBanner } from '@/components/common/error-banner';
import { PageHeader } from '@/components/common/page-header';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { StatGrid } from '@/components/workspace/stat-grid';
import { useAnalytics } from '@/hooks/use-system';
import {
  completionFraction,
  evidenceStats,
  hasPartialBreakdown,
  healthStats,
  isEmptyRepository,
  processingStats,
  repositoryStats,
  violationBars,
} from '@/lib/analytics';
import { formatNumber } from '@/lib/format';
import { ROUTES } from '@/routes/paths';

/**
 * Dashboard (H15) — the operational homepage.
 *
 * Answers "what is happening in this repository right now" from a **single**
 * request. Every figure is computed by the backend's `AnalyticsService`; this page
 * fetches one summary and lays it out. It performs no aggregation of its own, and
 * it never derives a repository figure from a list endpoint.
 */
export default function DashboardPage() {
  const query = useAnalytics();
  const summary = query.data;

  return (
    <div className="space-y-6">
      <PageHeader
        icon={LayoutDashboard}
        title="Dashboard"
        description="An at-a-glance view of processing activity and detected violations."
      />

      {query.isError ? (
        <ErrorBanner
          title="Could not load analytics"
          error={query.error}
          onRetry={() => void query.refetch()}
        />
      ) : null}

      {query.isLoading && !summary ? <DashboardSkeleton /> : null}

      {summary && isEmptyRepository(summary) ? (
        <EmptyState
          icon={Video}
          title="This repository is empty"
          description="Upload a video to start a detection job; processing activity and detected violations will appear here."
          action={
            <Button asChild size="sm">
              <Link to={ROUTES.videos}>Go to videos</Link>
            </Button>
          }
        />
      ) : null}

      {summary && !isEmptyRepository(summary) ? (
        <>
          <StatGrid stats={repositoryStats(summary)} className="sm:grid-cols-3 xl:grid-cols-6" />

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Violations</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <BarChart
                  data={violationBars(summary.violations.by_type)}
                  label="Confirmed violations by type"
                  emptyMessage="No violations have been confirmed yet."
                />
                {hasPartialBreakdown(summary) ? (
                  <p className="text-2xs text-muted-foreground">
                    {formatNumber(summary.violations.uncounted_jobs)} run(s) recorded before
                    per-type counts existed are included in the total but not in this
                    breakdown.
                  </p>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Review queue</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ProgressMeter
                  label="Reviewed"
                  fraction={completionFraction(
                    summary.review.events_reviewed,
                    summary.review.events_total,
                  )}
                  caption={`${formatNumber(summary.review.events_reviewed)} of ${formatNumber(
                    summary.review.events_total,
                  )} events acted on`}
                />
                <ProgressMeter
                  label="Evidence rendered"
                  fraction={completionFraction(
                    summary.evidence.events_with_artifacts,
                    summary.evidence.events_total,
                  )}
                  caption={`${formatNumber(
                    summary.evidence.events_with_artifacts,
                  )} of ${formatNumber(summary.evidence.events_total)} events have artifacts`}
                />
                {summary.review.events_pending > 0 ? (
                  <Button asChild variant="outline" size="sm">
                    <Link to={ROUTES.videos}>
                      Review {formatNumber(summary.review.events_pending)} pending
                    </Link>
                  </Button>
                ) : null}
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
                <CardTitle>Repository health</CardTitle>
              </CardHeader>
              <CardContent>
                <StatGrid stats={healthStats(summary)} />
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

            <Card>
              <CardHeader>
                <CardTitle>Recent activity</CardTitle>
              </CardHeader>
              <CardContent>
                <ActivityFeed entries={summary.recent_activity} />
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-20 w-full" />
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    </div>
  );
}

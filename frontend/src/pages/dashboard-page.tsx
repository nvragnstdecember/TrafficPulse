import { Activity, FileSearch, LayoutDashboard, Video } from 'lucide-react';

import { EmptyState } from '@/components/common/empty-state';
import { PageHeader } from '@/components/common/page-header';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

/**
 * Dashboard — the landing overview. Foundation scaffold: the layout and cards
 * are in place; live widgets (throughput, recent events, job queue) arrive with
 * their feature phases.
 */
export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        icon={LayoutDashboard}
        title="Dashboard"
        description="An at-a-glance view of processing activity and detected violations."
      />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <EmptyState
              icon={Video}
              title="No processing activity yet"
              description="Upload a video to start a detection job."
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Recent events</CardTitle>
          </CardHeader>
          <CardContent>
            <EmptyState
              icon={FileSearch}
              title="No events detected"
              description="Confirmed violations will appear here as jobs complete."
            />
          </CardContent>
        </Card>
        <Card className="md:col-span-2 xl:col-span-1">
          <CardHeader>
            <CardTitle>System</CardTitle>
          </CardHeader>
          <CardContent>
            <EmptyState
              icon={Activity}
              title="Metrics pending"
              description="Engine throughput and latency will surface here."
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

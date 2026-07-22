import { BarChart3 } from 'lucide-react';

import { EmptyState } from '@/components/common/empty-state';
import { PageHeader } from '@/components/common/page-header';

/** Analytics — trends + engine metrics (feature UI arrives in a later phase). */
export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        icon={BarChart3}
        title="Analytics"
        description="Violation trends, throughput, and engine performance."
      />
      <EmptyState
        icon={BarChart3}
        title="No analytics yet"
        description="Charts and aggregate metrics will be built on this foundation."
      />
    </div>
  );
}

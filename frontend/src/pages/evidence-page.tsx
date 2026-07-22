import { FileSearch } from 'lucide-react';

import { EmptyState } from '@/components/common/empty-state';
import { PageHeader } from '@/components/common/page-header';

/** Evidence — confirmed events + evidence manifests (feature UI in a later phase). */
export default function EvidencePage() {
  return (
    <div className="space-y-6">
      <PageHeader
        icon={FileSearch}
        title="Evidence"
        description="Browse confirmed violations and their evidence manifests."
      />
      <EmptyState
        icon={FileSearch}
        title="No evidence to review"
        description="Confirmed events and their frame references will be reviewable here."
      />
    </div>
  );
}

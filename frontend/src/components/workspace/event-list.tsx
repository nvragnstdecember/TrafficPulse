import { ListX, ShieldCheck } from 'lucide-react';

import { type ViolationType } from '@/api/types';
import { type EventFilters, type WorkspaceEvent, type WorkspaceSort } from '@/lib/workspace';

import { EmptyState } from '../common/empty-state';
import { ErrorBanner } from '../common/error-banner';
import { VirtualList } from '../common/virtual-list';
import { Skeleton } from '../ui/skeleton';
import { EventCard } from './event-card';
import { EventFiltersBar } from './event-filters';

export interface EventListProps {
  events: WorkspaceEvent[];
  /** Total confirmed events before filtering, for the "n of m" counter. */
  totalCount: number;
  selectedEventId: string | null;
  onSelect: (eventId: string) => void;
  filters: EventFilters;
  onFiltersChange: (filters: EventFilters) => void;
  sort: WorkspaceSort;
  onSortChange: (sort: WorkspaceSort) => void;
  availableViolations: ViolationType[];
  isLoading: boolean;
  isError: boolean;
  error?: unknown;
  onRetry: () => void;
}

const ROW_HEIGHT = 76;
const LIST_HEIGHT = 420;

/**
 * The event list panel (H7C): filters + a virtualized, selectable list of the
 * video's confirmed violations, with loading, error, and empty states.
 *
 * Filtering/sorting is applied by the caller (pure functions in `lib/workspace`),
 * so this component only renders — and virtualizes, so hundreds of events stay
 * smooth.
 */
export function EventList({
  events,
  totalCount,
  selectedEventId,
  onSelect,
  filters,
  onFiltersChange,
  sort,
  onSortChange,
  availableViolations,
  isLoading,
  isError,
  error,
  onRetry,
}: EventListProps) {
  return (
    <section aria-label="Detected events" className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-base font-semibold tracking-tight">Events</h2>
        <p className="text-xs tabular-nums text-muted-foreground">
          {isLoading ? 'Loading…' : `${events.length} of ${totalCount}`}
        </p>
      </div>

      <EventFiltersBar
        filters={filters}
        onFiltersChange={onFiltersChange}
        sort={sort}
        onSortChange={onSortChange}
        availableViolations={availableViolations}
      />

      {isError ? (
        <ErrorBanner title="Could not load events" error={error} onRetry={onRetry} />
      ) : isLoading ? (
        <div className="space-y-2" data-testid="event-list-loading">
          {[0, 1, 2, 3].map((index) => (
            <Skeleton key={index} className="h-16 w-full rounded-md" />
          ))}
        </div>
      ) : totalCount === 0 ? (
        <EmptyState
          icon={ShieldCheck}
          title="No violations detected"
          description="Processing found no confirmed violations in this video."
        />
      ) : events.length === 0 ? (
        <EmptyState
          icon={ListX}
          title="No events match the filters"
          description="Adjust the search, violation, or confidence filters to see more."
        />
      ) : (
        <VirtualList
          items={events}
          rowHeight={ROW_HEIGHT}
          height={LIST_HEIGHT}
          getKey={(event) => event.id}
          className="-mx-1 px-1"
          renderItem={(event) => (
            <div className="pb-2">
              <EventCard
                event={event}
                selected={event.id === selectedEventId}
                onSelect={onSelect}
              />
            </div>
          )}
        />
      )}
    </section>
  );
}

import { Download, ListChecks, ListX, ShieldCheck, X } from 'lucide-react';
import { useCallback, useMemo } from 'react';

import { type ViolationType } from '@/api/types';
import {
  type EventFilters,
  type WorkspaceEvent,
  type WorkspaceSort,
  DEFAULT_EVENT_FILTERS,
} from '@/lib/workspace';

import { EmptyState } from '../common/empty-state';
import { ErrorBanner } from '../common/error-banner';
import { VirtualList } from '../common/virtual-list';
import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { Skeleton } from '../ui/skeleton';
import { EventCard } from './event-card';
import { EventFiltersBar } from './event-filters';

export type ExportFormat = 'json' | 'csv';

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
  // --- multi-select + export (H7E) ---
  selectionMode: boolean;
  onSelectionModeChange: (on: boolean) => void;
  checkedIds: Set<string>;
  onToggleChecked: (eventId: string) => void;
  onCheckAll: () => void;
  onClearChecked: () => void;
  /** Export the checked events, or the visible list when none are checked. */
  onExport: (format: ExportFormat) => void;
}

const ROW_HEIGHT = 76;
const LIST_HEIGHT = 420;

/**
 * The event list panel (H7C; review tooling in H7E): filters + a virtualized,
 * selectable list, plus a multi-select bulk bar, an export menu, and keyboard
 * navigation (↑/↓ move the active event, Space toggles its checkbox, Home/End
 * jump to ends). Filtering/sorting is applied by the caller (pure functions), so
 * this component only renders and virtualizes.
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
  selectionMode,
  onSelectionModeChange,
  checkedIds,
  onToggleChecked,
  onCheckAll,
  onClearChecked,
  onExport,
}: EventListProps) {
  const activeIndex = useMemo(
    () => events.findIndex((event) => event.id === selectedEventId),
    [events, selectedEventId],
  );

  const checkedCount = events.reduce((n, event) => (checkedIds.has(event.id) ? n + 1 : n), 0);
  const allChecked = events.length > 0 && checkedCount === events.length;

  const onKeyDown = useCallback(
    (keyEvent: React.KeyboardEvent<HTMLDivElement>) => {
      if (events.length === 0) return;
      const index = activeIndex < 0 ? 0 : activeIndex;
      const move = (next: number) => {
        keyEvent.preventDefault();
        const clamped = Math.min(events.length - 1, Math.max(0, next));
        onSelect(events[clamped].id);
      };
      switch (keyEvent.key) {
        case 'ArrowDown':
          move(activeIndex < 0 ? 0 : index + 1);
          return;
        case 'ArrowUp':
          move(activeIndex < 0 ? 0 : index - 1);
          return;
        case 'Home':
          move(0);
          return;
        case 'End':
          move(events.length - 1);
          return;
        case ' ':
          if (selectionMode && activeIndex >= 0) {
            keyEvent.preventDefault();
            onToggleChecked(events[activeIndex].id);
          }
          return;
        default:
      }
    },
    [events, activeIndex, onSelect, onToggleChecked, selectionMode],
  );

  return (
    <section aria-label="Detected events" className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold tracking-tight">Events</h2>
        <div className="flex items-center gap-2">
          <p className="text-xs tabular-nums text-muted-foreground">
            {isLoading ? 'Loading…' : `${events.length} of ${totalCount}`}
          </p>
          <Button
            variant={selectionMode ? 'secondary' : 'ghost'}
            size="sm"
            aria-pressed={selectionMode}
            onClick={() => onSelectionModeChange(!selectionMode)}
          >
            <ListChecks className="size-4" />
            Select
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" disabled={events.length === 0}>
                <Download className="size-4" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>
                Export {checkedCount > 0 ? `${checkedCount} selected` : `${events.length} shown`}
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => onExport('json')}>JSON</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => onExport('csv')}>CSV</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <EventFiltersBar
        filters={filters}
        onFiltersChange={onFiltersChange}
        sort={sort}
        onSortChange={onSortChange}
        availableViolations={availableViolations}
      />

      {selectionMode && events.length > 0 ? (
        <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2.5 py-1.5 text-xs">
          <input
            type="checkbox"
            checked={allChecked}
            aria-label={allChecked ? 'Clear selection' : 'Select all events'}
            onChange={() => (allChecked ? onClearChecked() : onCheckAll())}
            className="size-4 cursor-pointer accent-primary"
          />
          <span className="tabular-nums text-muted-foreground">{checkedCount} selected</span>
          {checkedCount > 0 ? (
            <Button variant="ghost" size="sm" className="ml-auto" onClick={onClearChecked}>
              <X className="size-4" />
              Clear
            </Button>
          ) : null}
        </div>
      ) : null}

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
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() => onFiltersChange(DEFAULT_EVENT_FILTERS)}
            >
              <X className="size-4" />
              Clear filters
            </Button>
          }
        />
      ) : (
        <div onKeyDown={onKeyDown} role="presentation">
          <VirtualList
            items={events}
            rowHeight={ROW_HEIGHT}
            height={LIST_HEIGHT}
            getKey={(event) => event.id}
            scrollToIndex={activeIndex}
            aria-label="Event results"
            className="-mx-1 px-1"
            renderItem={(event) => (
              <div className="pb-2">
                <EventCard
                  event={event}
                  selected={event.id === selectedEventId}
                  onSelect={onSelect}
                  showCheckbox={selectionMode}
                  checked={checkedIds.has(event.id)}
                  onToggleChecked={onToggleChecked}
                />
              </div>
            )}
          />
        </div>
      )}
    </section>
  );
}

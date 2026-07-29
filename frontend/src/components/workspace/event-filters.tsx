import { ListFilter, SlidersHorizontal, X } from 'lucide-react';

import { type ViolationType } from '@/api/types';
import { formatPercent } from '@/lib/format';
import {
  DEFAULT_EVENT_FILTERS,
  type EventFilters,
  type WorkspaceSort,
  WORKSPACE_SORTS,
  hasActiveFilters,
  violationLabel,
} from '@/lib/workspace';

import { SearchInput } from '../common/search-input';
import { Button } from '../ui/button';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';

export interface EventFiltersBarProps {
  filters: EventFilters;
  onFiltersChange: (filters: EventFilters) => void;
  sort: WorkspaceSort;
  onSortChange: (sort: WorkspaceSort) => void;
  availableViolations: ViolationType[];
}

/** Search, violation + confidence filters, and sort for the event list. */
export function EventFiltersBar({
  filters,
  onFiltersChange,
  sort,
  onSortChange,
  availableViolations,
}: EventFiltersBarProps) {
  const toggleViolation = (violation: ViolationType) => {
    const next = filters.violationTypes.includes(violation)
      ? filters.violationTypes.filter((v) => v !== violation)
      : [...filters.violationTypes, violation];
    onFiltersChange({ ...filters, violationTypes: next });
  };

  return (
    <div className="space-y-2">
      <SearchInput
        value={filters.query}
        onValueChange={(query) => onFiltersChange({ ...filters, query })}
        placeholder="Search track, violation, or time (1:23)…"
      />
      <div className="flex flex-wrap items-center gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <ListFilter className="size-4" />
              Violation
              {filters.violationTypes.length > 0 ? ` (${filters.violationTypes.length})` : ''}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuLabel>Filter by violation</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {availableViolations.map((violation) => (
              <DropdownMenuCheckboxItem
                key={violation}
                checked={filters.violationTypes.includes(violation)}
                onCheckedChange={() => toggleViolation(violation)}
                onSelect={(event) => event.preventDefault()}
              >
                {violationLabel(violation)}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <SlidersHorizontal className="size-4" />
              {WORKSPACE_SORTS.find((option) => option.value === sort)?.label}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuLabel>Sort by</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {WORKSPACE_SORTS.map((option) => (
              <DropdownMenuCheckboxItem
                key={option.value}
                checked={sort === option.value}
                onCheckedChange={() => onSortChange(option.value)}
              >
                {option.label}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {hasActiveFilters(filters) ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFiltersChange(DEFAULT_EVENT_FILTERS)}
          >
            <X className="size-4" />
            Clear
          </Button>
        ) : null}
      </div>

      <div className="space-y-1.5">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="w-24 shrink-0">Confidence ≥</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={filters.minConfidence}
            onChange={(event) => {
              // Keep the range coherent: dragging the floor above the ceiling
              // pushes the ceiling rather than producing an empty result set.
              const min = Number(event.target.value);
              onFiltersChange({
                ...filters,
                minConfidence: min,
                maxConfidence: Math.max(min, filters.maxConfidence),
              });
            }}
            aria-label="Minimum confidence"
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
          />
          <span className="w-9 text-right tabular-nums text-foreground">
            {formatPercent(filters.minConfidence)}
          </span>
        </label>

        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="w-24 shrink-0">Confidence ≤</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={filters.maxConfidence}
            onChange={(event) => {
              const max = Number(event.target.value);
              onFiltersChange({
                ...filters,
                maxConfidence: max,
                minConfidence: Math.min(max, filters.minConfidence),
              });
            }}
            aria-label="Maximum confidence"
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
          />
          <span className="w-9 text-right tabular-nums text-foreground">
            {formatPercent(filters.maxConfidence)}
          </span>
        </label>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="w-24 shrink-0">Time range</span>
          <label className="flex flex-1 items-center gap-1">
            <span className="sr-only">From (seconds)</span>
            <input
              type="number"
              min={0}
              step={1}
              value={filters.fromSeconds || ''}
              placeholder="from"
              onChange={(event) =>
                onFiltersChange({
                  ...filters,
                  fromSeconds: Math.max(0, Number(event.target.value) || 0),
                })
              }
              aria-label="From second"
              className="w-full rounded-md border bg-background px-2 py-1 tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </label>
          <span aria-hidden="true">–</span>
          <label className="flex flex-1 items-center gap-1">
            <span className="sr-only">To (seconds)</span>
            <input
              type="number"
              min={0}
              step={1}
              value={filters.toSeconds ?? ''}
              placeholder="to"
              onChange={(event) => {
                const raw = event.target.value.trim();
                onFiltersChange({
                  ...filters,
                  toSeconds: raw === '' ? null : Math.max(0, Number(raw) || 0),
                });
              }}
              aria-label="To second"
              className="w-full rounded-md border bg-background px-2 py-1 tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </label>
        </div>
      </div>
    </div>
  );
}

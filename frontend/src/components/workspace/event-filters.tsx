import { ListFilter, SlidersHorizontal, X } from 'lucide-react';

import { type ViolationType } from '@/api/types';
import { formatPercent } from '@/lib/format';
import {
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
        placeholder="Search events…"
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
            onClick={() => onFiltersChange({ query: '', violationTypes: [], minConfidence: 0 })}
          >
            <X className="size-4" />
            Clear
          </Button>
        ) : null}
      </div>

      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="shrink-0">Min confidence</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={filters.minConfidence}
          onChange={(event) =>
            onFiltersChange({ ...filters, minConfidence: Number(event.target.value) })
          }
          aria-label="Minimum confidence"
          className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
        />
        <span className="w-9 text-right tabular-nums text-foreground">
          {formatPercent(filters.minConfidence)}
        </span>
      </label>
    </div>
  );
}

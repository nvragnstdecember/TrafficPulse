import { FileVideo } from 'lucide-react';
import { memo } from 'react';

import { formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import {
  type WorkspaceEvent,
  formatClock,
  severityLabel,
  severityTone,
  violationLabel,
  violationSeverity,
  violationTone,
} from '@/lib/workspace';

import { StatusChip } from '../common/status-chip';

export interface EventCardProps {
  event: WorkspaceEvent;
  selected: boolean;
  onSelect: (eventId: string) => void;
  /** Show a multi-select checkbox (bulk mode; H7E). */
  showCheckbox?: boolean;
  /** Whether this event is checked for a bulk action. */
  checked?: boolean;
  onToggleChecked?: (eventId: string) => void;
}

/**
 * One event in the list (H7C; severity + multi-select in H7E): violation +
 * severity, time, confidence, lane, track. The optional checkbox is a sibling of
 * the selection button (never nested), so bulk-select and open-for-review stay
 * independently keyboard-operable. Memoized so unchanged rows do not rerender as
 * the live list polls.
 */
export const EventCard = memo(function EventCard({
  event,
  selected,
  onSelect,
  showCheckbox = false,
  checked = false,
  onToggleChecked,
}: EventCardProps) {
  const severity = violationSeverity(event.violationType);

  return (
    <div className="flex items-center gap-2">
      {showCheckbox ? (
        <input
          type="checkbox"
          checked={checked}
          onChange={() => onToggleChecked?.(event.id)}
          aria-label={`Select ${violationLabel(event.violationType)} at ${formatClock(
            event.mediaSeconds,
          )}`}
          className="size-4 shrink-0 cursor-pointer accent-primary"
        />
      ) : null}
      <button
        type="button"
        onClick={() => onSelect(event.id)}
        aria-pressed={selected}
        aria-label={`${violationLabel(event.violationType)} at ${formatClock(event.mediaSeconds)}`}
        className={cn(
          'flex min-w-0 flex-1 items-center gap-3 rounded-md border p-2.5 text-left transition-colors',
          selected
            ? 'border-primary bg-accent'
            : 'border-transparent bg-card hover:border-border hover:bg-accent/50',
        )}
      >
        <span
          aria-hidden="true"
          className="flex size-12 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground"
        >
          <FileVideo className="size-5" />
        </span>
        <span className="min-w-0 flex-1 space-y-1">
          <span className="flex items-center gap-2">
            <StatusChip
              tone={violationTone(event.violationType)}
              label={violationLabel(event.violationType)}
              dot={false}
            />
            <StatusChip
              tone={severityTone(severity)}
              label={severityLabel(severity)}
              dot={false}
              className="text-2xs"
            />
            <span className="ml-auto font-mono text-xs tabular-nums text-muted-foreground">
              {formatClock(event.mediaSeconds)}
            </span>
          </span>
          <span className="flex items-center gap-3 text-xs text-muted-foreground">
            <span>Conf {event.confidence === null ? '—' : formatPercent(event.confidence)}</span>
            <span className="truncate">Lane {event.lane ?? event.cameraId}</span>
            {event.trackIds[0] ? <span className="truncate">Track {event.trackIds[0]}</span> : null}
          </span>
        </span>
      </button>
    </div>
  );
});

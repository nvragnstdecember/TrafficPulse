import { FileVideo } from 'lucide-react';

import { formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import { type WorkspaceEvent, formatClock, violationLabel, violationTone } from '@/lib/workspace';

import { StatusChip } from '../common/status-chip';

export interface EventCardProps {
  event: WorkspaceEvent;
  selected: boolean;
  onSelect: (eventId: string) => void;
}

/** One event in the list: violation, time, confidence, lane, track, thumbnail. */
export function EventCard({ event, selected, onSelect }: EventCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(event.id)}
      aria-pressed={selected}
      aria-label={`${violationLabel(event.violationType)} at ${formatClock(event.mediaSeconds)}`}
      className={cn(
        'flex w-full items-center gap-3 rounded-md border p-2.5 text-left transition-colors',
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
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
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
  );
}

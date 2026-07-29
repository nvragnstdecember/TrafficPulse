import { memo } from 'react';

import { formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import {
  CONFIDENCE_LABELS,
  type WorkspaceEvent,
  formatClock,
  severityLabel,
  severityTone,
  violationLabel,
  violationSeverity,
  violationTone,
} from '@/lib/workspace';

import { StatusChip } from '../common/status-chip';
import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip';
import { EventThumbnail } from './event-thumbnail';

export interface EventCardProps {
  event: WorkspaceEvent;
  selected: boolean;
  onSelect: (eventId: string) => void;
  /** Show a multi-select checkbox (bulk mode; H7E). */
  showCheckbox?: boolean;
  /** Whether this event is checked for a bulk action. */
  checked?: boolean;
  onToggleChecked?: (eventId: string) => void;
  /** Playable video source, so the card can show the frame it happened on. */
  thumbnailSrc?: string | null;
}

/**
 * One violation card in the review list (H7C; severity + multi-select in H7E;
 * thumbnail, observation window, and confidence provenance in Phase 2).
 *
 * Three rows, fixed shape: identity (violation + severity + time), evidence
 * (thumbnail + confidence + how long it was observed), attribution (track, lane).
 * The rows never reflow between events, so a list scans vertically — the property
 * that makes a dashboard usable at a glance rather than a wall of variable blocks.
 *
 * The confidence figure names its own source on hover. This platform publishes
 * confidence *components* and refuses to blend them, so a bare "97%" would be the
 * one number on the card that does not say what it measured.
 */
export const EventCard = memo(function EventCard({
  event,
  selected,
  onSelect,
  showCheckbox = false,
  checked = false,
  onToggleChecked,
  thumbnailSrc = null,
}: EventCardProps) {
  const severity = violationSeverity(event.violationType);
  const confidenceText =
    event.confidence === null ? '—' : formatPercent(event.confidence);
  const confidenceHint =
    event.confidence === null || !event.confidenceSource
      ? 'No confidence component was measured for this event'
      : `${CONFIDENCE_LABELS[event.confidenceSource]} confidence`;

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
        <EventThumbnail
          src={thumbnailSrc}
          seconds={event.mediaSeconds}
          violationType={event.violationType}
        />
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
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="cursor-help tabular-nums">Conf {confidenceText}</span>
              </TooltipTrigger>
              <TooltipContent>{confidenceHint}</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="cursor-help tabular-nums">
                  Obs {event.observationSeconds.toFixed(1)}s
                </span>
              </TooltipTrigger>
              <TooltipContent>
                Sustained from {formatClock(event.startSeconds)} to{' '}
                {formatClock(event.mediaSeconds)}
              </TooltipContent>
            </Tooltip>
            {event.trackIds[0] ? <span className="truncate">Track {event.trackIds[0]}</span> : null}
          </span>
        </span>
      </button>
    </div>
  );
});

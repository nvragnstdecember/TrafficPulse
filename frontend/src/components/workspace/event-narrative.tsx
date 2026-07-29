import { CheckCircle2, Circle, FileCheck2, TrendingUp } from 'lucide-react';
import { type ComponentType } from 'react';

import { cn } from '@/lib/utils';
import {
  type NarrativeStage,
  type NarrativeStep,
  formatClock,
} from '@/lib/workspace';

export interface EventNarrativeProps {
  steps: NarrativeStep[];
  /** Current playhead position, so the analyst sees where they are in the story. */
  currentTime?: number;
  /** Seek the player to a step's instant. */
  onSeek?: (seconds: number) => void;
}

const STAGE_ICONS: Record<NarrativeStage, ComponentType<{ className?: string }>> = {
  observation: Circle,
  accumulation: TrendingUp,
  confirmation: CheckCircle2,
  evidence: FileCheck2,
};

const STAGE_ACCENTS: Record<NarrativeStage, string> = {
  observation: 'text-muted-foreground',
  accumulation: 'text-warning',
  confirmation: 'text-destructive',
  evidence: 'text-success',
};

/**
 * How one violation evolved, as a vertical timeline (Phase 2).
 *
 * Answers the question a confirmed event alone cannot: *why* did the system decide
 * this? Each step is a moment the reasoner actually recorded — observation opens,
 * support accumulates against the rule's own threshold, the violation confirms,
 * evidence is finalized — and every step is seekable, so reading the story and
 * watching it are the same gesture.
 *
 * A step at or before the playhead is rendered as reached, which turns the list
 * into a progress read-out while the video plays.
 */
export function EventNarrative({ steps, currentTime, onSeek }: EventNarrativeProps) {
  if (steps.length === 0) return null;

  return (
    <ol className="space-y-0">
      {steps.map((step, index) => {
        const Icon = STAGE_ICONS[step.stage];
        const reached = currentTime !== undefined && currentTime >= step.seconds;
        const last = index === steps.length - 1;
        return (
          <li key={step.key} className="flex gap-3">
            <div className="flex flex-col items-center">
              <Icon
                className={cn(
                  'size-4 shrink-0 transition-opacity',
                  STAGE_ACCENTS[step.stage],
                  reached ? 'opacity-100' : 'opacity-40',
                )}
                aria-hidden="true"
              />
              {last ? null : (
                <span
                  aria-hidden="true"
                  className={cn('w-px flex-1 bg-border', reached && 'bg-muted-foreground/40')}
                />
              )}
            </div>
            <div className={cn('min-w-0 flex-1', last ? 'pb-0' : 'pb-3')}>
              <button
                type="button"
                onClick={onSeek ? () => onSeek(step.seconds) : undefined}
                disabled={!onSeek}
                className={cn(
                  'flex w-full items-baseline gap-2 text-left',
                  onSeek && 'rounded-sm hover:underline focus-visible:outline-none '
                    + 'focus-visible:ring-2 focus-visible:ring-ring',
                )}
              >
                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                  {formatClock(step.seconds)}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{step.title}</span>
              </button>
              {step.detail ? (
                <p className="text-xs text-muted-foreground">{step.detail}</p>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

import { type StatValue } from '@/lib/review-stats';
import { cn } from '@/lib/utils';

import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip';

export interface StatGridProps {
  stats: StatValue[];
  className?: string;
}

/**
 * A read-only grid of summary figures (Phase 2).
 *
 * One presentation for every statistics surface — the review dashboard and the
 * processing summary both render {@link StatValue}s, so the two can never drift
 * apart in look or in how they treat an absent number.
 *
 * An unmeasured value renders as an em dash, never as `0`. That distinction is the
 * whole point: this platform reports `null` when it did not measure something, and
 * a dashboard that quietly turned those into zeroes would misrepresent a run.
 */
export function StatGrid({ stats, className }: StatGridProps) {
  return (
    <dl className={cn('grid grid-cols-2 gap-3 sm:grid-cols-3', className)}>
      {stats.map((stat) => (
        <div key={stat.key} className="min-w-0 space-y-0.5">
          <dt className="truncate text-2xs uppercase tracking-wide text-muted-foreground">
            {stat.hint ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="cursor-help border-b border-dotted border-muted-foreground/50">
                    {stat.label}
                  </span>
                </TooltipTrigger>
                <TooltipContent>{stat.hint}</TooltipContent>
              </Tooltip>
            ) : (
              stat.label
            )}
          </dt>
          <dd
            className={cn(
              'truncate text-lg font-semibold tabular-nums',
              stat.value === null && 'text-muted-foreground',
            )}
          >
            {stat.value ?? '—'}
          </dd>
        </div>
      ))}
    </dl>
  );
}

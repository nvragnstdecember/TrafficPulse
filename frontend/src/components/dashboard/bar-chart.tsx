import { type BarDatum } from '@/lib/analytics';
import { cn } from '@/lib/utils';

export interface BarChartProps {
  data: BarDatum[];
  /** Accessible name for the whole chart. */
  label: string;
  className?: string;
  /** Rendered when there is nothing to chart. */
  emptyMessage?: string;
}

/**
 * A horizontal bar chart for categorical counts (H15).
 *
 * Horizontal, not a pie: the task is comparing category magnitudes, which bar
 * length supports directly and angular area does not — and violation names are
 * long enough that horizontal rows can label themselves without a legend.
 *
 * Built from CSS-sized elements rather than a charting library. It needs one mark
 * type, the data is pre-normalised by `violationBars`, and the frontend has kept
 * its dependency list deliberately small — a chart framework would add far more
 * weight than this component's worth.
 *
 * Rendered as a definition list so the underlying numbers are available to a
 * screen reader as text; the bars are decorative and are hidden from it.
 */
export function BarChart({ data, label, className, emptyMessage }: BarChartProps) {
  if (data.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {emptyMessage ?? 'Nothing to show yet.'}
      </p>
    );
  }

  return (
    <dl className={cn('space-y-2', className)} aria-label={label}>
      {data.map((datum) => (
        <div key={datum.key} className="space-y-1">
          <div className="flex items-baseline justify-between gap-2 text-xs">
            <dt className="truncate text-foreground">{datum.label}</dt>
            <dd className="shrink-0 tabular-nums text-muted-foreground">{datum.value}</dd>
          </div>
          <div
            className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
            aria-hidden="true"
          >
            <div
              className="h-full rounded-full bg-primary transition-[width]"
              // A non-zero count always paints a visible sliver, so "few" never
              // reads as "none".
              style={{ width: `${Math.max(datum.fraction * 100, datum.value > 0 ? 2 : 0)}%` }}
            />
          </div>
        </div>
      ))}
    </dl>
  );
}

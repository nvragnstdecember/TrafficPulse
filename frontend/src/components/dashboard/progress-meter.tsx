import { formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface ProgressMeterProps {
  label: string;
  /** 0..1, or null when the ratio is undefined (nothing to complete). */
  fraction: number | null;
  /** Caption under the bar, e.g. "3 of 12 reviewed". */
  caption?: string;
  className?: string;
}

/**
 * A labelled completion bar (H15) — one whole divided into done and not-done.
 *
 * Used for review progress and evidence coverage, both of which are genuinely a
 * fraction of a known total. A `null` fraction renders an empty track and an em
 * dash rather than a full or empty bar: "no events to review" is not "0% reviewed",
 * and the dashboard must not imply work is outstanding when none exists.
 */
export function ProgressMeter({ label, fraction, caption, className }: ProgressMeterProps) {
  const measured = fraction !== null;
  const percent = measured ? Math.min(100, Math.max(0, fraction * 100)) : 0;

  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-sm font-medium tabular-nums text-foreground">
          {measured ? formatPercent(fraction) : '—'}
        </span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        // An unmeasured ratio reports no current value, which is what
        // indeterminate means — never a fabricated 0.
        aria-valuenow={measured ? Math.round(percent) : undefined}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width]"
          style={{ width: `${percent}%` }}
        />
      </div>
      {caption ? <p className="text-2xs text-muted-foreground">{caption}</p> : null}
    </div>
  );
}

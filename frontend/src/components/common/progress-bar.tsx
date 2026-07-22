import { cn } from '@/lib/utils';

export type ProgressTone = 'primary' | 'success' | 'destructive';

export interface ProgressBarProps {
  /** 0..1 completion, or null for an indeterminate (unknown-progress) bar. */
  value: number | null;
  label: string;
  tone?: ProgressTone;
  className?: string;
}

const TONE_FILL: Record<ProgressTone, string> = {
  primary: 'bg-primary',
  success: 'bg-success',
  destructive: 'bg-destructive',
};

/**
 * A determinate/indeterminate progress bar (H7C).
 *
 * `value === null` renders the indeterminate variant, which is what the workspace
 * shows while a job reports no progress yet. Exposed as a real ARIA progressbar
 * so screen readers announce the percentage.
 */
export function ProgressBar({ value, label, tone = 'primary', className }: ProgressBarProps) {
  const determinate = value !== null && Number.isFinite(value);
  const ratio = determinate ? Math.min(1, Math.max(0, value)) : 0;

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? Math.round(ratio * 100) : undefined}
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}
    >
      {determinate ? (
        <div
          className={cn('h-full rounded-full transition-[width] duration-fast', TONE_FILL[tone])}
          style={{ width: `${ratio * 100}%` }}
        />
      ) : (
        <div className={cn('h-full w-1/3 animate-pulse rounded-full', TONE_FILL[tone])} />
      )}
    </div>
  );
}

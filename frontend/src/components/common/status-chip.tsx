import { type JobStatus } from '@/api/types';
import { cn } from '@/lib/utils';

import { Badge, type BadgeProps } from '../ui/badge';

export type StatusTone = 'success' | 'warning' | 'error' | 'info' | 'neutral';

const TONE_VARIANT: Record<StatusTone, NonNullable<BadgeProps['variant']>> = {
  success: 'success',
  warning: 'warning',
  error: 'destructive',
  info: 'default',
  neutral: 'muted',
};

const TONE_DOT: Record<StatusTone, string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-destructive',
  info: 'bg-primary',
  neutral: 'bg-muted-foreground',
};

export interface StatusChipProps {
  tone: StatusTone;
  label: string;
  dot?: boolean;
  className?: string;
}

/** A labelled status pill with a leading state dot. */
export function StatusChip({ tone, label, dot = true, className }: StatusChipProps) {
  return (
    <Badge variant={TONE_VARIANT[tone]} className={cn('capitalize', className)}>
      {dot ? (
        <span className={cn('size-1.5 rounded-full', TONE_DOT[tone])} aria-hidden="true" />
      ) : null}
      {label}
    </Badge>
  );
}

/** Map a processing-job status to a chip tone. */
export function jobStatusTone(status: JobStatus): StatusTone {
  switch (status) {
    case 'succeeded':
      return 'success';
    case 'running':
      return 'info';
    case 'pending':
      return 'neutral';
    case 'failed':
      return 'error';
    case 'cancelled':
      return 'neutral';
  }
}

/** Map an engine-readiness string to a chip tone. */
export function engineTone(engine: string): StatusTone {
  if (engine === 'ready') return 'success';
  if (engine === 'unconfigured') return 'warning';
  return 'neutral';
}

import { AlertTriangle, RotateCw } from 'lucide-react';

import { toErrorMessage } from '@/api/errors';
import { cn } from '@/lib/utils';

import { Button } from '../ui/button';

export interface ErrorBannerProps {
  title?: string;
  /** A string message, or any thrown value (normalized via toErrorMessage). */
  error?: unknown;
  onRetry?: () => void;
  className?: string;
}

/** An inline error surface with an optional graceful retry. */
export function ErrorBanner({
  title = 'Something went wrong',
  error,
  onRetry,
  className,
}: ErrorBannerProps) {
  const message = error === undefined ? undefined : toErrorMessage(error);
  return (
    <div
      role="alert"
      className={cn(
        'flex items-start gap-3 rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm',
        className,
      )}
    >
      <AlertTriangle className="mt-0.5 size-5 shrink-0 text-destructive" aria-hidden="true" />
      <div className="flex-1 space-y-1">
        <p className="font-medium text-destructive">{title}</p>
        {message ? <p className="text-destructive/90">{message}</p> : null}
      </div>
      {onRetry ? (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RotateCw className="size-4" />
          Retry
        </Button>
      ) : null}
    </div>
  );
}

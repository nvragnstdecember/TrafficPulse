import { AlertOctagon, Home, RotateCw } from 'lucide-react';
import { Link } from 'react-router-dom';

import { ROUTES } from '@/routes/paths';

import { Button } from '../ui/button';

export interface FullPageErrorProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  showHome?: boolean;
}

/** A full-height error surface used by the error boundary and route errors. */
export function FullPageError({
  title = 'Something went wrong',
  message = 'An unexpected error occurred while rendering this page.',
  onRetry,
  showHome = true,
}: FullPageErrorProps) {
  return (
    <div
      role="alert"
      className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-4 text-center"
    >
      <span className="flex size-14 items-center justify-center rounded-full bg-destructive/10 text-destructive">
        <AlertOctagon className="size-7" aria-hidden="true" />
      </span>
      <div className="space-y-1">
        <h1 className="text-lg font-semibold">{title}</h1>
        <p className="mx-auto max-w-md text-sm text-muted-foreground">{message}</p>
      </div>
      <div className="flex items-center gap-2">
        {onRetry ? (
          <Button onClick={onRetry}>
            <RotateCw className="size-4" />
            Try again
          </Button>
        ) : null}
        {showHome ? (
          <Button variant="outline" asChild>
            <Link to={ROUTES.dashboard}>
              <Home className="size-4" />
              Back to dashboard
            </Link>
          </Button>
        ) : null}
      </div>
    </div>
  );
}

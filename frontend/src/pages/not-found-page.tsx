import { Compass, Home } from 'lucide-react';
import { Link } from 'react-router-dom';

import { Button } from '@/components/ui/button';
import { ROUTES } from '@/routes/paths';

/** 404 — shown for any unmatched route. */
export default function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
      <span className="flex size-14 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <Compass className="size-7" aria-hidden="true" />
      </span>
      <div className="space-y-1">
        <p className="text-sm font-medium text-muted-foreground">404</p>
        <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
        <p className="mx-auto max-w-md text-sm text-muted-foreground">
          The page you’re looking for doesn’t exist or may have moved.
        </p>
      </div>
      <Button asChild>
        <Link to={ROUTES.dashboard}>
          <Home className="size-4" />
          Back to dashboard
        </Link>
      </Button>
    </div>
  );
}

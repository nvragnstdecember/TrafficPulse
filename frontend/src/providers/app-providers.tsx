import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/toaster';
import { ErrorBoundary } from '@/components/common/error-boundary';
import { FullPageError } from '@/components/common/full-page-error';

import { QueryProvider } from './query-provider';
import { ThemeProvider } from './theme-provider';

/**
 * The single provider stack (H7B): a last-resort error boundary, theme, data
 * fetching, tooltips, and the toast surface — composed once and wrapped around
 * the whole app. The top-level boundary's fallback avoids router-dependent
 * navigation (router errors are handled inside the routes by `RouteError`).
 */
export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary
      fallback={() => <FullPageError onRetry={() => window.location.reload()} showHome={false} />}
    >
      <ThemeProvider>
        <QueryProvider>
          <TooltipProvider delayDuration={200}>
            {children}
            <Toaster />
          </TooltipProvider>
        </QueryProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

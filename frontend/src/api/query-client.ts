import { MutationCache, QueryCache, QueryClient } from '@tanstack/react-query';

import { ApiError } from './errors';

export interface QueryClientOptions {
  /**
   * Global failure handler (e.g. surface a toast). Not called for canceled
   * requests — a navigation-cancelled query is not a user-facing error.
   */
  onError?: (error: unknown) => void;
}

/**
 * TanStack Query configuration (H7B).
 *
 * Caching, retry, and refetch policy live here once. Retries are limited and
 * skip non-retryable failures (4xx and canceled requests), so a bad request or a
 * navigation-cancelled query fails fast instead of hammering the backend. A
 * single global `onError` powers app-wide error surfacing.
 */
export function createQueryClient(options: QueryClientOptions = {}): QueryClient {
  const handleError = (error: unknown) => {
    if (error instanceof ApiError && error.isCanceled) return;
    options.onError?.(error);
  };

  return new QueryClient({
    queryCache: new QueryCache({ onError: handleError }),
    mutationCache: new MutationCache({ onError: handleError }),
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          if (error instanceof ApiError && !error.isRetryable) return false;
          return failureCount < 2;
        },
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
      },
      mutations: {
        retry: false,
      },
    },
  });
}

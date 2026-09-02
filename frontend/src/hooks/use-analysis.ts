import { useQuery } from '@tanstack/react-query';

import { ApiError } from '@/api/errors';
import { queryKeys } from '@/api/query-keys';
import { analysisService } from '@/services/analysis.service';

/**
 * A finished run's helmet analysis, or `null` when the run has none.
 *
 * A 404 is **not** an error here. A run legitimately has no analysis when the
 * deployment configured none, when the run was enforcing the no-helmet rule instead
 * (and so carries its own helmet surface), or when it was recovered after a restart --
 * the fold is derived in-process and deliberately not persisted. Mapping that to `null`
 * rather than to an error state is what lets the panel simply not render, instead of
 * showing a failure for a configuration that is working exactly as intended.
 *
 * Enabled only once a job id exists and the run has finished: the fold is written when
 * the run completes, so asking earlier can only 404.
 */
export function useHelmetAnalysis(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.helmetAnalysis.forJob(jobId ?? 'none'),
    enabled: Boolean(jobId) && enabled,
    queryFn: async ({ signal }) => {
      try {
        return await analysisService.getHelmetAnalysis(jobId as string, signal);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    // The fold is immutable once written: a finished run's observations never change.
    staleTime: Infinity,
    retry: false,
  });
}

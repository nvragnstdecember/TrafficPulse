import { useQuery } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import { systemService } from '@/services/system.service';

/** Backend health + engine readiness (polled for the status footer). */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => systemService.getHealth(signal),
    staleTime: 15_000,
    refetchInterval: 30_000,
  });
}

/** Aggregate job counts + the latest engine metrics. */
export function useMetrics() {
  return useQuery({
    queryKey: queryKeys.metrics,
    queryFn: ({ signal }) => systemService.getMetrics(signal),
  });
}

/**
 * The repository analytics summary (H15) — the dashboard's single data source.
 *
 * Polled gently: the figures describe a whole repository, so they change on the
 * timescale of a processing run, not of a frame. One request serves every section.
 */
export function useAnalytics() {
  return useQuery({
    queryKey: queryKeys.analytics,
    queryFn: ({ signal }) => systemService.getAnalytics(signal),
    staleTime: 15_000,
    refetchInterval: 30_000,
  });
}

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

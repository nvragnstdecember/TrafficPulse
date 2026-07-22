import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import { type HealthResponse, type MetricsResponse } from '@/api/types';

/**
 * System service (H7B): health + engine/job metrics.
 *
 * Services are the only layer that names endpoints. They accept an optional
 * `AbortSignal` so callers (and TanStack Query) can cancel in flight.
 */
export const systemService = {
  getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return apiClient.get<HealthResponse>(endpoints.health, { signal });
  },
  getMetrics(signal?: AbortSignal): Promise<MetricsResponse> {
    return apiClient.get<MetricsResponse>(endpoints.metrics, { signal });
  },
};

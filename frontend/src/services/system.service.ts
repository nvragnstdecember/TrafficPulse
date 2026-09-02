import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import {
  type AnalyticsSummary,
  type HealthResponse,
  type MetricsResponse,
  type SystemPosture,
} from '@/api/types';

/**
 * System service (H7B): health, engine/job metrics, and the analytics summary.
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
  /** The whole dashboard in one request (H15) — no per-widget round-trips. */
  getAnalytics(signal?: AbortSignal): Promise<AnalyticsSummary> {
    return apiClient.get<AnalyticsSummary>(endpoints.analyticsSummary, { signal });
  },
  /**
   * What the deployment can honestly claim, per capability.
   *
   * Separate from health on purpose: health says the service is working, this says
   * what its configuration entitles anyone to conclude from the output.
   */
  getPosture(signal?: AbortSignal): Promise<SystemPosture> {
    return apiClient.get<SystemPosture>(endpoints.posture, { signal });
  },
};

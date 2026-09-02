import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import { type HelmetAnalysis } from '@/api/types';

/**
 * Helmet-analysis service: a finished run's per-rider classification.
 *
 * Kept out of `events.service` deliberately. An analysis produces no confirmed event,
 * and serving it from the same module as events is the first step towards a caller
 * treating the two as interchangeable — which is exactly the conflation the whole
 * analysis/rule split exists to prevent.
 */
export const analysisService = {
  /**
   * One job's helmet analysis.
   *
   * 404s for a job that declared no analysis, has not finished, or was recovered after
   * a restart (the fold is derived in-process, never persisted). Callers treat that as
   * "no analysis for this run", not as an error worth alarming about.
   */
  getHelmetAnalysis(jobId: string, signal?: AbortSignal): Promise<HelmetAnalysis> {
    return apiClient.get<HelmetAnalysis>(endpoints.helmetAnalysis(jobId), { signal });
  },
};

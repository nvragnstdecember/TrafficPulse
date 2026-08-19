import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import { type EventListParams } from '@/api/query-keys';
import {
  type ConfirmedEvent,
  type EventListResponse,
  type EvidenceManifest,
  type ReviewDecisionRequest,
  type ReviewResponse,
} from '@/api/types';

/**
 * Events + evidence service (H7B): list/detail events and fetch evidence
 * manifests (references only — the backend renders no media).
 */
export const eventsService = {
  list(params: EventListParams, signal?: AbortSignal): Promise<EventListResponse> {
    return apiClient.get<EventListResponse>(endpoints.events, {
      query: {
        video_id: params.videoId,
        job_id: params.jobId,
        limit: params.limit,
        offset: params.offset,
        sort: params.sort,
      },
      signal,
    });
  },
  get(eventId: string, signal?: AbortSignal): Promise<ConfirmedEvent> {
    return apiClient.get<ConfirmedEvent>(endpoints.event(eventId), { signal });
  },
  getEvidence(eventId: string, signal?: AbortSignal): Promise<EvidenceManifest> {
    return apiClient.get<EvidenceManifest>(endpoints.evidence(eventId), { signal });
  },
  /** The event's review case + its full append-only audit history (H9). */
  getReview(eventId: string, signal?: AbortSignal): Promise<ReviewResponse> {
    return apiClient.get<ReviewResponse>(endpoints.review(eventId), { signal });
  },
  /** Record one analyst action; returns the resulting case + history. */
  decide(eventId: string, request: ReviewDecisionRequest): Promise<ReviewResponse> {
    return apiClient.post<ReviewResponse>(endpoints.review(eventId), request);
  },
};

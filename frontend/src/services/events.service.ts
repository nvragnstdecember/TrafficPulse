import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import { type EventListParams } from '@/api/query-keys';
import { type ConfirmedEvent, type EventListResponse, type EvidenceManifest } from '@/api/types';

/**
 * Events + evidence service (H7B): list/detail events and fetch evidence
 * manifests (references only — the backend renders no media).
 */
export const eventsService = {
  list(params: EventListParams, signal?: AbortSignal): Promise<EventListResponse> {
    return apiClient.get<EventListResponse>(endpoints.events, {
      query: {
        video_id: params.videoId,
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
};

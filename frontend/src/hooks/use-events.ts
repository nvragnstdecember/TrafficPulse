import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { ApiError } from '@/api/errors';
import { type EventListParams, queryKeys } from '@/api/query-keys';
import { eventsService } from '@/services/events.service';

export interface EventsListOptions {
  /**
   * Poll interval in ms while a job is processing (H7D). Events become listable
   * as the run confirms them, so the list refetches on this cadence and stops
   * (pass `false`) once processing is no longer active.
   */
  refetchInterval?: number | false;
}

/** Paginated event list; keeps the previous page visible while fetching the next. */
export function useEventsList(params: EventListParams, options?: EventsListOptions) {
  return useQuery({
    queryKey: queryKeys.events.list(params),
    queryFn: ({ signal }) => eventsService.list(params, signal),
    placeholderData: keepPreviousData,
    refetchInterval: options?.refetchInterval ?? false,
  });
}

/** Full confirmed-event detail; disabled until an id is provided. */
export function useEvent(eventId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.events.detail(eventId ?? ''),
    queryFn: ({ signal }) => eventsService.get(eventId as string, signal),
    enabled: Boolean(eventId),
  });
}

/**
 * Evidence manifest for an event; disabled until an id is provided.
 *
 * Evidence can lag its event (delayed generation), so a not-yet-available
 * manifest surfaces as an error the caller can retry (H7D). A definite 4xx
 * ("not there") is not retried automatically — the UI offers a manual retry —
 * while transient failures use the client's exponential backoff.
 */
export function useEvidence(eventId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.evidence.detail(eventId ?? ''),
    queryFn: ({ signal }) => eventsService.getEvidence(eventId as string, signal),
    enabled: Boolean(eventId),
    retry: (failureCount, error) => {
      if (error instanceof ApiError && !error.isRetryable) return false;
      return failureCount < 2;
    },
  });
}

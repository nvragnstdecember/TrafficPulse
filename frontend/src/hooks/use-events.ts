import { keepPreviousData, useQuery } from '@tanstack/react-query';

import { type EventListParams, queryKeys } from '@/api/query-keys';
import { eventsService } from '@/services/events.service';

/** Paginated event list; keeps the previous page visible while fetching the next. */
export function useEventsList(params: EventListParams) {
  return useQuery({
    queryKey: queryKeys.events.list(params),
    queryFn: ({ signal }) => eventsService.list(params, signal),
    placeholderData: keepPreviousData,
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

/** Evidence manifest for an event; disabled until an id is provided. */
export function useEvidence(eventId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.evidence.detail(eventId ?? ''),
    queryFn: ({ signal }) => eventsService.getEvidence(eventId as string, signal),
    enabled: Boolean(eventId),
  });
}

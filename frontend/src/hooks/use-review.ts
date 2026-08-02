import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import { type ReviewAction, type ReviewResponse } from '@/api/types';
import { toErrorMessage } from '@/api/errors';
import { eventsService } from '@/services/events.service';
import { notify } from '@/store/notifications-store';

/**
 * One event's review case + audit history, and the mutation that records a
 * decision (H9).
 *
 * Server state only — there is deliberately no client store mirroring the review.
 * A decision is a fact on disk, and keeping a second copy in a zustand store would
 * reintroduce exactly the drift the append-only backend design exists to prevent.
 * The mutation seeds the response straight into the cache, so the panel updates
 * from the server's own answer rather than from an optimistic guess that could
 * disagree with what was actually persisted.
 */
export function useReview(eventId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.review.detail(eventId ?? ''),
    queryFn: ({ signal }) => eventsService.getReview(eventId as string, signal),
    enabled: Boolean(eventId),
  });
}

export interface DecideInput {
  eventId: string;
  action: ReviewAction;
  reviewer?: string;
  note?: string | null;
  reason?: string | null;
}

export function useDecideReview() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ eventId, ...request }: DecideInput) =>
      eventsService.decide(eventId, request),
    onSuccess: (response: ReviewResponse, variables) => {
      queryClient.setQueryData(queryKeys.review.detail(variables.eventId), response);
      // The list carries each row's review_status for badging and filtering, so it
      // has to be refetched for the badge to follow the decision.
      void queryClient.invalidateQueries({ queryKey: queryKeys.events.all });
    },
    onError: (error) => {
      // A 409 here means the case moved on — usually another reviewer. Surfacing it
      // is the point: silently swallowing it would let an analyst believe their
      // decision was recorded when somebody else's stands.
      notify({ title: 'Could not record the decision', description: toErrorMessage(error) });
    },
  });
}

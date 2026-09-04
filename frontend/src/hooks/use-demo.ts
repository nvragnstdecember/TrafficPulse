import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import {
  type ExpectationComparison,
  type ExpectationDeclaration,
  type ExpectationRecord,
} from '@/api/types';
import { ApiError } from '@/api/errors';
import { demoService } from '@/services/demo.service';

/**
 * Controlled-demonstration hooks: the declared expectation and its comparison.
 *
 * The declaration is client state the *server* owns, so it lives in react-query
 * like every other resource rather than in a store — which is also what makes the
 * demonstration reproducible across a refresh, a restart, or a different machine.
 */

/**
 * A video's declared expectation.
 *
 * A video with no declaration 404s, which is a *state* (offer to declare one) rather
 * than a failure, so it is swallowed into `null` and never retried. Any other error
 * surfaces normally.
 */
export function useExpectation(videoId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.expectations.forVideo(videoId ?? ''),
    enabled: Boolean(videoId),
    retry: false,
    queryFn: async ({ signal }): Promise<ExpectationRecord | null> => {
      try {
        return await demoService.getExpectation(videoId as string, signal);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
  });
}

/**
 * Expected beside detected, for one run.
 *
 * Only asked for once a run has finished: the comparison reads persisted events, so
 * requesting it mid-run would report a partial answer as if it were the result.
 */
export function useExpectationComparison(
  videoId: string | undefined,
  jobId: string | undefined,
  enabled: boolean,
) {
  return useQuery({
    queryKey: queryKeys.expectations.comparison(videoId ?? '', jobId),
    enabled: Boolean(videoId) && enabled,
    queryFn: ({ signal }): Promise<ExpectationComparison> =>
      demoService.comparison(videoId as string, jobId, signal),
  });
}

export interface DeclareInput {
  videoId: string;
  declaration: ExpectationDeclaration;
}

/**
 * Declare (or re-declare) what a controlled clip contains.
 *
 * Invalidates the comparison as well as the declaration: the expected column has
 * changed, and a stale table would show a matched row for a family nobody claims
 * any more. It deliberately does **not** invalidate events — declaring cannot change
 * what was confirmed, and refetching them would suggest it might.
 */
export function useDeclareExpectation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ videoId, declaration }: DeclareInput) =>
      demoService.declare(videoId, declaration),
    onSuccess: (record, { videoId }) => {
      queryClient.setQueryData(queryKeys.expectations.forVideo(videoId), record);
      void queryClient.invalidateQueries({ queryKey: queryKeys.expectations.all });
    },
  });
}

/** Withdraw a declaration. Confirmed events are untouched, and stay listed. */
export function useWithdrawExpectation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (videoId: string) => demoService.withdraw(videoId),
    onSuccess: (_result, videoId) => {
      queryClient.setQueryData(queryKeys.expectations.forVideo(videoId), null);
      void queryClient.invalidateQueries({ queryKey: queryKeys.expectations.all });
    },
  });
}

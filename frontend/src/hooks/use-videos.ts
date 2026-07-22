import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import { type JobStatusResponse } from '@/api/types';
import {
  type StartProcessingInput,
  type UploadVideoInput,
  videosService,
} from '@/services/videos.service';

/** Poll cadence while a job is still active, in ms. */
export const JOB_POLL_INTERVAL_MS = 1500;

/**
 * One job's status. When `poll` is set, it auto-refetches while the job is
 * pending/running and stops once it is terminal (succeeded/failed/cancelled) —
 * no manual interval wiring. Transient poll failures fall back to the client's
 * exponential retry backoff, and polling resumes automatically on reconnect.
 */
export function useJob(jobId: string | undefined, options?: { poll?: boolean }) {
  return useQuery({
    queryKey: queryKeys.jobs.detail(jobId ?? ''),
    queryFn: ({ signal }) => videosService.getJob(jobId as string, signal),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      if (!options?.poll) return false;
      const status = query.state.data?.status;
      return status === 'pending' || status === 'running' ? JOB_POLL_INTERVAL_MS : false;
    },
  });
}

/** Cancel a running job; seeds the returned status into the cache (H7D). */
export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => videosService.cancelJob(jobId),
    onSuccess: (status: JobStatusResponse) => {
      queryClient.setQueryData(queryKeys.jobs.detail(status.job_id), status);
      void queryClient.invalidateQueries({ queryKey: queryKeys.jobs.all });
      void queryClient.invalidateQueries({ queryKey: queryKeys.metrics });
    },
  });
}

/** Upload a source video; refreshes metrics on success. */
export function useUploadVideo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: UploadVideoInput) => videosService.upload(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.metrics });
    },
  });
}

/** Start a processing job; refreshes jobs + metrics on success. */
export function useStartProcessing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: StartProcessingInput) => videosService.startProcessing(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.jobs.all });
      void queryClient.invalidateQueries({ queryKey: queryKeys.metrics });
    },
  });
}

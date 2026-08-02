import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { type VideoListParams, queryKeys } from '@/api/query-keys';
import { type JobStatusResponse } from '@/api/types';
import { shouldPollJob } from '@/lib/job';
import {
  type StartProcessingInput,
  type UploadVideoInput,
  videosService,
} from '@/services/videos.service';

/** Poll cadence while a job is still active, in ms. */
export const JOB_POLL_INTERVAL_MS = 1500;

/**
 * The historical video library (H11): every stored video, newest first.
 *
 * Metadata only — a page of this costs no event, evidence, or overlay fetch, so it
 * stays cheap however large the repository grows. A video's analysis is loaded by
 * the existing event/evidence queries once one is opened.
 */
export function useVideoLibrary(params: VideoListParams = {}) {
  return useQuery({
    queryKey: queryKeys.videos.list(params),
    queryFn: ({ signal }) => videosService.list(params, signal),
  });
}

/**
 * One job's status. When `poll` is set, it auto-refetches until the job has fully
 * settled — which means both the run *and* its annotated-video render, since the
 * overlay resolves after the job reaches `succeeded` (see `shouldPollJob`). No
 * manual interval wiring. Transient poll failures fall back to the client's
 * exponential retry backoff, and polling resumes automatically on reconnect.
 */
export function useJob(jobId: string | undefined, options?: { poll?: boolean }) {
  return useQuery({
    queryKey: queryKeys.jobs.detail(jobId ?? ''),
    queryFn: ({ signal }) => videosService.getJob(jobId as string, signal),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      if (!options?.poll) return false;
      return shouldPollJob(query.state.data) ? JOB_POLL_INTERVAL_MS : false;
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.videos.all });
    },
  });
}

/** Upload a source video; refreshes metrics + the library on success. */
export function useUploadVideo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: UploadVideoInput) => videosService.upload(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.metrics });
      // A new upload is a new library row; without this the library would be stale
      // for whoever returns to it from the workspace.
      void queryClient.invalidateQueries({ queryKey: queryKeys.videos.all });
    },
  });
}

/** Start a processing job; refreshes jobs + metrics + the library on success. */
export function useStartProcessing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: StartProcessingInput) => videosService.startProcessing(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.jobs.all });
      void queryClient.invalidateQueries({ queryKey: queryKeys.metrics });
      void queryClient.invalidateQueries({ queryKey: queryKeys.videos.all });
    },
  });
}

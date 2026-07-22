import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import {
  type StartProcessingInput,
  type UploadVideoInput,
  videosService,
} from '@/services/videos.service';

/**
 * One job's status. When `poll` is set, it auto-refetches while the job is
 * pending/running and stops once it is terminal — no manual interval wiring.
 */
export function useJob(jobId: string | undefined, options?: { poll?: boolean }) {
  return useQuery({
    queryKey: queryKeys.jobs.detail(jobId ?? ''),
    queryFn: ({ signal }) => videosService.getJob(jobId as string, signal),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      if (!options?.poll) return false;
      const status = query.state.data?.status;
      return status === 'pending' || status === 'running' ? 2000 : false;
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

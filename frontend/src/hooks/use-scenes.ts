import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { queryKeys } from '@/api/query-keys';
import { type SceneDraft, type SceneSummary, type StoredScene } from '@/api/types';
import { ApiError } from '@/api/errors';
import { scenesService } from '@/services/scenes.service';

/**
 * A video's calibrated scene (H12/H13).
 *
 * An uncalibrated video 404s, which is a *state* rather than a failure — the
 * workspace renders "not calibrated yet" and offers the drawing surface. So the 404
 * is swallowed into `null` here and never retried, while any other error still
 * surfaces normally.
 */
export function useVideoScene(videoId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.scenes.forVideo(videoId ?? ''),
    enabled: Boolean(videoId),
    retry: false,
    queryFn: async ({ signal }): Promise<SceneSummary | null> => {
      try {
        return await scenesService.getForVideo(videoId as string, signal);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
  });
}

/**
 * One stored scene revision, by the hash events carry.
 *
 * What makes a saved calibration reloadable rather than write-only: the analyst can
 * see the geometry that is actually bound to this video, correct one polygon without
 * redrawing the rest, and check that a stored revision is what a demonstration claims
 * it is. Enabled only when a hash is known, so an uncalibrated video fetches nothing.
 */
export function useSceneRevision(sceneHash: string | null | undefined) {
  return useQuery({
    queryKey: [...queryKeys.scenes.all, 'revision', sceneHash ?? ''] as const,
    enabled: Boolean(sceneHash),
    // A revision is immutable — content-addressed — so it never needs refetching.
    staleTime: Infinity,
    queryFn: ({ signal }): Promise<StoredScene> =>
      scenesService.get(sceneHash as string, signal),
  });
}

export interface CalibrateInput {
  videoId: string;
  draft: SceneDraft;
}

/**
 * Author and bind a video's scene.
 *
 * Invalidates the video library and the video's own row as well as the scene:
 * calibrating changes which violations that video supports, and a stale
 * `supported_violations` would offer rules the run would then refuse.
 */
export function useCalibrateScene() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ videoId, draft }: CalibrateInput) =>
      scenesService.calibrate(videoId, draft),
    onSuccess: (summary, { videoId }) => {
      queryClient.setQueryData(queryKeys.scenes.forVideo(videoId), summary);
      void queryClient.invalidateQueries({ queryKey: queryKeys.videos.all });
    },
  });
}

/** Check a draft without saving it; returns the errors and what it would unlock. */
export function useValidateScene() {
  return useMutation({
    mutationFn: ({ videoId, draft }: CalibrateInput) => scenesService.validate(videoId, draft),
  });
}

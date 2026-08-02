import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import {
  type SceneDraft,
  type SceneSummary,
  type SceneValidationResponse,
} from '@/api/types';

/**
 * Scene calibration service (H12/H13).
 *
 * Three operations, mapping one-to-one onto the API's scene resource: read a
 * video's binding, author it, and check a draft without committing to it. The
 * service carries no drawing or geometry logic — that lives in `lib/calibration`
 * and the calibrator component.
 */
export const scenesService = {
  /** The scene a video is calibrated against; 404s when it has none. */
  getForVideo(videoId: string, signal?: AbortSignal): Promise<SceneSummary> {
    return apiClient.get<SceneSummary>(endpoints.videoScene(videoId), { signal });
  },
  /**
   * Author this video's scene and bind it. Idempotent: an unchanged drawing
   * rebuilds identical content and changes nothing server-side.
   */
  calibrate(videoId: string, draft: SceneDraft, signal?: AbortSignal): Promise<SceneSummary> {
    return apiClient.put<SceneSummary>(endpoints.videoScene(videoId), draft, { signal });
  },
  /**
   * Check a draft without saving it, and learn which violations it would unlock.
   *
   * Returns 200 with `valid: false` for an unusable drawing rather than an error,
   * so a half-finished calibration is a state the UI renders while the analyst is
   * still drawing.
   */
  validate(
    videoId: string,
    draft: SceneDraft,
    signal?: AbortSignal,
  ): Promise<SceneValidationResponse> {
    return apiClient.post<SceneValidationResponse>(
      endpoints.videoSceneValidate(videoId),
      draft,
      { signal },
    );
  },
};

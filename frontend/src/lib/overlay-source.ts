import { endpoints } from '@/api/endpoints';
import { type JobStatusResponse } from '@/api/types';
import { env } from '@/lib/env';

/**
 * The playable overlay (annotated) video URL for a finished job, or `null`.
 *
 * The backend renders the source clip with detection boxes, association lines,
 * observation state, and confirmed-violation banners drawn on every frame, and
 * serves it once the run succeeds. Until then (running/failed, or a run that
 * produced no overlay metadata) there is nothing to show and the caller falls back
 * to the original uploaded video — which is always preserved separately.
 */
export function overlayVideoSource(job: JobStatusResponse | undefined | null): string | null {
  if (!job || job.status !== 'succeeded' || !job.overlay_available) return null;
  return `${env.apiBaseUrl}${endpoints.jobOverlay(job.job_id)}`;
}

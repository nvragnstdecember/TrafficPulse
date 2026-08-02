import { endpoints } from '@/api/endpoints';
import { type JobStatusResponse } from '@/api/types';
import { env } from '@/lib/env';

import { overlayVideoSource } from './overlay-source';

/**
 * The stored source video's URL — the upload as it was received, never modified.
 *
 * Exists because the workspace's original playback source is a browser object URL
 * for the `File` the analyst picked, which dies with the tab. A video opened from
 * the historical library was never picked in this session, so without a server-side
 * source there is nothing to play.
 */
export function videoMediaSource(videoId: string | null | undefined): string | null {
  if (!videoId) return null;
  return `${env.apiBaseUrl}${endpoints.videoMedia(videoId)}`;
}

export interface VideoSourceInput {
  job: JobStatusResponse | undefined | null;
  /** Object URL for a file picked in this session, when there is one. */
  objectUrl: string | null;
  videoId: string | null | undefined;
  /** Whether the backend still holds the stored file (library `media_available`). */
  mediaAvailable?: boolean;
}

/**
 * The one video URL the whole workspace plays — player, timeline, thumbnails, and
 * evidence viewer all read this single source.
 *
 * Preference order, and why:
 *
 * 1. **the annotated overlay**, once rendered — it is the explainable artifact, and
 *    showing the raw clip when boxes exist hides the reasoning the analyst is there
 *    to check;
 * 2. **the local object URL**, when this session picked the file — already in
 *    memory, so playback starts without a download;
 * 3. **the stored source over HTTP** — what makes a historical video playable at
 *    all, and the honest fallback for a run that produced no overlay.
 *
 * `null` means there is genuinely nothing to play (no job, no local file, and the
 * stored bytes are gone), which the player renders as an empty state rather than a
 * broken element.
 */
export function workspaceVideoSource({
  job,
  objectUrl,
  videoId,
  mediaAvailable = true,
}: VideoSourceInput): string | null {
  const overlay = overlayVideoSource(job);
  if (overlay) return overlay;
  if (objectUrl) return objectUrl;
  return mediaAvailable ? videoMediaSource(videoId) : null;
}

import { type VideoSummary } from '@/api/types';
import { type StatusTone } from '@/components/common/status-chip';

/**
 * Pure presentation logic for the historical video library (H11).
 *
 * Kept out of the components so the rules — what a video's state is called, when it
 * can be opened, how review progress reads — are testable without rendering, in the
 * same spirit as `lib/job.ts` for the processing lifecycle.
 */

/** How a video's processing state reads in the library. */
export function libraryStatusLabel(video: VideoSummary): string {
  switch (video.status) {
    case null:
      return 'Not processed';
    case 'pending':
      return 'Queued';
    case 'running':
      return 'Processing';
    case 'succeeded':
      return 'Analysed';
    case 'failed':
      return 'Failed';
    case 'cancelled':
      return 'Cancelled';
  }
}

export function libraryStatusTone(video: VideoSummary): StatusTone {
  switch (video.status) {
    case 'succeeded':
      return 'success';
    case 'running':
    case 'pending':
      return 'info';
    case 'failed':
      return 'error';
    case 'cancelled':
    case null:
      return 'neutral';
  }
}

/**
 * Review progress for one video, or `null` when it has no events to review.
 *
 * Reports events *acted on*, which is what the backend can answer from a directory
 * listing — deliberately not "decided", because opening a case also writes a
 * journal and only a per-event fold can tell the two apart.
 */
export function reviewProgressLabel(video: VideoSummary): string | null {
  if (video.event_count === 0) return null;
  if (video.events_reviewed === 0) return `${video.event_count} to review`;
  if (video.events_reviewed >= video.event_count) return 'All reviewed';
  return `${video.events_reviewed} of ${video.event_count} reviewed`;
}

/**
 * Whether opening this video can show an analysis, rather than starting one.
 *
 * A video with a succeeded run has events and evidence waiting; anything else means
 * opening it will (re)process, which is a different promise to make in the UI.
 */
export function hasAnalysis(video: VideoSummary): boolean {
  return video.status === 'succeeded';
}

/**
 * Whether the library can offer to open this video at all.
 *
 * A video whose stored file is gone and that never produced an overlay has nothing
 * to play and nothing to re-run against — the row stays, because its events and
 * review history are still valid, but opening it would strand the analyst on an
 * empty player.
 */
export function canOpen(video: VideoSummary): boolean {
  return video.media_available || video.overlay_available;
}

/** Case-insensitive filename/id match, for narrowing a large library. */
export function filterVideos(videos: VideoSummary[], query: string): VideoSummary[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return videos;
  return videos.filter(
    (video) =>
      video.filename.toLowerCase().includes(needle) ||
      video.video_id.toLowerCase().includes(needle),
  );
}

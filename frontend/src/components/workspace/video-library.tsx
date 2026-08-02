import { FileVideo, FolderOpen, Play, RotateCw } from 'lucide-react';
import { useState } from 'react';

import { type VideoSummary } from '@/api/types';
import { useVideoLibrary } from '@/hooks/use-videos';
import { formatBytes, formatDateTime, formatDuration, formatNumber } from '@/lib/format';
import {
  canOpen,
  filterVideos,
  hasAnalysis,
  libraryStatusLabel,
  libraryStatusTone,
  reviewProgressLabel,
} from '@/lib/library';
import { cn } from '@/lib/utils';

import { EmptyState } from '../common/empty-state';
import { ErrorBanner } from '../common/error-banner';
import { SearchInput } from '../common/search-input';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';

/** Show the search field only once a library is big enough to need narrowing. */
const SEARCH_THRESHOLD = 5;

export interface VideoLibraryProps {
  onOpen: (video: VideoSummary) => void;
  className?: string;
}

function VideoRow({ video, onOpen }: { video: VideoSummary; onOpen: () => void }) {
  const openable = canOpen(video);
  const review = reviewProgressLabel(video);
  const facts = [
    formatDateTime(video.uploaded_at),
    video.duration_seconds != null ? formatDuration(video.duration_seconds) : null,
    `${video.width}×${video.height}`,
    formatBytes(video.size_bytes),
  ].filter((fact): fact is string => Boolean(fact));

  return (
    <li className="flex flex-wrap items-center gap-3 rounded-md border p-3 sm:flex-nowrap">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
        <FileVideo className="size-4" aria-hidden="true" />
      </span>

      <div className="min-w-0 flex-1 space-y-1">
        <p className="truncate text-sm font-medium">{video.filename}</p>
        <p className="truncate text-xs text-muted-foreground">{facts.join(' · ')}</p>
      </div>

      <div className="flex shrink-0 items-center gap-3">
        <div className="hidden text-right sm:block">
          <p className="text-sm tabular-nums">
            {video.event_count > 0
              ? `${formatNumber(video.event_count)} event${video.event_count === 1 ? '' : 's'}`
              : hasAnalysis(video)
                ? 'No violations'
                : '—'}
          </p>
          {review ? <p className="text-2xs text-muted-foreground">{review}</p> : null}
        </div>
        <StatusChip tone={libraryStatusTone(video)} label={libraryStatusLabel(video)} />
        <Button
          size="sm"
          variant={hasAnalysis(video) ? 'default' : 'outline'}
          onClick={onOpen}
          disabled={!openable}
          // The stored file is gone and no overlay was ever rendered, so opening it
          // would land the analyst on an empty player. The row stays: its events and
          // review history are still valid.
          title={openable ? undefined : 'The stored video file is no longer available'}
          aria-label={`${hasAnalysis(video) ? 'Open' : 'Process'} ${video.filename}`}
        >
          {hasAnalysis(video) ? <Play className="size-4" /> : <RotateCw className="size-4" />}
          {hasAnalysis(video) ? 'Open' : 'Process'}
        </Button>
      </div>
    </li>
  );
}

/**
 * The historical video library (H11): every video the backend still holds.
 *
 * Closes the loop H10 opened. Recovery rebuilt the repository's indices after a
 * restart, but the workspace's only notion of "a video" was the one upload in this
 * browser's local storage — so persisted work existed and was unreachable. This
 * lists what the server actually has, and hands a chosen row to the workspace.
 *
 * Metadata only, by construction: the list renders from one `/api/videos` page and
 * fetches no events, evidence, or overlays. Those load when a video is opened,
 * through the queries that already serve the workspace.
 */
export function VideoLibrary({ onOpen, className }: VideoLibraryProps) {
  const [query, setQuery] = useState('');
  const { data, isLoading, isError, error, refetch } = useVideoLibrary();

  const videos = data?.items ?? [];
  const shown = filterVideos(videos, query);

  return (
    <Card className={className} aria-busy={isLoading}>
      <CardHeader className="flex-row items-center justify-between gap-3 p-4 pb-2">
        <CardTitle className="flex items-center gap-2">
          <FolderOpen className="size-4 text-muted-foreground" aria-hidden="true" />
          Video library
          {videos.length > 0 ? (
            <span className="text-xs font-normal text-muted-foreground tabular-nums">
              {shown.length === videos.length
                ? formatNumber(data?.total ?? videos.length)
                : `${shown.length} of ${videos.length}`}
            </span>
          ) : null}
        </CardTitle>
        {videos.length >= SEARCH_THRESHOLD ? (
          <SearchInput
            value={query}
            onValueChange={setQuery}
            placeholder="Filter by filename…"
            aria-label="Filter videos"
            containerClassName="w-48"
          />
        ) : null}
      </CardHeader>

      <CardContent className={cn('p-4 pt-2')}>
        {isError ? (
          <ErrorBanner
            title="Could not load the video library"
            error={error}
            onRetry={() => void refetch()}
          />
        ) : isLoading ? (
          <ul className="space-y-2" aria-label="Loading videos">
            {[0, 1, 2].map((row) => (
              <li key={row}>
                <Skeleton className="h-16 w-full" />
              </li>
            ))}
          </ul>
        ) : videos.length === 0 ? (
          <EmptyState
            icon={FileVideo}
            title="No videos yet"
            description="Videos you upload are kept on the server and listed here, so you can reopen an analysis without uploading it again."
            className="border-0 p-6"
          />
        ) : shown.length === 0 ? (
          <EmptyState
            title="No videos match that filter"
            description="Try a different filename."
            className="border-0 p-6"
          />
        ) : (
          <ul className="space-y-2" aria-label="Stored videos">
            {shown.map((video) => (
              <VideoRow key={video.video_id} video={video} onOpen={() => onOpen(video)} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

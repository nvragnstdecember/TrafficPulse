import { SkipBack, SkipForward, ZoomIn, ZoomOut } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';

import { cn } from '@/lib/utils';
import {
  type MarkerCluster,
  type TimelineMarker,
  type ViolationTone,
  clusterMarkers,
  formatClock,
  violationLabel,
} from '@/lib/workspace';

import { Button } from '../ui/button';
import { usePlayer } from './player-context';

export interface TimelineProps {
  markers: TimelineMarker[];
  duration: number;
  selectedEventId: string | null;
  onSelect: (eventId: string) => void;
}

const ZOOM_LEVELS = [1, 2, 4, 8];

const TONE_MARKER: Record<ViolationTone, string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-destructive',
  info: 'bg-primary',
  neutral: 'bg-muted-foreground',
};

function clusterTone(cluster: MarkerCluster): ViolationTone {
  // Highest-severity tone wins for a dense cluster.
  const tones = cluster.markers.map((m) => m.event.violationType);
  if (tones.some((t) => t === 'no_helmet' || t === 'red_light_jumping')) return 'error';
  if (tones.some((t) => t === 'wrong_way' || t === 'speeding' || t === 'triple_riding'))
    return 'warning';
  return 'info';
}

/**
 * The professional timeline beneath the player (H7C): a scrubbable track with the
 * current-position playhead, violation markers (overlaps clustered), hover
 * preview, click-to-seek, and zoom/scroll. Playback state comes from the shared
 * controller; selection is lifted to the caller.
 */
export function Timeline({ markers, duration, selectedEventId, onSelect }: TimelineProps) {
  const { state, controls } = usePlayer();
  const [zoom, setZoom] = useState(1);
  const [hovered, setHovered] = useState<MarkerCluster | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const clusters = useMemo(() => clusterMarkers(markers, 0.02 / zoom), [markers, zoom]);
  const playheadRatio = duration > 0 ? Math.min(1, state.currentTime / duration) : 0;
  const canSeek = duration > 0;

  const zoomIndex = ZOOM_LEVELS.indexOf(zoom);

  // Ordered marker times for prev/next violation navigation.
  const orderedMarkers = useMemo(() => [...markers].sort((a, b) => a.time - b.time), [markers]);
  const hasMarkers = orderedMarkers.length > 0;

  function seekFromClientX(clientX: number) {
    const track = trackRef.current;
    if (!track || !canSeek) return;
    const rect = track.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    controls.seek(ratio * duration);
  }

  /** Jump to the nearest violation after (1) or before (-1) the playhead. */
  function jumpToViolation(direction: 1 | -1) {
    if (!hasMarkers) return;
    const now = state.currentTime;
    const epsilon = 0.05;
    const target =
      direction > 0
        ? orderedMarkers.find((marker) => marker.time > now + epsilon)
        : [...orderedMarkers].reverse().find((marker) => marker.time < now - epsilon);
    if (!target) return;
    onSelect(target.event.id);
    controls.seek(target.time);
  }

  return (
    <section aria-label="Timeline" className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          Timeline
          <span className="ml-2 font-mono tabular-nums text-foreground">
            {formatClock(state.currentTime)}
          </span>
        </p>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon"
            aria-label="Previous violation"
            disabled={!hasMarkers}
            onClick={() => jumpToViolation(-1)}
          >
            <SkipBack className="size-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            aria-label="Next violation"
            disabled={!hasMarkers}
            onClick={() => jumpToViolation(1)}
          >
            <SkipForward className="size-4" />
          </Button>
          <div className="mx-0.5 h-5 w-px bg-border" aria-hidden="true" />
          <Button
            variant="outline"
            size="icon"
            aria-label="Zoom out"
            disabled={zoomIndex <= 0}
            onClick={() => setZoom(ZOOM_LEVELS[Math.max(0, zoomIndex - 1)])}
          >
            <ZoomOut className="size-4" />
          </Button>
          <span className="w-8 text-center text-xs tabular-nums text-muted-foreground">
            {zoom}×
          </span>
          <Button
            variant="outline"
            size="icon"
            aria-label="Zoom in"
            disabled={zoomIndex >= ZOOM_LEVELS.length - 1}
            onClick={() => setZoom(ZOOM_LEVELS[Math.min(ZOOM_LEVELS.length - 1, zoomIndex + 1)])}
          >
            <ZoomIn className="size-4" />
          </Button>
        </div>
      </div>

      {hovered ? (
        <div className="rounded-md border bg-popover px-3 py-1.5 text-xs shadow-elevation-2">
          <span className="font-mono tabular-nums text-muted-foreground">
            {formatClock(hovered.time)}
          </span>
          <span className="ml-2">
            {hovered.markers.length > 1
              ? `${hovered.markers.length} events`
              : violationLabel(hovered.markers[0].violationType)}
          </span>
        </div>
      ) : null}

      <div className="overflow-x-auto pb-1">
        <div
          ref={trackRef}
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) seekFromClientX(event.clientX);
          }}
          className={cn(
            'relative h-12 rounded-md border bg-muted/40',
            canSeek ? 'cursor-pointer' : 'cursor-default',
          )}
          style={{ width: `${zoom * 100}%` }}
        >
          {/* playhead */}
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-0 z-10 h-full w-0.5 bg-primary"
            style={{ left: `${playheadRatio * 100}%` }}
          />

          {clusters.map((cluster) => {
            const isSelected = cluster.markers.some((m) => m.id === selectedEventId);
            const tone = clusterTone(cluster);
            const primary = cluster.markers[0];
            return (
              <button
                key={cluster.key}
                type="button"
                aria-label={`${violationLabel(primary.violationType)} at ${formatClock(cluster.time)}${
                  cluster.markers.length > 1 ? ` and ${cluster.markers.length - 1} more` : ''
                }`}
                title={`${formatClock(cluster.time)} — ${violationLabel(primary.violationType)}`}
                onMouseEnter={() => setHovered(cluster)}
                onMouseLeave={() => setHovered(null)}
                onFocus={() => setHovered(cluster)}
                onBlur={() => setHovered(null)}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(primary.event.id);
                  controls.seek(primary.time);
                }}
                aria-current={isSelected ? 'true' : undefined}
                className={cn(
                  'absolute top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full ring-offset-background transition-transform hover:scale-110',
                  isSelected
                    ? 'z-30 scale-125 ring-2 ring-primary ring-offset-2'
                    : 'z-20 ring-ring',
                  cluster.markers.length > 1 ? 'size-5' : 'size-3',
                  TONE_MARKER[tone],
                )}
                style={{ left: `${cluster.positionRatio * 100}%` }}
              >
                {cluster.markers.length > 1 ? (
                  <span className="text-2xs font-semibold text-white">
                    {cluster.markers.length}
                  </span>
                ) : null}
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}

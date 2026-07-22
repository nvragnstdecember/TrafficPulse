import {
  ChevronLeft,
  ChevronRight,
  FileVideo,
  Maximize,
  Minimize,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { type EvidenceManifest } from '@/api/types';
import { formatDateTime, formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import { type WorkspaceEvent, formatClock, violationLabel } from '@/lib/workspace';

import { CopyButton } from '../common/copy-button';
import { Button } from '../ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';

/** How far before/after the trigger the evidence frames sit, in seconds. */
export const EVIDENCE_WINDOW_SECONDS = 1.5;
const ZOOM_MIN = 1;
const ZOOM_MAX = 5;
const ZOOM_STEP = 0.5;

export interface EvidenceViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  event: WorkspaceEvent | null;
  evidence: EvidenceManifest | undefined;
  /** Object URL of the local video (the only real pixels), or null. */
  objectUrl: string | null;
  fps?: number | null;
}

interface EvidenceFrame {
  key: string;
  label: string;
  seconds: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * A focused evidence viewer (H7E) — a modal over the local video showing the
 * violation frames with zoom, pan, fullscreen, and frame navigation, beside the
 * manifest's typed references and the event's technical metadata.
 *
 * The backend renders no media, so the pixels come from the uploaded video at the
 * evidence media-times (trigger, and a window before/after); the manifest
 * artifacts are shown as the references they are. When the local file is not in
 * this session the viewer still shows the metadata and explains playback is
 * unavailable. Focus is trapped and restored by the underlying Dialog.
 */
export function EvidenceViewer({
  open,
  onOpenChange,
  event,
  evidence,
  objectUrl,
  fps,
}: EvidenceViewerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [activeKey, setActiveKey] = useState('trigger');
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const frameStep = fps && fps > 0 ? 1 / fps : 1 / 30;

  const frames = useMemo<EvidenceFrame[]>(() => {
    if (!event) return [];
    const t = event.mediaSeconds;
    return [
      { key: 'before', label: 'Before', seconds: Math.max(0, t - EVIDENCE_WINDOW_SECONDS) },
      { key: 'trigger', label: 'Trigger', seconds: t },
      { key: 'after', label: 'After', seconds: t + EVIDENCE_WINDOW_SECONDS },
    ];
  }, [event]);

  const activeFrame = frames.find((frame) => frame.key === activeKey) ?? frames[1] ?? frames[0];

  const seek = useCallback((seconds: number) => {
    const el = videoRef.current;
    if (el) el.currentTime = Math.max(0, seconds);
  }, []);

  // Reset transform + seek to the trigger whenever the viewer (re)opens.
  useEffect(() => {
    if (!open) return;
    setActiveKey('trigger');
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, [open, event?.id]);

  // Seek to the active frame's media time.
  useEffect(() => {
    if (open && activeFrame) seek(activeFrame.seconds);
  }, [open, activeFrame, seek]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const onChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const zoomBy = useCallback((delta: number) => {
    setZoom((z) => {
      const next = clamp(Number((z + delta).toFixed(2)), ZOOM_MIN, ZOOM_MAX);
      if (next === 1) setOffset({ x: 0, y: 0 });
      return next;
    });
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = stageRef.current;
    if (!el) return;
    if (typeof document !== 'undefined' && document.fullscreenElement) {
      document.exitFullscreen?.()?.catch?.(() => {});
    } else {
      el.requestFullscreen?.()?.catch?.(() => {});
    }
  }, []);

  const stepFrame = useCallback(
    (direction: 1 | -1) => {
      const el = videoRef.current;
      if (el) seek(el.currentTime + direction * frameStep);
    },
    [seek, frameStep],
  );

  if (!event) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Evidence — {violationLabel(event.violationType)}
            <span className="font-mono text-xs font-normal text-muted-foreground">{event.id}</span>
          </DialogTitle>
        </DialogHeader>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div className="space-y-2">
            <div
              ref={stageRef}
              className="relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-lg border bg-black"
              onWheel={(e) => {
                if (!objectUrl) return;
                e.preventDefault();
                zoomBy(e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP);
              }}
              onPointerDown={(e) => {
                if (zoom <= 1) return;
                dragRef.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y };
                (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
              }}
              onPointerMove={(e) => {
                const drag = dragRef.current;
                if (!drag) return;
                setOffset({ x: drag.ox + (e.clientX - drag.x), y: drag.oy + (e.clientY - drag.y) });
              }}
              onPointerUp={() => (dragRef.current = null)}
            >
              {objectUrl ? (
                <video
                  ref={videoRef}
                  src={objectUrl}
                  muted
                  playsInline
                  aria-label={`Evidence frame: ${activeFrame?.label} at ${formatClock(
                    activeFrame?.seconds,
                  )}`}
                  className={cn('h-full w-full origin-center', zoom > 1 && 'cursor-grab')}
                  style={{
                    transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                  }}
                />
              ) : (
                <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted-foreground">
                  <FileVideo className="size-8" aria-hidden="true" />
                  <p>Re-select the local video file to preview evidence frames.</p>
                </div>
              )}

              <span className="pointer-events-none absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-xs text-white">
                {activeFrame?.label} · {formatClock(activeFrame?.seconds)}
              </span>
            </div>

            {/* Controls */}
            <div className="flex flex-wrap items-center gap-1">
              <div className="flex items-center gap-1" role="group" aria-label="Evidence frames">
                {frames.map((frame) => (
                  <Button
                    key={frame.key}
                    variant={frame.key === activeKey ? 'secondary' : 'ghost'}
                    size="sm"
                    aria-pressed={frame.key === activeKey}
                    onClick={() => setActiveKey(frame.key)}
                  >
                    {frame.label}
                  </Button>
                ))}
              </div>
              <div className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
              <Button
                variant="ghost"
                size="icon"
                aria-label="Previous frame"
                disabled={!objectUrl}
                onClick={() => stepFrame(-1)}
              >
                <ChevronLeft className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Next frame"
                disabled={!objectUrl}
                onClick={() => stepFrame(1)}
              >
                <ChevronRight className="size-4" />
              </Button>
              <div className="ml-auto flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Zoom out"
                  disabled={!objectUrl || zoom <= ZOOM_MIN}
                  onClick={() => zoomBy(-ZOOM_STEP)}
                >
                  <ZoomOut className="size-4" />
                </Button>
                <span className="w-10 text-center text-xs tabular-nums text-muted-foreground">
                  {zoom.toFixed(1)}×
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Zoom in"
                  disabled={!objectUrl || zoom >= ZOOM_MAX}
                  onClick={() => zoomBy(ZOOM_STEP)}
                >
                  <ZoomIn className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Reset view"
                  disabled={!objectUrl}
                  onClick={resetView}
                >
                  <RotateCcw className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                  disabled={!objectUrl}
                  onClick={toggleFullscreen}
                >
                  {isFullscreen ? <Minimize className="size-4" /> : <Maximize className="size-4" />}
                </Button>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <aside className="space-y-3 overflow-y-auto text-sm" aria-label="Evidence metadata">
            <dl className="space-y-1.5">
              <MetaRow label="Event" value={event.id} copyable copyLabel="Copy event ID" />
              {evidence ? (
                <MetaRow
                  label="Package"
                  value={evidence.evidence_package_id}
                  copyable
                  copyLabel="Copy evidence ID"
                />
              ) : null}
              <MetaRow label="Camera" value={event.cameraId} />
              <MetaRow label="Rule" value={event.ruleId} />
              <MetaRow label="Confidence" value={formatPercent(event.confidence)} />
              <MetaRow label="Triggered" value={formatDateTime(event.triggerAt)} />
            </dl>

            <div className="space-y-1">
              <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">Artifacts</h4>
              {evidence ? (
                <ul className="divide-y rounded-md border">
                  {(
                    [
                      ['Before frame', evidence.before_frame],
                      ['Trigger frame', evidence.trigger_frame],
                      ['After frame', evidence.after_frame],
                      ['Clip', evidence.clip],
                      ['Trajectory', evidence.trajectory],
                      ['Plate crop', evidence.plate_crop],
                    ] as const
                  ).map(([label, artifact]) => (
                    <li key={label} className="flex items-center justify-between gap-2 px-2.5 py-1">
                      <span className="shrink-0 text-xs">{label}</span>
                      <span className="truncate font-mono text-2xs text-muted-foreground">
                        {artifact ? artifact.locator : '—'}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted-foreground">
                  The evidence manifest is not available for this event.
                </p>
              )}
            </div>
          </aside>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function MetaRow({
  label,
  value,
  copyable = false,
  copyLabel,
}: {
  label: string;
  value: string;
  copyable?: boolean;
  copyLabel?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-2xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="flex min-w-0 items-center gap-1">
        <span className="truncate font-mono text-xs">{value}</span>
        {copyable ? <CopyButton value={value} label={copyLabel ?? `Copy ${label}`} /> : null}
      </dd>
    </div>
  );
}

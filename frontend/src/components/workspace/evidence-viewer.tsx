import { Download, ImageOff, Maximize, Minimize, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { type EvidenceManifest } from '@/api/types';
import {
  type EvidenceFrame,
  evidenceFrames,
  evidencePackageUrl,
  manifestArtifacts,
  shortDigest,
} from '@/lib/evidence';
import { formatDateTime, formatPercent } from '@/lib/format';
import { cn } from '@/lib/utils';
import { type WorkspaceEvent, violationLabel } from '@/lib/workspace';

import { CopyButton } from '../common/copy-button';
import { Button } from '../ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';

const ZOOM_MIN = 1;
const ZOOM_MAX = 5;
const ZOOM_STEP = 0.5;

export interface EvidenceViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  event: WorkspaceEvent | null;
  evidence: EvidenceManifest | undefined;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * A focused evidence viewer (H7E, rebuilt on backend artifacts in H14).
 *
 * Shows the frames the **backend rendered** — the ones the engine actually picked,
 * drawn with the same overlay renderer as the annotated video and served under a
 * verified content hash — with zoom, pan, and fullscreen, beside the manifest's
 * references and the event's technical metadata.
 *
 * It infers nothing. Before H14 this component seeked the playing video to
 * `trigger ± 1.5s` and presented the result as evidence, which could differ from
 * the frames the manifest recorded. Now a frame appears only if the manifest
 * references a rendered artifact for it; when none exists (a repository written
 * before H14, or a render that failed) the viewer says so and still shows every
 * piece of metadata it has.
 */
export function EvidenceViewer({ open, onOpenChange, event, evidence }: EvidenceViewerProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [activeKind, setActiveKind] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [failed, setFailed] = useState(false);
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  const frames = useMemo<EvidenceFrame[]>(
    () => (event ? evidenceFrames(event.id, evidence) : []),
    [event, evidence],
  );

  const activeFrame =
    frames.find((frame) => frame.kind === activeKind) ??
    frames.find((frame) => frame.kind === 'trigger_frame') ??
    frames[0];

  // Reset the transform and fall back to the trigger frame whenever the viewer
  // (re)opens or the event changes.
  useEffect(() => {
    if (!open) return;
    setActiveKind(null);
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setFailed(false);
  }, [open, event?.id]);

  useEffect(() => setFailed(false), [activeFrame?.url]);

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

  if (!event) return null;

  const hasFrames = frames.length > 0;

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
                if (!activeFrame) return;
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
              {activeFrame && !failed ? (
                <img
                  key={activeFrame.url}
                  src={activeFrame.url}
                  alt={`Evidence frame: ${activeFrame.label}`}
                  onError={() => setFailed(true)}
                  className={cn(
                    'h-full w-full origin-center object-contain',
                    zoom > 1 && 'cursor-grab',
                  )}
                  style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}
                />
              ) : (
                <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted-foreground">
                  <ImageOff className="size-8" aria-hidden="true" />
                  <p>
                    {failed
                      ? 'The rendered evidence frame could not be loaded.'
                      : 'No rendered evidence frames exist for this event.'}
                  </p>
                </div>
              )}

              {activeFrame ? (
                <span className="pointer-events-none absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-xs text-white">
                  {activeFrame.label} · {shortDigest(activeFrame.artifact)}
                </span>
              ) : null}
            </div>

            {/* Controls */}
            <div className="flex flex-wrap items-center gap-1">
              <div className="flex items-center gap-1" role="group" aria-label="Evidence frames">
                {frames.map((frame) => (
                  <Button
                    key={frame.kind}
                    variant={frame.kind === activeFrame?.kind ? 'secondary' : 'ghost'}
                    size="sm"
                    aria-pressed={frame.kind === activeFrame?.kind}
                    onClick={() => setActiveKind(frame.kind)}
                  >
                    {frame.label}
                  </Button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Zoom out"
                  disabled={!activeFrame || zoom <= ZOOM_MIN}
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
                  disabled={!activeFrame || zoom >= ZOOM_MAX}
                  onClick={() => zoomBy(ZOOM_STEP)}
                >
                  <ZoomIn className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Reset view"
                  disabled={!activeFrame}
                  onClick={resetView}
                >
                  <RotateCcw className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                  disabled={!activeFrame}
                  onClick={toggleFullscreen}
                >
                  {isFullscreen ? <Minimize className="size-4" /> : <Maximize className="size-4" />}
                </Button>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <aside className="space-y-3 overflow-y-auto text-sm" aria-label="Evidence metadata">
            <Button asChild variant="outline" size="sm" className="w-full">
              <a href={evidencePackageUrl(event.id)} download>
                <Download className="size-4" />
                Download evidence package
              </a>
            </Button>

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
                  {manifestArtifacts(evidence).map(({ label, artifact }) => (
                    <li key={label} className="flex items-center justify-between gap-2 px-2.5 py-1">
                      <span className="shrink-0 text-xs">{label}</span>
                      <span
                        className="truncate font-mono text-2xs text-muted-foreground"
                        title={artifact?.sha256 ?? artifact?.locator ?? undefined}
                      >
                        {artifact ? shortDigest(artifact) : '—'}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-muted-foreground">
                  The evidence manifest is not available for this event.
                </p>
              )}
              {evidence && !hasFrames ? (
                <p className="text-2xs text-muted-foreground">
                  This event has no rendered frames — its manifest references frames that were
                  never rendered.
                </p>
              ) : null}
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

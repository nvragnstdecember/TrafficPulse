import { Check, Eraser, MapPin, Save, TrafficCone } from 'lucide-react';
import { useMemo, useState } from 'react';

import { type ScenePoint, type SignalPhaseSpec, type ViolationType } from '@/api/types';
import { useCalibrateScene, useValidateScene, useVideoScene } from '@/hooks/use-scenes';
import {
  CALIBRATION_TOOLS,
  type CalibrationShapes,
  type CalibrationTool,
  EMPTY_SHAPES,
  buildSceneDraft,
  canSubmit,
  centroid,
  derivedSceneNotice,
  isPolygonComplete,
  isSegmentComplete,
  perpendicular,
  previewUnlocked,
} from '@/lib/calibration';
import { violationLabel } from '@/lib/workspace';
import { cn } from '@/lib/utils';
import { notify } from '@/store/notifications-store';

import { ErrorBanner } from '../common/error-banner';
import { StatusChip } from '../common/status-chip';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { SignalScheduleEditor } from './signal-schedule-editor';

export interface SceneCalibratorProps {
  videoId: string;
  frameWidth: number;
  frameHeight: number;
  /** The video source, shown as a still backdrop to draw against. */
  posterSrc: string | null;
  schedule: SignalPhaseSpec[];
  onScheduleChange: (schedule: SignalPhaseSpec[]) => void;
  /** Re-run processing with the rules the calibrated scene unlocked. */
  onReprocess?: (supported: ViolationType[]) => void;
}

const POLYGON_TOOLS: Record<CalibrationTool, keyof CalibrationShapes | null> = {
  lane: 'lane',
  'no-stopping': 'noStopping',
  junction: 'junction',
  'signal-roi': 'signalRoi',
  direction: null,
  'stop-line': null,
};

const SEGMENT_TOOLS: Partial<Record<CalibrationTool, 'direction' | 'stopLine'>> = {
  direction: 'direction',
  'stop-line': 'stopLine',
};

function toPath(points: ScenePoint[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

/**
 * The scene calibration surface (H12's drawing tools, extended for H13's junction).
 *
 * An SVG overlaid on a still of the video, with `viewBox` set to the video's own
 * pixel dimensions — so a pointer position *is* a scene coordinate and there is no
 * scaling factor to get wrong. Everything the analyst draws is in the same space the
 * backend validates against.
 *
 * The component owns only the drawing interaction. What a shape *means*, how it
 * becomes a `SceneDraft`, and which violations it unlocks all live in
 * `lib/calibration`, and whether a draft is actually valid is answered by the
 * server — this never second-guesses the contract.
 */
export function SceneCalibrator({
  videoId,
  frameWidth,
  frameHeight,
  posterSrc,
  schedule,
  onScheduleChange,
  onReprocess,
}: SceneCalibratorProps) {
  const [tool, setTool] = useState<CalibrationTool>('lane');
  const [shapes, setShapes] = useState<CalibrationShapes>(EMPTY_SHAPES);

  const sceneQuery = useVideoScene(videoId);
  const calibrate = useCalibrateScene();
  const validate = useValidateScene();

  const saved = sceneQuery.data ?? null;
  // A scene nobody drew has to say so, and has to say *which* honest outcome it
  // was — estimating a direction and abstaining from one are different claims.
  const derivedNotice = useMemo(() => derivedSceneNotice(saved), [saved]);
  const preview = useMemo(() => previewUnlocked(shapes), [shapes]);
  const validationErrors = validate.data?.valid === false ? validate.data.errors : [];

  const draft = useMemo(
    () =>
      buildSceneDraft({
        shapes,
        frameWidth,
        frameHeight,
        cameraId: `cam-${videoId}`,
        sceneName: `Scene for ${videoId}`,
      }),
    [shapes, frameWidth, frameHeight, videoId],
  );

  function addPoint(point: ScenePoint): void {
    const polygonKey = POLYGON_TOOLS[tool];
    if (polygonKey) {
      setShapes((current) => ({
        ...current,
        [polygonKey]: [...(current[polygonKey] as ScenePoint[]), point],
      }));
      return;
    }
    const segmentKey = SEGMENT_TOOLS[tool];
    if (!segmentKey) return;
    setShapes((current) => {
      const existing = current[segmentKey];
      // First click sets the start; the second completes it; a third starts over.
      if (!existing || isSegmentComplete(existing)) {
        return { ...current, [segmentKey]: { from: point, to: point } };
      }
      return { ...current, [segmentKey]: { from: existing.from, to: point } };
    });
  }

  function handleClick(event: React.MouseEvent<SVGSVGElement>): void {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = ((event.clientX - rect.left) / rect.width) * frameWidth;
    const y = ((event.clientY - rect.top) / rect.height) * frameHeight;
    addPoint([Number(x.toFixed(1)), Number(y.toFixed(1))]);
  }

  function clearCurrentTool(): void {
    const polygonKey = POLYGON_TOOLS[tool];
    if (polygonKey) {
      setShapes((current) => ({ ...current, [polygonKey]: [] }));
      return;
    }
    const segmentKey = SEGMENT_TOOLS[tool];
    if (segmentKey) setShapes((current) => ({ ...current, [segmentKey]: null }));
  }

  function handleValidate(): void {
    validate.mutate({ videoId, draft });
  }

  function handleSave(): void {
    calibrate.mutate(
      { videoId, draft },
      {
        onSuccess: (summary) => {
          notify({
            title: 'Scene saved.',
            description: `Unlocked: ${summary.supported_violations
              .map((v) => violationLabel(v))
              .join(', ')}`,
          });
        },
      },
    );
  }

  const activeTool = CALIBRATION_TOOLS.find((spec) => spec.id === tool);
  const stopLineNormal =
    isSegmentComplete(shapes.stopLine) && shapes.stopLine
      ? perpendicular(shapes.stopLine, centroid(shapes.junction))
      : null;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 p-4 pb-2">
        <CardTitle className="flex items-center gap-2">
          <TrafficCone className="size-4 text-muted-foreground" aria-hidden="true" />
          Scene calibration
        </CardTitle>
        {saved ? (
          <StatusChip tone="success" label={`Calibrated · ${saved.zone_count} zone(s)`} />
        ) : (
          <StatusChip tone="warning" label="Not calibrated" />
        )}
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-2">
        <p className="text-xs text-muted-foreground">
          Draw this camera&apos;s geometry over the video. Wrong-way needs a lane and a
          direction; illegal stopping needs a no-stopping zone; red-light needs a stop line
          and the junction beyond it.
        </p>

        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Calibration tools">
          {CALIBRATION_TOOLS.map((spec) => (
            <Button
              key={spec.id}
              size="sm"
              variant={spec.id === tool ? 'default' : 'outline'}
              onClick={() => setTool(spec.id)}
              aria-pressed={spec.id === tool}
            >
              {spec.label}
            </Button>
          ))}
        </div>
        {activeTool ? (
          <p className="text-2xs text-muted-foreground">{activeTool.hint}</p>
        ) : null}

        <div className="relative overflow-hidden rounded-lg border bg-black">
          {posterSrc ? (
            <video
              src={posterSrc}
              preload="metadata"
              muted
              playsInline
              className="h-full w-full"
              aria-label="Calibration backdrop"
            />
          ) : (
            <div
              className="flex aspect-video items-center justify-center text-xs text-muted-foreground"
              data-testid="calibration-no-backdrop"
            >
              No preview available — geometry can still be drawn.
            </div>
          )}
          <svg
            viewBox={`0 0 ${frameWidth} ${frameHeight}`}
            className="absolute inset-0 h-full w-full cursor-crosshair"
            onClick={handleClick}
            role="application"
            aria-label="Scene drawing surface"
            data-testid="calibration-surface"
          >
            <Polygon points={shapes.lane} className="stroke-sky-400" />
            <Polygon points={shapes.noStopping} className="stroke-amber-400" />
            <Polygon points={shapes.junction} className="stroke-fuchsia-400" />
            <Polygon points={shapes.signalRoi} className="stroke-emerald-400" />
            <Segment segment={shapes.direction} className="stroke-sky-300" arrow />
            <Segment segment={shapes.stopLine} className="stroke-red-400" />
            {/* The derived entry direction, drawn so the analyst can see which way
                crossing the line counts — the single easiest thing to get backwards. */}
            <Segment segment={stopLineNormal} className="stroke-red-300" arrow dashed />
          </svg>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="outline" onClick={clearCurrentTool}>
            <Eraser className="size-4" />
            Clear {activeTool?.label.toLowerCase()}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleValidate}
            disabled={!canSubmit(shapes) || validate.isPending}
          >
            <Check className="size-4" />
            Check
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!canSubmit(shapes) || calibrate.isPending}
          >
            <Save className="size-4" />
            {calibrate.isPending ? 'Saving…' : 'Save scene'}
          </Button>
        </div>

        {derivedNotice ? (
          <div
            role="note"
            className="rounded-md border border-warning/40 p-3 text-2xs text-muted-foreground"
          >
            <p className="text-xs font-medium text-foreground">{derivedNotice.title}</p>
            <p className="mt-0.5">{derivedNotice.body}</p>
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-1.5" aria-label="Violations unlocked">
          <span className="text-2xs uppercase tracking-wide text-muted-foreground">
            Unlocks
          </span>
          {(saved?.supported_violations ?? preview).map((violation) => (
            <Badge key={violation} variant="muted">
              {violationLabel(violation)}
            </Badge>
          ))}
        </div>

        {validationErrors.length > 0 ? (
          <div role="alert" className="space-y-1 rounded-md border border-warning/40 p-3">
            <p className="text-xs font-medium">This drawing is not usable yet</p>
            <ul className="space-y-0.5 text-2xs text-muted-foreground">
              {validationErrors.map((message) => (
                <li key={message}>{message}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {validate.data?.valid ? (
          <p className="flex items-center gap-1.5 text-xs text-success">
            <MapPin className="size-3.5" aria-hidden="true" />
            Valid drawing — saving will bind it to this video.
          </p>
        ) : null}
        {calibrate.isError ? (
          <ErrorBanner title="Could not save the scene" error={calibrate.error} />
        ) : null}

        <SignalScheduleEditor
          schedule={schedule}
          onChange={onScheduleChange}
          enabled={(saved?.supported_violations ?? preview).includes('red_light_jumping')}
        />

        {saved && onReprocess ? (
          <Button
            size="sm"
            variant="outline"
            onClick={() => onReprocess(saved.supported_violations)}
          >
            Re-run analysis with this scene
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

function Polygon({ points, className }: { points: ScenePoint[]; className: string }) {
  if (points.length === 0) return null;
  const shared = cn('fill-none stroke-2', className);
  return (
    <>
      {isPolygonComplete(points) ? (
        <polygon points={toPath(points)} className={cn(shared, 'fill-current/10')} />
      ) : (
        <polyline points={toPath(points)} className={shared} />
      )}
      {points.map(([x, y], index) => (
        <circle key={`${x}-${y}-${index}`} cx={x} cy={y} r={4} className={className} />
      ))}
    </>
  );
}

function Segment({
  segment,
  className,
  arrow = false,
  dashed = false,
}: {
  segment: { from: ScenePoint; to: ScenePoint } | null;
  className: string;
  arrow?: boolean;
  dashed?: boolean;
}) {
  if (!segment) return null;
  return (
    <>
      <line
        x1={segment.from[0]}
        y1={segment.from[1]}
        x2={segment.to[0]}
        y2={segment.to[1]}
        strokeDasharray={dashed ? '6 4' : undefined}
        className={cn('stroke-2', className)}
      />
      {arrow ? <circle cx={segment.to[0]} cy={segment.to[1]} r={5} className={className} /> : null}
    </>
  );
}

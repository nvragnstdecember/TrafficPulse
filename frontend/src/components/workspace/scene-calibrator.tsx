import { Check, DownloadCloud, Eraser, MapPin, Save, TrafficCone } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { type ScenePoint, type SignalPhaseSpec, type ViolationType } from '@/api/types';
import {
  useCalibrateScene,
  useSceneRevision,
  useValidateScene,
  useVideoScene,
} from '@/hooks/use-scenes';
import {
  CALIBRATION_TOOLS,
  type CalibrationShapes,
  type CalibrationTool,
  EMPTY_SHAPES,
  EMPTY_TUNING,
  TUNING_DEFAULTS,
  type TuningInput,
  buildSceneDraft,
  canSubmit,
  centroid,
  derivedSceneNotice,
  isPolygonComplete,
  isSegmentComplete,
  perpendicular,
  previewUnlocked,
  sceneToShapes,
  sceneToTuning,
  tuningErrors,
} from '@/lib/calibration';
import { violationLabel } from '@/lib/workspace';
import { cn } from '@/lib/utils';
import { notify } from '@/store/notifications-store';

import { CollapsibleSection } from '../common/collapsible-section';
import { ErrorBanner } from '../common/error-banner';
import { StatusChip } from '../common/status-chip';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { SignalScheduleEditor } from './signal-schedule-editor';

export interface SceneCalibratorProps {
  videoId: string;
  frameWidth: number;
  frameHeight: number;
  /** The video source, shown as a still backdrop to draw against. */
  posterSrc: string | null;
  schedule: SignalPhaseSpec[];
  onScheduleChange: (schedule: SignalPhaseSpec[]) => void;
  /** Clip duration in seconds, for the frame picker and the signal timeline. */
  durationSeconds?: number;
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
 *
 * What the wording is careful about
 * ----------------------------------
 * Everything on this panel is **declared**, not inferred. Which way traffic legally
 * travels, where stopping is prohibited, what the signal was showing, how long a
 * vehicle may dwell — none of it is visible in pixels, and the copy says so rather
 * than letting a viewer assume the system worked it out. That honesty is the point of
 * the surface, not a disclaimer bolted onto it.
 */
export function SceneCalibrator({
  videoId,
  frameWidth,
  frameHeight,
  posterSrc,
  schedule,
  onScheduleChange,
  durationSeconds,
  onReprocess,
}: SceneCalibratorProps) {
  const [tool, setTool] = useState<CalibrationTool>('lane');
  const [shapes, setShapes] = useState<CalibrationShapes>(EMPTY_SHAPES);
  const [tuning, setTuning] = useState<TuningInput>(EMPTY_TUNING);
  const [notes, setNotes] = useState('');
  const [frameSeconds, setFrameSeconds] = useState(0);

  const sceneQuery = useVideoScene(videoId);
  const calibrate = useCalibrateScene();
  const validate = useValidateScene();

  const saved = sceneQuery.data ?? null;
  const revision = useSceneRevision(saved?.scene_hash);

  // A scene nobody drew has to say so, and has to say *which* honest outcome it
  // was — estimating a direction and abstaining from one are different claims.
  const derivedNotice = useMemo(() => derivedSceneNotice(saved), [saved]);
  const preview = useMemo(() => previewUnlocked(shapes), [shapes]);
  const validationErrors = validate.data?.valid === false ? validate.data.errors : [];
  const thresholdErrors = useMemo(() => tuningErrors(tuning), [tuning]);

  const draft = useMemo(
    () =>
      buildSceneDraft({
        shapes,
        frameWidth,
        frameHeight,
        cameraId: `cam-${videoId}`,
        sceneName: `Scene for ${videoId}`,
        tuning,
        notes,
      }),
    [shapes, frameWidth, frameHeight, videoId, tuning, notes],
  );

  // The backdrop is a real <video>; seeking it is how a representative frame is
  // chosen. Drawing against frame 0 is the one thing that reliably puts geometry in
  // the wrong place, because frame 0 of a traffic clip is often empty road.
  const backdropRef = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const element = backdropRef.current;
    if (!element) return;
    try {
      element.currentTime = frameSeconds;
    } catch {
      // Seeking before metadata loads throws in some browsers; the next change wins.
    }
  }, [frameSeconds]);

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

  /**
   * Pull the bound revision's geometry back onto the drawing surface.
   *
   * Explicit rather than automatic: an analyst mid-drawing must not have their work
   * replaced because a query resolved. Loading is a decision, and the button says
   * what it will do.
   */
  function handleLoadSaved(): void {
    const scene = revision.data;
    if (!scene) return;
    setShapes(sceneToShapes(scene));
    setTuning(sceneToTuning(scene));
    setNotes(scene.scene.description ?? '');
    notify({
      title: 'Saved calibration loaded.',
      description: 'Edit and save again to store a new revision; the old one is kept.',
    });
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
  const unlocked = saved?.supported_violations ?? preview;
  const seekable = Boolean(posterSrc) && Boolean(durationSeconds && durationSeconds > 0);

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
          Declare this camera&apos;s context by drawing it over the video. Wrong-way needs a
          lane and a direction; illegal stopping needs a no-stopping zone; red-light needs a
          stop line and the junction beyond it.
        </p>
        <p
          className="rounded-md border border-dashed p-2.5 text-2xs text-muted-foreground"
          data-testid="calibration-declared-notice"
        >
          <span className="font-medium text-foreground">Everything here is declared.</span>{' '}
          Which way traffic legally travels, where stopping is prohibited, and what the
          signal was showing are facts about the site and its rules — not things the camera
          can establish from pixels. TrafficPulse reasons over what you declare; it does not
          infer it, and it does not verify it.
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
              ref={backdropRef}
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

        {seekable ? (
          <div className="space-y-1">
            <Label htmlFor="calibration-frame" className="text-2xs uppercase tracking-wide">
              Representative frame · {frameSeconds.toFixed(1)}s
            </Label>
            <input
              id="calibration-frame"
              type="range"
              min={0}
              max={durationSeconds}
              step={0.1}
              value={frameSeconds}
              onChange={(event) => setFrameSeconds(Number(event.target.value))}
              className="w-full accent-primary"
            />
            <p className="text-2xs text-muted-foreground">
              Scrub to a frame that shows the traffic you are describing. The geometry is in
              the video&apos;s pixel space, so it applies to the whole clip regardless of
              which frame you drew it on.
            </p>
          </div>
        ) : null}

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
            disabled={
              !canSubmit(shapes) || calibrate.isPending || thresholdErrors.length > 0
            }
          >
            <Save className="size-4" />
            {calibrate.isPending ? 'Saving…' : 'Save scene'}
          </Button>
          {revision.data ? (
            <Button size="sm" variant="outline" onClick={handleLoadSaved}>
              <DownloadCloud className="size-4" />
              Load saved calibration
            </Button>
          ) : null}
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
          {unlocked.map((violation) => (
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

        <ThresholdFields
          tuning={tuning}
          onChange={setTuning}
          errors={thresholdErrors}
          showDwell={unlocked.includes('illegal_stopping')}
        />

        <SignalScheduleEditor
          schedule={schedule}
          onChange={onScheduleChange}
          enabled={unlocked.includes('red_light_jumping')}
          durationSeconds={durationSeconds}
        />

        <div className="space-y-1.5">
          <Label htmlFor="scene-notes" className="text-2xs uppercase tracking-wide">
            Scene notes
          </Label>
          <textarea
            id="scene-notes"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            rows={2}
            maxLength={2000}
            placeholder="Why this geometry was drawn as it was. Stored inside the scene, so anyone resolving an event’s scene hash later reads it too."
            className="w-full rounded-md border bg-background px-2 py-1.5 text-xs
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>

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

/**
 * The site thresholds an analyst may set.
 *
 * Blank means "use the provisional default", and the default is shown as a
 * placeholder rather than pre-filled: a number sitting in a field reads as a choice,
 * and the scene records operator choices as such. The dwell threshold is surfaced
 * directly because it describes the *site*; the rest are collapsed because changing
 * them changes how a shipped rule behaves and should be a deliberate act.
 */
function ThresholdFields({
  tuning,
  onChange,
  errors,
  showDwell,
}: {
  tuning: TuningInput;
  onChange: (tuning: TuningInput) => void;
  errors: string[];
  showDwell: boolean;
}) {
  function set(key: keyof TuningInput, raw: string): void {
    const next = { ...tuning };
    if (raw.trim() === '') delete next[key];
    else next[key] = Number(raw);
    onChange(next);
  }

  return (
    <section className="space-y-2 rounded-md border p-3" aria-label="Rule thresholds">
      <h3 className="text-2xs uppercase tracking-wide text-muted-foreground">
        Thresholds (this site)
      </h3>
      {showDwell ? (
        <div className="flex flex-wrap items-center gap-2">
          <Label htmlFor="dwell-threshold" className="text-xs">
            Stopping dwell
          </Label>
          <Input
            id="dwell-threshold"
            type="number"
            min={0}
            step={0.5}
            className="h-8 w-24"
            placeholder={String(TUNING_DEFAULTS.stationaryDurationSeconds)}
            value={tuning.stationaryDurationSeconds ?? ''}
            onChange={(event) => set('stationaryDurationSeconds', event.target.value)}
          />
          <span className="text-2xs text-muted-foreground">
            seconds a vehicle may stand inside the no-stopping zone (default{' '}
            {TUNING_DEFAULTS.stationaryDurationSeconds}s)
          </span>
        </div>
      ) : (
        <p className="text-2xs text-muted-foreground">
          Draw a no-stopping zone to set a dwell threshold.
        </p>
      )}

      <CollapsibleSection title="Advanced thresholds" defaultOpen={false}>
        <div className="space-y-2 pt-1">
          <NumberField
            id="heading-deviation"
            label="Heading deviation"
            suffix={`degrees off the legal direction that count as opposing (default ${TUNING_DEFAULTS.headingDeviationMaxDegrees}°)`}
            value={tuning.headingDeviationMaxDegrees}
            placeholder={TUNING_DEFAULTS.headingDeviationMaxDegrees}
            step={5}
            onChange={(raw) => set('headingDeviationMaxDegrees', raw)}
          />
          <NumberField
            id="wrong-way-persistence"
            label="Wrong-way persistence"
            suffix={`seconds of sustained opposition before confirming (default ${TUNING_DEFAULTS.wrongWayMinPersistenceSeconds}s)`}
            value={tuning.wrongWayMinPersistenceSeconds}
            placeholder={TUNING_DEFAULTS.wrongWayMinPersistenceSeconds}
            step={0.5}
            onChange={(raw) => set('wrongWayMinPersistenceSeconds', raw)}
          />
          <NumberField
            id="red-light-debounce"
            label="Red-light debounce"
            suffix={`seconds after the stop-line crossing (default ${TUNING_DEFAULTS.redLightMinPersistenceSeconds}s — a debounce, not a grace period)`}
            value={tuning.redLightMinPersistenceSeconds}
            placeholder={TUNING_DEFAULTS.redLightMinPersistenceSeconds}
            step={0.1}
            onChange={(raw) => set('redLightMinPersistenceSeconds', raw)}
          />
          <p className="text-2xs text-muted-foreground">
            These are stored on the scene as <em>provisional</em>: operator-chosen values,
            not tuned against ground truth. Leave a field blank to keep the default.
          </p>
        </div>
      </CollapsibleSection>

      {errors.length > 0 ? (
        <ul role="alert" className="space-y-0.5 text-2xs text-destructive">
          {errors.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function NumberField({
  id,
  label,
  suffix,
  value,
  placeholder,
  step,
  onChange,
}: {
  id: string;
  label: string;
  suffix: string;
  value: number | undefined;
  placeholder: number;
  step: number;
  onChange: (raw: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        min={0}
        step={step}
        className="h-8 w-24"
        placeholder={String(placeholder)}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value)}
      />
      <span className="text-2xs text-muted-foreground">{suffix}</span>
    </div>
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

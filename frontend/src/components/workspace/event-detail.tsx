import { Crosshair, Download, FileJson, Image, MousePointerClick, Play } from 'lucide-react';

import {
  type ConfirmedEvent,
  type EvidenceManifest,
  type MeasuredValue,
  type ReviewStatus,
} from '@/api/types';
import { reviewStatusLabel, reviewStatusTone } from '@/lib/review';
import { formatDateTime, formatPercent } from '@/lib/format';
import {
  CONFIDENCE_LABELS,
  type WorkspaceEvent,
  buildEventNarrative,
  formatClock,
  severityLabel,
  severityTone,
  violationDescription,
  violationLabel,
  violationSeverity,
  violationTone,
} from '@/lib/workspace';

import { CollapsibleSection } from '../common/collapsible-section';
import { CopyButton } from '../common/copy-button';
import { EmptyState } from '../common/empty-state';
import { ErrorBanner } from '../common/error-banner';
import { ProgressBar } from '../common/progress-bar';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { EventNarrative } from './event-narrative';

export interface EventDetailProps {
  event: WorkspaceEvent | null;
  detail: ConfirmedEvent | undefined;
  evidence: EvidenceManifest | undefined;
  /** Detail request in flight. */
  isLoading: boolean;
  /** Detail request failed (retryable). */
  detailError?: unknown;
  /** Retry loading the detail. */
  onRetryDetail?: () => void;
  /** Evidence request in flight (may lag the event — delayed generation). */
  isEvidenceLoading?: boolean;
  /** Evidence request failed / not yet available (retryable). */
  evidenceError?: unknown;
  /** Retry loading the evidence manifest. */
  onRetryEvidence?: () => void;
  /** Seek the player to a media-time position (seconds). */
  onSeek: (seconds: number) => void;
  /** Playhead position, so the narrative shows how far the story has played. */
  currentTime?: number;
  /** Replay this event's review window from its lead-in. */
  onReplay?: () => void;
  /** The analyst decision surface, rendered as the Review tab (H9). */
  reviewSlot?: React.ReactNode;
  /** Current review status, badged in the header. */
  reviewStatus?: ReviewStatus;
  // --- H7E quick actions ---
  /** Open the full evidence viewer for this event. */
  onOpenEvidenceViewer?: () => void;
  /** Export this event as JSON (the backend's confirmed-event contract). */
  onExportJson?: () => void;
  /** Export this event's evidence manifest as JSON. */
  onExportManifest?: () => void;
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <dt className="text-2xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="truncate text-sm">{value}</dd>
    </div>
  );
}

function MeasurementTable({ title, rows }: { title: string; rows: MeasuredValue[] }) {
  return (
    <div className="space-y-1">
      <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">{title}</h4>
      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">None recorded.</p>
      ) : (
        <dl className="divide-y rounded-md border">
          {rows.map((row) => (
            <div key={row.name} className="flex items-center justify-between gap-3 px-3 py-1.5">
              <dt className="truncate text-sm">{row.name}</dt>
              <dd className="shrink-0 font-mono text-sm tabular-nums">
                {row.value}
                {row.unit ? ` ${row.unit}` : ''}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/**
 * The selected event's detail panel (H7C; review tooling in H7E).
 *
 * Preserves the H7C tabbed layout (Overview / Measurements / Evidence) and adds
 * the analyst-review surface: a severity badge, quick actions (jump, open the
 * evidence viewer, copy ids, export JSON / manifest), a confidence meter, a
 * plain-language rule explanation, collapsible technical metadata, and local
 * analyst notes. The backend renders no media, so evidence artifacts remain the
 * typed references they are.
 */
export function EventDetail({
  event,
  detail,
  evidence,
  isLoading,
  detailError,
  onRetryDetail,
  isEvidenceLoading,
  evidenceError,
  onRetryEvidence,
  onSeek,
  currentTime,
  onReplay,
  reviewSlot,
  reviewStatus,
  onOpenEvidenceViewer,
  onExportJson,
  onExportManifest,
}: EventDetailProps) {
  if (!event) {
    return (
      <EmptyState
        icon={MousePointerClick}
        title="No event selected"
        description="Pick an event from the list or a marker on the timeline to inspect it."
      />
    );
  }

  const severity = violationSeverity(event.violationType);

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3 p-4 pb-2">
        <div className="min-w-0 space-y-1">
          <CardTitle className="truncate">{violationLabel(event.violationType)}</CardTitle>
          <div className="flex items-center gap-1">
            <p className="truncate font-mono text-xs text-muted-foreground">{event.id}</p>
            <CopyButton value={event.id} label="Copy event ID" className="size-6" />
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <StatusChip
            tone={violationTone(event.violationType)}
            label={formatClock(event.mediaSeconds)}
            dot={false}
          />
          <StatusChip tone={severityTone(severity)} label={`${severityLabel(severity)} severity`} />
          {reviewStatus ? (
            <StatusChip
              tone={reviewStatusTone(reviewStatus)}
              label={reviewStatusLabel(reviewStatus)}
            />
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="space-y-3 p-4 pt-2">
        {/* Quick actions */}
        <div className="flex flex-wrap items-center gap-1.5">
          {onReplay ? (
            <Button variant="outline" size="sm" onClick={onReplay}>
              <Play className="size-4" />
              Replay
            </Button>
          ) : null}
          <Button variant="outline" size="sm" onClick={() => onSeek(event.mediaSeconds)}>
            <Crosshair className="size-4" />
            Jump to {formatClock(event.mediaSeconds)}
          </Button>
          {onOpenEvidenceViewer ? (
            <Button variant="outline" size="sm" onClick={onOpenEvidenceViewer}>
              <Image className="size-4" />
              Evidence viewer
            </Button>
          ) : null}
          {onExportJson ? (
            <Button variant="ghost" size="sm" onClick={onExportJson}>
              <FileJson className="size-4" />
              JSON
            </Button>
          ) : null}
          {onExportManifest ? (
            <Button variant="ghost" size="sm" onClick={onExportManifest} disabled={!evidence}>
              <Download className="size-4" />
              Manifest
            </Button>
          ) : null}
        </div>

        <Tabs defaultValue="overview">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="measurements">Measurements</TabsTrigger>
            <TabsTrigger value="evidence">Evidence</TabsTrigger>
            {reviewSlot ? <TabsTrigger value="review">Review</TabsTrigger> : null}
          </TabsList>

          <TabsContent value="overview" className="space-y-3">
            <p className="text-sm text-muted-foreground">
              {violationDescription(event.violationType)}
            </p>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-2xs uppercase tracking-wide text-muted-foreground">
                  {event.confidenceSource
                    ? CONFIDENCE_LABELS[event.confidenceSource]
                    : 'Confidence'}
                </span>
                <span className="tabular-nums">
                  {event.confidence === null ? 'Not measured' : formatPercent(event.confidence)}
                </span>
              </div>
              <ProgressBar
                value={event.confidence}
                label="Event confidence"
                tone={severity === 'high' ? 'destructive' : 'primary'}
              />
              {event.confidenceComponents.length > 1 ? (
                <ul className="flex flex-wrap gap-x-3 gap-y-0.5 pt-0.5 text-2xs text-muted-foreground">
                  {event.confidenceComponents.map((component) => (
                    <li key={component.key} className="tabular-nums">
                      {component.label} {formatPercent(component.value)}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <dl className="grid grid-cols-2 gap-3">
              <Field label="Camera" value={event.cameraId} />
              <Field label="Rule" value={event.ruleId} />
              <Field label="Lane" value={event.lane ?? '—'} />
              <Field label="Tracks" value={event.trackIds.join(', ') || '—'} />
              <Field label="Triggered" value={formatDateTime(event.triggerAt)} />
              <Field
                label="Observed for"
                value={`${event.observationSeconds.toFixed(2)}s (from ${formatClock(
                  event.startSeconds,
                )})`}
              />
            </dl>

            {detail ? (
              <CollapsibleSection title="Technical metadata" defaultOpen={false}>
                <dl className="grid grid-cols-2 gap-3">
                  <Field label="Rule version" value={detail.rule_version ?? '—'} />
                  <Field label="Code version" value={detail.code_version ?? '—'} />
                  <Field label="Scene config" value={detail.scene_config_hash ?? '—'} />
                  <Field
                    label="Models"
                    value={detail.models.map((m) => `${m.name}@${m.version}`).join(', ') || '—'}
                  />
                </dl>
              </CollapsibleSection>
            ) : null}

          </TabsContent>

          <TabsContent value="timeline" className="space-y-3">
            <p className="text-sm text-muted-foreground">
              How this violation evolved, reconstructed from the reasoner&rsquo;s own
              window. Pick a step to jump the player there.
            </p>
            <EventNarrative
              steps={buildEventNarrative(event, {
                thresholdSeconds:
                  detail?.thresholds.find((threshold) => /persistence/i.test(threshold.name))
                    ?.value ?? null,
                hasEvidence: Boolean(evidence),
              })}
              currentTime={currentTime}
              onSeek={onSeek}
            />
          </TabsContent>

          <TabsContent value="measurements" className="space-y-3">
            {isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : detailError ? (
              <ErrorBanner
                title="Could not load measurements"
                error={detailError}
                onRetry={onRetryDetail}
              />
            ) : detail ? (
              <>
                <MeasurementTable title="Measurements" rows={detail.measurements} />
                <MeasurementTable title="Thresholds" rows={detail.thresholds} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Detail unavailable.</p>
            )}
          </TabsContent>

          <TabsContent value="evidence" className="space-y-3">
            {evidence ? (
              <>
                <div className="flex items-center gap-1.5">
                  {onOpenEvidenceViewer ? (
                    <Button variant="outline" size="sm" onClick={onOpenEvidenceViewer}>
                      <Image className="size-4" />
                      Open viewer
                    </Button>
                  ) : null}
                  <CopyButton value={evidence.evidence_package_id} label="Copy evidence ID">
                    <span className="text-xs">Copy ID</span>
                  </CopyButton>
                </div>

                <div className="space-y-1">
                  <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">
                    Artifacts
                  </h4>
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
                      <li
                        key={label}
                        className="flex items-center justify-between gap-3 px-3 py-1.5"
                      >
                        <span className="shrink-0 text-sm">{label}</span>
                        <span className="truncate font-mono text-xs text-muted-foreground">
                          {artifact ? artifact.locator : '—'}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-1">
                  <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">
                    Rule trace
                  </h4>
                  {evidence.rule_trace.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No trace recorded.</p>
                  ) : (
                    <ol className="space-y-1">
                      {evidence.rule_trace.map((step) => (
                        <li key={step.index} className="rounded-md border px-3 py-1.5 text-sm">
                          <span className="font-medium">{step.label}</span>
                          {step.note ? (
                            <span className="text-muted-foreground"> — {step.note}</span>
                          ) : null}
                        </li>
                      ))}
                    </ol>
                  )}
                </div>

                <div className="space-y-1">
                  <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">Models</h4>
                  <ul className="text-sm text-muted-foreground">
                    {evidence.models.map((model) => (
                      <li key={`${model.name}@${model.version}`} className="font-mono text-xs">
                        {model.name}@{model.version}
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            ) : (isEvidenceLoading ?? isLoading) ? (
              <Skeleton className="h-24 w-full" />
            ) : evidenceError ? (
              <ErrorBanner
                title="Evidence unavailable"
                error={evidenceError}
                onRetry={onRetryEvidence}
              />
            ) : (
              <p className="text-sm text-muted-foreground">No evidence manifest for this event.</p>
            )}
          </TabsContent>
          {reviewSlot ? (
            <TabsContent value="review" className="space-y-3">
              {reviewSlot}
            </TabsContent>
          ) : null}
        </Tabs>
      </CardContent>
    </Card>
  );
}

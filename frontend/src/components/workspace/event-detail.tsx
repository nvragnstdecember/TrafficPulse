import { Crosshair, MousePointerClick } from 'lucide-react';

import { type ConfirmedEvent, type EvidenceManifest, type MeasuredValue } from '@/api/types';
import { formatDateTime, formatPercent } from '@/lib/format';
import { type WorkspaceEvent, formatClock, violationLabel, violationTone } from '@/lib/workspace';

import { EmptyState } from '../common/empty-state';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';

export interface EventDetailProps {
  event: WorkspaceEvent | null;
  detail: ConfirmedEvent | undefined;
  evidence: EvidenceManifest | undefined;
  isLoading: boolean;
  /** Seek the player to a media-time position (seconds). */
  onSeek: (seconds: number) => void;
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
 * The selected event's detail panel (H7C): identity + timing, the rule's
 * measurements against its thresholds, and the evidence manifest (artifact
 * references, rule trace, and model provenance).
 *
 * The backend renders no media, so evidence artifacts are shown as the typed
 * references they are — locator, media type, and content hash.
 */
export function EventDetail({ event, detail, evidence, isLoading, onSeek }: EventDetailProps) {
  if (!event) {
    return (
      <EmptyState
        icon={MousePointerClick}
        title="No event selected"
        description="Pick an event from the list or a marker on the timeline to inspect it."
      />
    );
  }

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3 p-4 pb-2">
        <div className="min-w-0 space-y-1">
          <CardTitle className="truncate">{violationLabel(event.violationType)}</CardTitle>
          <p className="truncate font-mono text-xs text-muted-foreground">{event.id}</p>
        </div>
        <StatusChip
          tone={violationTone(event.violationType)}
          label={formatClock(event.mediaSeconds)}
          dot={false}
        />
      </CardHeader>

      <CardContent className="space-y-3 p-4 pt-2">
        <Button variant="outline" size="sm" onClick={() => onSeek(event.mediaSeconds)}>
          <Crosshair className="size-4" />
          Jump to {formatClock(event.mediaSeconds)}
        </Button>

        <Tabs defaultValue="overview">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="measurements">Measurements</TabsTrigger>
            <TabsTrigger value="evidence">Evidence</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <dl className="grid grid-cols-2 gap-3">
              <Field label="Camera" value={event.cameraId} />
              <Field label="Rule" value={event.ruleId} />
              <Field label="Confidence" value={formatPercent(event.confidence)} />
              <Field label="Lane" value={event.lane ?? '—'} />
              <Field label="Tracks" value={event.trackIds.join(', ') || '—'} />
              <Field label="Triggered" value={formatDateTime(event.triggerAt)} />
              {detail ? (
                <>
                  <Field label="Rule version" value={detail.rule_version ?? '—'} />
                  <Field label="Code version" value={detail.code_version ?? '—'} />
                </>
              ) : null}
            </dl>
          </TabsContent>

          <TabsContent value="measurements" className="space-y-3">
            {isLoading ? (
              <Skeleton className="h-24 w-full" />
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
            ) : isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : (
              <p className="text-sm text-muted-foreground">No evidence manifest for this event.</p>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

import { BarChart3, CheckCircle2 } from 'lucide-react';

import { type EngineMetrics, type JobStatusResponse, type ModelRef } from '@/api/types';
import { processingSummary, reviewStats, violationBreakdown } from '@/lib/review-stats';
import { type WorkspaceEvent } from '@/lib/workspace';

import { CollapsibleSection } from '../common/collapsible-section';
import { StatusChip } from '../common/status-chip';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { StatGrid } from './stat-grid';

export interface ReviewSummaryProps {
  events: WorkspaceEvent[];
  metrics: EngineMetrics | null;
  job: JobStatusResponse | null | undefined;
  /** Wall-clock seconds the run took, when known. */
  elapsedSeconds: number | null;
  /** Run-level model provenance, from any confirmed event. */
  models: ModelRef[];
  /** True once the run has finished — switches the headline to the completion state. */
  complete: boolean;
}

/**
 * Run statistics and the post-processing summary (Phase 2, features 7 + 8).
 *
 * Two closely-related read-outs in one card rather than two, because they answer
 * one question — "what did this run actually do?" — and splitting them would make
 * an analyst read the same numbers twice. The completion header appears only once
 * the run is finished, so the same component covers the in-flight and settled
 * states without a second layout.
 *
 * Everything is read-only and derived from data already on screen: the engine's
 * own metrics snapshot and the confirmed events. Nothing is recomputed and no
 * value is invented — see `lib/review-stats`.
 */
export function ReviewSummary({
  events,
  metrics,
  job,
  elapsedSeconds,
  models,
  complete,
}: ReviewSummaryProps) {
  const breakdown = violationBreakdown(events);

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 p-4 pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          {complete ? (
            <CheckCircle2 className="size-4 text-success" aria-hidden="true" />
          ) : (
            <BarChart3 className="size-4 text-muted-foreground" aria-hidden="true" />
          )}
          {complete ? 'Processing complete' : 'Run statistics'}
        </CardTitle>
        {breakdown.length > 0 ? (
          <div className="flex flex-wrap justify-end gap-1">
            {breakdown.map((entry) => (
              <StatusChip
                key={entry.type}
                tone="neutral"
                label={`${entry.label} ${entry.count}`}
                dot={false}
                className="text-2xs"
              />
            ))}
          </div>
        ) : null}
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-2">
        <StatGrid stats={reviewStats(events, metrics)} />

        <CollapsibleSection title="Processing summary" defaultOpen={complete}>
          <div className="space-y-3">
            <StatGrid stats={processingSummary(job, metrics, elapsedSeconds)} />
            <div className="space-y-1">
              <h4 className="text-2xs uppercase tracking-wide text-muted-foreground">
                Model versions
              </h4>
              {models.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No model provenance recorded for this run.
                </p>
              ) : (
                <ul className="space-y-0.5">
                  {models.map((model) => (
                    <li
                      key={`${model.name}@${model.version}`}
                      className="truncate font-mono text-xs text-muted-foreground"
                    >
                      {model.name}@{model.version}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </CollapsibleSection>
      </CardContent>
    </Card>
  );
}

import { ScanFace } from 'lucide-react';

import { type HelmetAnalysis, type RiderAnalysis } from '@/api/types';
import {
  analysisStats,
  enforcementHint,
  enforcementLabel,
  enforcementTone,
  flipRate,
  helmetLabel,
  helmetTone,
  percent,
  sortRiders,
} from '@/lib/analysis';

import { CollapsibleSection } from '../common/collapsible-section';
import { StatusChip } from '../common/status-chip';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip';
import { StatGrid } from './stat-grid';

export interface HelmetAnalysisPanelProps {
  analysis: HelmetAnalysis | null | undefined;
}

/**
 * Helmet perception for a finished run — classification, never enforcement.
 *
 * Renders nothing when the run has no analysis, which is the normal state for a
 * deployment that did not configure one. That is a configuration fact, not a failure,
 * so it produces no empty state and no error.
 *
 * The panel is built around one separation, and every layout decision serves it: a
 * rider's **helmet reading** and whether that reading is **actionable** are two
 * columns, two vocabularies, and two chips. They are never merged, never colour-coded
 * as one thing, and a `no_helmet` reading is styled `warning` rather than
 * `destructive`, because red in this design system means a violation was confirmed and
 * nothing here confirms anything.
 */
export function HelmetAnalysisPanel({ analysis }: HelmetAnalysisPanelProps) {
  if (!analysis) return null;

  const riders = sortRiders(analysis.riders);
  const flips = flipRate(analysis.riders);

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 p-4 pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <ScanFace className="size-4 text-muted-foreground" aria-hidden="true" />
          Helmet analysis
        </CardTitle>
        <StatusChip
          tone="neutral"
          label={`enforcement ${analysis.enforcement}`}
          className="text-2xs"
        />
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-2">
        <p className="text-xs leading-relaxed text-muted-foreground">
          Per-rider helmet classification for this run. <strong>No violation is
          decided here</strong> — nothing on this panel is a confirmed event, and none
          of it appears in the event list. Labels are smoothed over a short per-rider
          window so they are readable; that smoothing is a display aid and has not been
          validated as an accuracy improvement.
        </p>

        <StatGrid stats={analysisStats(analysis)} />

        {analysis.multi_rider_riders > 0 ? (
          <p className="rounded border border-warning/40 bg-warning/5 px-3 py-2 text-xs leading-relaxed">
            <strong>{analysis.multi_rider_riders}</strong> of {analysis.riders_observed}{' '}
            rider(s) share a motorcycle with someone else. TrafficPulse does not attempt
            to tell driver from pillion — the tracker supplies no velocity, so which end
            of the bike is the front is genuinely unknown — and it will not attribute a
            helmet state to a rider whose role it cannot determine.
          </p>
        ) : null}

        {riders.length > 0 ? (
          <CollapsibleSection title={`Riders (${riders.length})`} defaultOpen>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Rider</TableHead>
                    <TableHead>Motorcycle</TableHead>
                    <TableHead>Reading</TableHead>
                    <TableHead className="text-right">Confidence</TableHead>
                    <TableHead className="text-right">Support</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {riders.map((rider) => (
                    <RiderRow key={rider.rider_track_id} rider={rider} />
                  ))}
                </TableBody>
              </Table>
            </div>
          </CollapsibleSection>
        ) : (
          <p className="text-sm text-muted-foreground">
            No riders were associated to a motorcycle in this clip, so the classifier was
            never run. For footage with no motorcycles, that is the correct outcome.
          </p>
        )}

        <CollapsibleSection title="Measurement notes">
          <ul className="space-y-1.5 text-xs leading-relaxed text-muted-foreground">
            {flips ? (
              <li>
                The raw per-frame label changed at least once on{' '}
                <strong>
                  {flips.flipped} of {flips.total}
                </strong>{' '}
                rider tracks in this run. Per-frame instability is a known, measured
                property of this classifier on real video; the displayed label is the
                smoothed one.
              </li>
            ) : null}
            <li>
              {analysis.gate_abstentions} crop(s) were rejected by the quality gate before
              any inference ran — too small, off-frame, or with no pixels. They cost no
              inference and are reported rather than dropped.
            </li>
            <li>
              Counts are of rider <em>tracks</em>, not frames or crops. A rider seen on
              thirty frames is one rider.
            </li>
            <li>
              Published accuracy figures for this classifier are per-crop and conditional
              on a rider reaching it at all. They are not end-to-end system accuracy, and
              they say nothing about multi-rider traffic.
            </li>
          </ul>
        </CollapsibleSection>
      </CardContent>
    </Card>
  );
}

function RiderRow({ rider }: { rider: RiderAnalysis }) {
  const confidence = percent(rider.confidence);
  return (
    <TableRow>
      <TableCell className="font-mono text-xs">{rider.rider_track_id}</TableCell>
      <TableCell className="font-mono text-xs text-muted-foreground">
        {rider.motorcycle_track_id ?? '—'}
        {rider.multi_rider ? (
          <span className="ml-1 text-2xs uppercase tracking-wide text-warning">
            ×{rider.rider_count}
          </span>
        ) : null}
      </TableCell>
      <TableCell>
        <StatusChip
          tone={helmetTone(rider.helmet_state)}
          label={helmetLabel(rider.helmet_state)}
          className="text-2xs"
        />
      </TableCell>
      <TableCell
        className={`text-right tabular-nums ${confidence ? '' : 'text-muted-foreground'}`}
      >
        {/* An em dash, never 0%: the classifier never scored this crop. */}
        {confidence ?? '—'}
      </TableCell>
      <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-help border-b border-dotted border-muted-foreground/40">
              {percent(rider.agreement)} / {rider.samples}f
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            {percent(rider.agreement)} of the smoothing window agreed with this label,
            over {rider.samples} classified frame(s). The raw label changed{' '}
            {rider.raw_label_flips} time(s) on this track.
          </TooltipContent>
        </Tooltip>
      </TableCell>
      <TableCell>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-help">
              <StatusChip
                tone={enforcementTone(rider.enforcement)}
                label={enforcementLabel(rider.enforcement)}
                className="text-2xs normal-case"
              />
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            {enforcementHint(rider.enforcement)}
          </TooltipContent>
        </Tooltip>
      </TableCell>
    </TableRow>
  );
}

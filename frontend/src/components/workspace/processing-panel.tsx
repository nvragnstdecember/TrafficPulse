import { Ban, RefreshCcw, Repeat2, Trash2 } from 'lucide-react';
import { useRef, useState } from 'react';

import { type ProcessingController } from '@/hooks/use-processing';
import { formatBytes, formatDuration, formatNumber, formatPercent } from '@/lib/format';
import { isActivePhase, isCancellablePhase, phaseLabel, phaseTone } from '@/lib/job';
import { acceptAttribute } from '@/lib/upload';
import { cn } from '@/lib/utils';

import { ConfirmDialog } from '../common/confirm-dialog';
import { ProgressBar } from '../common/progress-bar';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

export interface ProcessingPanelProps {
  controller: ProcessingController;
}

const LOG_TONE = {
  info: 'text-muted-foreground',
  success: 'text-success',
  error: 'text-destructive',
} as const;

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <p className="text-2xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="font-mono text-sm tabular-nums">{value}</p>
    </div>
  );
}

/**
 * The processing panel (H7C): live upload/job progress, timing, and the activity
 * log, plus the workflow actions (cancel, retry, replace, remove).
 *
 * Purely presentational — every action is delegated to the {@link ProcessingController}
 * from `useProcessing`, so no network or polling logic lives in the view.
 */
export function ProcessingPanel({ controller }: ProcessingPanelProps) {
  const {
    phase,
    job,
    video,
    progressRatio,
    elapsedSeconds,
    etaSeconds,
    logs,
    error,
    isCancelling,
    actions,
  } = controller;
  const replaceInputRef = useRef<HTMLInputElement | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);

  const active = isActivePhase(phase);
  const canCancel = phase === 'uploading' || isCancellablePhase(phase);
  const canRetry = phase === 'failed' || phase === 'cancelled';
  const tone =
    phase === 'failed'
      ? 'destructive'
      : phase === 'completed'
        ? 'success'
        : phase === 'cancelled'
          ? 'destructive'
          : 'primary';

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3 p-4 pb-2">
        <div className="min-w-0 space-y-1">
          <CardTitle className="truncate">{video?.filename ?? 'Processing'}</CardTitle>
          <p className="text-xs text-muted-foreground">
            {video
              ? `${formatBytes(video.size_bytes)}${
                  video.width && video.height ? ` · ${video.width}×${video.height}` : ''
                }${video.fps ? ` · ${video.fps.toFixed(0)} fps` : ''}`
              : 'Preparing upload…'}
          </p>
        </div>
        <StatusChip tone={phaseTone(phase)} label={phaseLabel(phase)} />
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-2">
        <div className="space-y-1.5">
          <ProgressBar
            value={active && progressRatio === null ? null : (progressRatio ?? 0)}
            label={`${phaseLabel(phase)} progress`}
            tone={tone}
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="tabular-nums">{formatPercent(progressRatio)}</span>
            <span className="tabular-nums">
              {formatDuration(elapsedSeconds)}
              {etaSeconds != null ? ` · ${formatDuration(etaSeconds)} left` : ''}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="Frames"
            value={
              job
                ? `${formatNumber(job.frames_processed)}${
                    job.frames_total ? ` / ${formatNumber(job.frames_total)}` : ''
                  }`
                : '—'
            }
          />
          <Stat label="Throughput" value={job?.fps ? `${job.fps.toFixed(1)} fps` : '—'} />
          <Stat label="Events" value={job ? formatNumber(job.event_count) : '—'} />
          <Stat label="Job" value={job?.job_id.slice(0, 8) ?? '—'} />
        </div>

        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center gap-2">
          {canCancel ? (
            <Button variant="outline" size="sm" onClick={actions.cancel} disabled={isCancelling}>
              <Ban className="size-4" />
              {isCancelling ? 'Cancelling…' : phase === 'uploading' ? 'Cancel upload' : 'Cancel'}
            </Button>
          ) : null}
          {canRetry ? (
            <Button variant="outline" size="sm" onClick={actions.retry}>
              <RefreshCcw className="size-4" />
              Retry
            </Button>
          ) : null}
          <Button
            variant="outline"
            size="sm"
            disabled={active}
            onClick={() => replaceInputRef.current?.click()}
          >
            <Repeat2 className="size-4" />
            Replace
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setConfirmRemove(true)}>
            <Trash2 className="size-4" />
            Remove
          </Button>
          <input
            ref={replaceInputRef}
            type="file"
            accept={acceptAttribute()}
            className="sr-only"
            aria-hidden="true"
            tabIndex={-1}
            data-testid="replace-input"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = '';
              if (file) actions.replace(file);
            }}
          />
        </div>

        <section aria-label="Activity log" className="space-y-1">
          <h3 className="text-2xs uppercase tracking-wide text-muted-foreground">Activity</h3>
          {logs.length === 0 ? (
            <p className="text-xs text-muted-foreground">No activity yet.</p>
          ) : (
            <ul className="max-h-32 space-y-0.5 overflow-y-auto text-xs">
              {logs.map((entry) => (
                <li key={entry.id} className={cn('flex gap-2', LOG_TONE[entry.level])}>
                  <span className="shrink-0 font-mono tabular-nums opacity-70">
                    {new Date(entry.at).toLocaleTimeString()}
                  </span>
                  <span className="min-w-0">{entry.message}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </CardContent>

      <ConfirmDialog
        open={confirmRemove}
        onOpenChange={setConfirmRemove}
        title="Remove this video?"
        description="The upload and its processing job are cleared from the workspace."
        confirmLabel="Remove"
        destructive
        onConfirm={actions.remove}
      />
    </Card>
  );
}

import { History, Loader2, Save } from 'lucide-react';
import { useEffect, useState } from 'react';

import { type ReviewAction, type ReviewCase, type ReviewEntry } from '@/api/types';
import { formatDateTime } from '@/lib/format';
import {
  availableDecisions,
  historyLabel,
  reviewActionLabel,
  reviewStatusLabel,
  reviewStatusTone,
} from '@/lib/review';
import { cn } from '@/lib/utils';

import { ErrorBanner } from '../common/error-banner';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Skeleton } from '../ui/skeleton';

export interface ReviewPanelProps {
  eventId: string;
  reviewCase: ReviewCase | undefined;
  history: ReviewEntry[];
  isLoading: boolean;
  error?: unknown;
  onRetry?: () => void;
  /** Record an action; the note box's contents ride along with it. */
  onDecide: (action: ReviewAction, note: string | null) => void;
  /** A decision is in flight. */
  isDeciding: boolean;
  /** Reviewer identity to display (the API records whoever the client says). */
  reviewer: string;
}

/** Which decisions get emphasis, so the primary path is obvious at a glance. */
const DECISION_VARIANTS: Partial<
  Record<ReviewAction, 'default' | 'outline' | 'destructive' | 'ghost'>
> = {
  open: 'default',
  approve: 'default',
  reject: 'outline',
  false_positive: 'outline',
  needs_more_evidence: 'ghost',
  reopen: 'outline',
};

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-2xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="truncate text-xs tabular-nums">{value}</dd>
    </div>
  );
}

/**
 * The analyst decision surface (H9): status, decisions, notes, metadata, history.
 *
 * Sits inside the existing event-detail panel as one more tab rather than a dialog
 * or a page of its own, so deciding stays part of reviewing: evidence, playback,
 * narrative, and decision are all reachable without the video ever unmounting.
 *
 * Only the decisions legal from the current state are rendered. That mirrors the
 * backend's transition table purely so the UI does not offer a click that will be
 * refused — the server still validates every request, and this list is never the
 * safeguard.
 *
 * Notes are sent with whichever action the analyst takes: typing a justification
 * and then clicking *Approve* records both in one journal entry, which is how a
 * reviewer actually works. The separate *Save note* button exists for the case
 * where somebody wants to annotate without deciding yet.
 */
export function ReviewPanel({
  eventId,
  reviewCase,
  history,
  isLoading,
  error,
  onRetry,
  onDecide,
  isDeciding,
  reviewer,
}: ReviewPanelProps) {
  const [note, setNote] = useState('');

  // Reset the draft when the analyst moves to another event, so a note is never
  // carried onto a case it was not written for.
  useEffect(() => {
    setNote('');
  }, [eventId]);

  if (error) {
    return <ErrorBanner title="Could not load the review" error={error} onRetry={onRetry} />;
  }
  if (isLoading || !reviewCase) {
    return <Skeleton className="h-40 w-full" />;
  }

  const status = reviewCase.status;
  const decisions = availableDecisions(status);
  const savedNote = reviewCase.note;

  function decide(action: ReviewAction): void {
    const trimmed = note.trim();
    onDecide(action, trimmed.length > 0 ? trimmed : null);
    setNote('');
  }

  return (
    <div className="space-y-4">
      {/* Status + reviewer */}
      <div className="flex flex-wrap items-center gap-2">
        <StatusChip tone={reviewStatusTone(status)} label={reviewStatusLabel(status)} />
        <span className="text-xs text-muted-foreground">
          Reviewing as <span className="font-medium text-foreground">{reviewer}</span>
        </span>
      </div>

      {/* Notes — sent with whichever action follows */}
      <label className="block space-y-1">
        <span className="text-2xs uppercase tracking-wide text-muted-foreground">
          Analyst note
        </span>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          rows={3}
          placeholder="Why this decision? Recorded with the action you take next."
          aria-label="Analyst note"
          className="w-full resize-y rounded-md border bg-background p-2 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
        />
      </label>

      {/* Decisions */}
      <div className="flex flex-wrap items-center gap-1.5">
        {decisions.map((action) => (
          <Button
            key={action}
            size="sm"
            variant={DECISION_VARIANTS[action] ?? 'outline'}
            disabled={isDeciding}
            onClick={() => decide(action)}
          >
            {isDeciding ? <Loader2 className="size-4 animate-spin" /> : null}
            {reviewActionLabel(action)}
          </Button>
        ))}
        <Button
          size="sm"
          variant="ghost"
          disabled={isDeciding || note.trim().length === 0}
          onClick={() => decide('note')}
        >
          <Save className="size-4" />
          Save note
        </Button>
      </div>

      {savedNote ? (
        <p className="rounded-md border bg-muted/40 p-2 text-sm">
          <span className="text-2xs uppercase tracking-wide text-muted-foreground">
            Latest saved note
          </span>
          <br />
          {savedNote}
        </p>
      ) : null}

      {/* Review metadata */}
      <dl className="space-y-1 rounded-md border p-3">
        <MetadataRow label="Status" value={reviewStatusLabel(status)} />
        <MetadataRow label="Reviewer" value={reviewCase.reviewer_id ?? '—'} />
        <MetadataRow
          label="Decided"
          value={reviewCase.decided_at ? formatDateTime(reviewCase.decided_at) : '—'}
        />
        <MetadataRow
          label="Updated"
          value={reviewCase.updated_at ? formatDateTime(reviewCase.updated_at) : '—'}
        />
        {reviewCase.reason ? <MetadataRow label="Reason" value={reviewCase.reason} /> : null}
      </dl>

      {/* Audit history */}
      <section className="space-y-2" aria-label="Audit history">
        <h4 className="flex items-center gap-1.5 text-2xs uppercase tracking-wide text-muted-foreground">
          <History className="size-3.5" aria-hidden="true" />
          Audit history
        </h4>
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No analyst action recorded yet. Starting a review will appear here.
          </p>
        ) : (
          <ol className="space-y-0">
            {history.map((entry, index) => (
              <li key={entry.entry_id} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <span
                    aria-hidden="true"
                    className={cn(
                      'mt-1.5 size-1.5 shrink-0 rounded-full',
                      index === history.length - 1 ? 'bg-primary' : 'bg-muted-foreground/40',
                    )}
                  />
                  {index === history.length - 1 ? null : (
                    <span aria-hidden="true" className="w-px flex-1 bg-border" />
                  )}
                </div>
                <div className={cn('min-w-0 flex-1', index === history.length - 1 || 'pb-3')}>
                  <p className="text-sm font-medium">{historyLabel(entry)}</p>
                  <p className="text-2xs text-muted-foreground">
                    {entry.reviewer} · {formatDateTime(entry.at)}
                  </p>
                  {entry.note ? <p className="text-xs">{entry.note}</p> : null}
                  {entry.reason ? (
                    <p className="text-xs text-muted-foreground">{entry.reason}</p>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}

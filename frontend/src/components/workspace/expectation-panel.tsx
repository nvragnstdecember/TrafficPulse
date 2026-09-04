import { ClipboardCheck, Loader2, Save, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { type ViolationType } from '@/api/types';
import {
  useDeclareExpectation,
  useExpectation,
  useExpectationComparison,
  useWithdrawExpectation,
} from '@/hooks/use-demo';
import {
  DEMO_SELECTABLE_VIOLATIONS,
  OUTCOME_LABEL,
  OUTCOME_MEANING,
  OUTCOME_TONE,
  comparisonSummary,
  isAsDeclared,
} from '@/lib/demo';
import { formatDateTime } from '@/lib/format';
import { violationLabel } from '@/lib/workspace';
import { notify } from '@/store/notifications-store';

import { ErrorBanner } from '../common/error-banner';
import { StatusChip } from '../common/status-chip';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';

export interface ExpectationPanelProps {
  videoId: string;
  /** The run to compare against; comparison is only meaningful once it finished. */
  jobId: string | null;
  /** Whether that run has completed — a mid-run comparison would report a partial answer. */
  runComplete: boolean;
  /** Open one of the confirmed events behind a detected count. */
  onOpenEvent?: (eventId: string) => void;
}

const REVIEWER = 'analyst';

/**
 * Controlled demonstration: declare what a clip contains, then compare (§7).
 *
 * The one surface where ground truth and detections appear together, and it is built
 * so they can never be confused. The left column is what somebody **declared**; the
 * right is what the reasoners **confirmed**, as counts of real event ids that open
 * into the same evidence viewer as any other event. The panel writes nothing into the
 * event list, and the backend keeps the declaration in a store no rule can read.
 *
 * It renders only its declaration form until a run has finished. A comparison drawn
 * mid-run would show a partial answer in a table headed "detected", which is exactly
 * the sort of thing that gets misread in a live demo.
 */
export function ExpectationPanel({
  videoId,
  jobId,
  runComplete,
  onOpenEvent,
}: ExpectationPanelProps) {
  const expectationQuery = useExpectation(videoId);
  const comparisonQuery = useExpectationComparison(
    videoId,
    jobId ?? undefined,
    runComplete,
  );
  const declare = useDeclareExpectation();
  const withdraw = useWithdrawExpectation();

  const saved = expectationQuery.data ?? null;
  const [selected, setSelected] = useState<ViolationType[]>([]);
  const [notes, setNotes] = useState('');
  // Mirror the stored declaration into the form once it arrives, and again whenever
  // it changes. Keyed on the record itself: react-query keeps the reference stable
  // while the content is unchanged, so a background refetch cannot stamp over what
  // the analyst is currently typing.
  useEffect(() => {
    setSelected(saved?.expected_violations ?? []);
    setNotes(saved?.notes ?? '');
  }, [saved]);

  const comparison = comparisonQuery.data ?? null;
  const summary = useMemo(() => comparisonSummary(comparison), [comparison]);

  function toggle(violation: ViolationType): void {
    setSelected((current) =>
      current.includes(violation)
        ? current.filter((v) => v !== violation)
        : [...current, violation],
    );
  }

  function handleDeclare(): void {
    declare.mutate(
      {
        videoId,
        declaration: {
          expected_violations: selected,
          notes: notes.trim(),
          declared_by: REVIEWER,
        },
      },
      {
        onSuccess: () =>
          notify({
            title: 'Expectation declared.',
            description:
              'Recorded as this demonstration’s ground truth. It does not affect processing.',
          }),
      },
    );
  }

  function handleWithdraw(): void {
    withdraw.mutate(videoId, {
      onSuccess: () => notify({ title: 'Expectation withdrawn. Confirmed events are unchanged.' }),
    });
  }

  const dirty =
    notes.trim() !== (saved?.notes ?? '').trim() ||
    selected.length !== (saved?.expected_violations.length ?? 0) ||
    selected.some((v) => !saved?.expected_violations.includes(v));

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 p-4 pb-2">
        <CardTitle className="flex items-center gap-2">
          <ClipboardCheck className="size-4 text-muted-foreground" aria-hidden="true" />
          Controlled demo · expected vs detected
        </CardTitle>
        {comparison?.expectation ? (
          <StatusChip
            tone={isAsDeclared(comparison) ? 'success' : 'warning'}
            label={isAsDeclared(comparison) ? 'As declared' : 'Differs from declaration'}
          />
        ) : (
          <StatusChip tone="neutral" label={saved ? 'Declared' : 'Nothing declared'} />
        )}
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-2">
        <p className="text-xs text-muted-foreground">
          For a clip somebody <em>built</em> to contain known situations. What you declare
          here is this demonstration&apos;s ground truth — it is never shown to the
          reasoners, never becomes an event, and never appears in the event list. It is
          compared with what the run confirmed, afterwards.
        </p>

        <fieldset className="space-y-2">
          <legend className="text-2xs uppercase tracking-wide text-muted-foreground">
            This clip was built to contain
          </legend>
          <div className="flex flex-wrap gap-1.5">
            {DEMO_SELECTABLE_VIOLATIONS.map((violation) => {
              const active = selected.includes(violation);
              return (
                <Button
                  key={violation}
                  size="sm"
                  variant={active ? 'default' : 'outline'}
                  aria-pressed={active}
                  onClick={() => toggle(violation)}
                >
                  {violationLabel(violation)}
                </Button>
              );
            })}
          </div>
          {selected.length === 0 ? (
            <p className="text-2xs text-muted-foreground">
              Nothing selected. Declaring an empty expectation is legitimate — it claims the
              clip contains no violation, which the run can then confirm or contradict.
            </p>
          ) : null}
        </fieldset>

        <div className="space-y-1.5">
          <Label htmlFor="expectation-notes" className="text-2xs uppercase tracking-wide">
            Test context
          </Label>
          <textarea
            id="expectation-notes"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            rows={3}
            maxLength={4000}
            placeholder="What this controlled scenario is, and how its context was chosen. e.g. “Synthetic clip; the no-stopping zone is artificially designated for demonstration; the signal schedule is declared, not observed.”"
            className="w-full rounded-md border bg-background px-2 py-1.5 text-xs
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={handleDeclare} disabled={declare.isPending || !dirty}>
            <Save className="size-4" />
            {declare.isPending ? 'Saving…' : saved ? 'Update declaration' : 'Declare'}
          </Button>
          {saved ? (
            <Button
              size="sm"
              variant="outline"
              onClick={handleWithdraw}
              disabled={withdraw.isPending}
            >
              <Trash2 className="size-4" />
              Withdraw
            </Button>
          ) : null}
          {saved ? (
            <span className="text-2xs text-muted-foreground">
              Declared by {saved.declared_by} · {formatDateTime(saved.declared_at)}
            </span>
          ) : null}
        </div>

        {declare.isError ? (
          <ErrorBanner title="Could not save the declaration" error={declare.error} />
        ) : null}
        {withdraw.isError ? (
          <ErrorBanner title="Could not withdraw the declaration" error={withdraw.error} />
        ) : null}

        <ComparisonTable
          summary={summary}
          comparison={comparison}
          runComplete={runComplete}
          isLoading={comparisonQuery.isLoading}
          onOpenEvent={onOpenEvent}
        />
      </CardContent>
    </Card>
  );
}

function ComparisonTable({
  summary,
  comparison,
  runComplete,
  isLoading,
  onOpenEvent,
}: {
  summary: string;
  comparison: ReturnType<typeof useExpectationComparison>['data'] | null;
  runComplete: boolean;
  isLoading: boolean;
  onOpenEvent?: (eventId: string) => void;
}) {
  if (!runComplete) {
    return (
      <p className="rounded-md border border-dashed p-3 text-2xs text-muted-foreground">
        The comparison appears once a run has finished. Declared families and confirmed
        events are only put side by side against a completed run — a partial answer under
        a “detected” heading is the easiest thing in this workspace to misread.
      </p>
    );
  }
  if (isLoading) {
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
        Comparing…
      </p>
    );
  }
  if (!comparison || comparison.rows.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Nothing declared and nothing confirmed for this run.
      </p>
    );
  }

  return (
    <section className="space-y-2" aria-label="Expected versus detected">
      <p className="text-xs font-medium">{summary}</p>
      <table className="w-full text-left text-xs">
        <thead className="text-2xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th scope="col" className="py-1 pr-2 font-normal">
              Violation
            </th>
            <th scope="col" className="py-1 pr-2 font-normal">
              Expected
            </th>
            <th scope="col" className="py-1 pr-2 font-normal">
              Detected
            </th>
            <th scope="col" className="py-1 font-normal">
              Outcome
            </th>
          </tr>
        </thead>
        <tbody>
          {comparison.rows.map((row) => (
            <tr key={row.violation_type} className="border-t">
              <th scope="row" className="py-1.5 pr-2 font-medium">
                {violationLabel(row.violation_type)}
              </th>
              <td className="py-1.5 pr-2 text-muted-foreground">
                {row.expected ? 'Declared' : '—'}
              </td>
              <td className="py-1.5 pr-2">
                {row.detected_count === 0 ? (
                  <span className="text-muted-foreground">0</span>
                ) : (
                  <span className="flex flex-wrap items-center gap-1">
                    <span className="tabular-nums">{row.detected_count}</span>
                    {onOpenEvent
                      ? row.event_ids.map((eventId, index) => (
                          <button
                            key={eventId}
                            type="button"
                            onClick={() => onOpenEvent(eventId)}
                            className="rounded-sm text-2xs text-muted-foreground underline
                              underline-offset-2 hover:text-foreground focus-visible:outline-none
                              focus-visible:ring-2 focus-visible:ring-ring"
                          >
                            open #{index + 1}
                          </button>
                        ))
                      : null}
                  </span>
                )}
              </td>
              <td className="py-1.5">
                <StatusChip
                  tone={OUTCOME_TONE[row.outcome]}
                  label={OUTCOME_LABEL[row.outcome]}
                  dot={false}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <ul className="space-y-0.5 text-2xs text-muted-foreground">
        {[...new Set(comparison.rows.map((row) => row.outcome))].map((outcome) => (
          <li key={outcome}>
            <span className="font-medium text-foreground">{OUTCOME_LABEL[outcome]}</span>{' '}
            — {OUTCOME_MEANING[outcome]}
          </li>
        ))}
      </ul>
      <p className="text-2xs text-muted-foreground">
        No precision, recall or F1 is shown. Over one hand-authored clip those would be
        arithmetic against ground truth the same person wrote.
      </p>
    </section>
  );
}

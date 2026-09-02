import { ShieldCheck } from 'lucide-react';

import { StatusChip } from '@/components/common/status-chip';
import { type LiveSessionMessage } from '@/api/types';
import { type LiveEventRow, formatWallClock } from '@/lib/live';
import { violationLabel, violationTone } from '@/lib/workspace';

export interface LiveEventFeedProps {
  events: LiveEventRow[];
  session: LiveSessionMessage | null;
  monitoring: boolean;
}

/**
 * The live event feed, and the notice that makes an empty feed readable.
 *
 * An empty violation list has two completely different meanings — "nothing
 * happened" and "that violation is not being evaluated on this camera" — and a
 * viewer cannot tell them apart from the list itself. So the unavailable
 * violations are printed here with the server's own reason for each, next to the
 * feed rather than on a settings page, because this is the list they qualify.
 *
 * Events are the confirmed ones only. Nothing appears here because a model emitted
 * a box: the reasoners' persistence and evidence rules are what put a row on this
 * list, exactly as they do for an uploaded video.
 */
export function LiveEventFeed({ events, session, monitoring }: LiveEventFeedProps) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex min-h-0 flex-1 flex-col rounded-lg border bg-card">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <h2 className="text-sm font-medium">Live events</h2>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {events.length}
          </span>
        </div>

        <ul className="min-h-0 flex-1 divide-y overflow-y-auto">
          {events.length === 0 ? (
            <li className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted-foreground">
              <ShieldCheck className="size-6" aria-hidden="true" />
              <span>
                {monitoring
                  ? 'No violation confirmed yet. A violation appears here only once a reasoner has sustained it — never from a single frame.'
                  : 'Start monitoring to watch for violations.'}
              </span>
            </li>
          ) : (
            events.map((event) => (
              <li key={event.id} className="flex items-start gap-3 px-3 py-2.5">
                <span className="mt-0.5 font-mono text-xs tabular-nums text-muted-foreground">
                  {formatWallClock(event.receivedAt)}
                </span>
                <span className="min-w-0 flex-1 space-y-1">
                  <StatusChip
                    tone={violationTone(event.violationType)}
                    label={event.label}
                    dot={false}
                  />
                  <span className="block truncate text-xs text-muted-foreground">
                    {event.trackIds.length > 0
                      ? `Track ${event.trackIds.join(', ')}`
                      : 'No track attributed'}
                    {' · '}
                    sustained {event.observedSeconds.toFixed(1)}s
                  </span>
                </span>
              </li>
            ))
          )}
        </ul>
      </div>

      {session ? (
        <div className="rounded-lg border bg-card p-3 text-xs">
          <p className="mb-2 font-medium">
            Evaluated on this camera:{' '}
            <span className="font-normal text-muted-foreground">
              {session.running_violations.length > 0
                ? session.running_violations.map(violationLabel).join(', ')
                : 'nothing'}
            </span>
          </p>
          {session.unavailable_violations.length > 0 ? (
            <dl className="space-y-2">
              {session.unavailable_violations.map((entry) => (
                <div key={entry.violation_type}>
                  <dt className="font-medium text-muted-foreground">
                    {violationLabel(entry.violation_type)} — not evaluated
                  </dt>
                  <dd className="leading-relaxed text-muted-foreground">{entry.reason}</dd>
                </div>
              ))}
            </dl>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

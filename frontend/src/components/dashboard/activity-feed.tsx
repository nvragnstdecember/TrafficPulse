import { ClipboardCheck, Cpu, Upload } from 'lucide-react';

import { type ActivityEntry } from '@/api/types';
import { activityKindLabel } from '@/lib/analytics';
import { formatDateTime } from '@/lib/format';

const ICONS: Record<string, typeof Upload> = {
  upload: Upload,
  run: Cpu,
  review: ClipboardCheck,
};

export interface ActivityFeedProps {
  entries: ActivityEntry[];
  emptyMessage?: string;
}

/**
 * The recent-activity feed (H15): what has happened in this repository lately.
 *
 * A list, not a chart. The entries are mixed-type, text-heavy, and individually
 * meaningful — exactly the case a chronological list serves better than any
 * visualization.
 *
 * Every timestamp here is a **wall-clock** instant the backend recorded (an
 * upload, a run transition, a review action). Media time never reaches this
 * component: an event's `trigger_at` is anchored at a fixed 1970 epoch, so
 * rendering it as a date would put all activity half a century in the past.
 */
export function ActivityFeed({ entries, emptyMessage }: ActivityFeedProps) {
  if (entries.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {emptyMessage ?? 'No recorded activity yet.'}
      </p>
    );
  }

  return (
    <ul className="space-y-2.5" aria-label="Recent activity">
      {entries.map((entry) => {
        const Icon = ICONS[entry.kind] ?? Cpu;
        return (
          <li key={`${entry.kind}-${entry.subject_id}-${entry.at}`} className="flex gap-2.5">
            <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <div className="min-w-0 flex-1 space-y-0.5">
              <p className="truncate text-xs text-foreground">{entry.summary}</p>
              <p className="text-2xs text-muted-foreground">
                <span className="uppercase tracking-wide">
                  {activityKindLabel(entry.kind)}
                </span>
                {' · '}
                {formatDateTime(entry.at)}
              </p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

import { Plus, Trash2 } from 'lucide-react';

import { type SignalPhaseSpec, type SignalState } from '@/api/types';
import { SIGNAL_STATES, isScheduleUsable, sortSchedule } from '@/lib/calibration';

import { Button } from '../ui/button';
import { Input } from '../ui/input';

export interface SignalScheduleEditorProps {
  schedule: SignalPhaseSpec[];
  onChange: (schedule: SignalPhaseSpec[]) => void;
  /** Whether the scene has the junction geometry that makes a schedule meaningful. */
  enabled: boolean;
}

const STATE_LABEL: Record<SignalState, string> = {
  red: 'Red',
  amber: 'Amber',
  green: 'Green',
  off: 'Off',
  unknown: 'Unknown',
};

/**
 * The per-run signal schedule (H13).
 *
 * Timing belongs to the *video*, not to the camera: a phase names a media-time
 * instant, and a scene is shared across many clips. So this is not part of the scene
 * draft — it travels with the processing request, and the analyst enters it in
 * seconds from the start of the clip, which is the number a player's scrub bar
 * reads out.
 *
 * A schedule the backend would refuse is refused here too, with the reason: an empty
 * one resolves every instant to `unknown` and could never confirm, so offering to
 * run with it would promise an analysis that structurally cannot happen.
 */
export function SignalScheduleEditor({
  schedule,
  onChange,
  enabled,
}: SignalScheduleEditorProps) {
  if (!enabled) return null;

  function updatePhase(index: number, patch: Partial<SignalPhaseSpec>): void {
    onChange(schedule.map((phase, i) => (i === index ? { ...phase, ...patch } : phase)));
  }

  return (
    <section className="space-y-2 rounded-md border p-3" aria-label="Signal schedule">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-2xs uppercase tracking-wide text-muted-foreground">
          Signal timing (this video)
        </h3>
        <Button
          size="sm"
          variant="outline"
          onClick={() =>
            onChange([
              ...schedule,
              { at_seconds: schedule.length ? schedule[schedule.length - 1].at_seconds + 5 : 0, state: 'red' },
            ])
          }
        >
          <Plus className="size-4" />
          Add phase
        </Button>
      </div>

      <p className="text-2xs text-muted-foreground">
        The signal state is declared, not detected — TrafficPulse does not read a signal
        head from pixels. Scrub the video to each change and record the time here.
      </p>

      {schedule.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No phases yet. Red-light detection needs at least one.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {schedule.map((phase, index) => (
            <li key={index} className="flex items-center gap-2">
              <Input
                type="number"
                min={0}
                step={0.1}
                value={phase.at_seconds}
                aria-label={`Phase ${index + 1} start seconds`}
                className="h-8 w-24"
                onChange={(event) =>
                  updatePhase(index, { at_seconds: Number(event.target.value) })
                }
              />
              <span className="text-2xs text-muted-foreground">s</span>
              <select
                value={phase.state}
                aria-label={`Phase ${index + 1} state`}
                className="h-8 rounded-md border bg-background px-2 text-sm"
                onChange={(event) =>
                  updatePhase(index, { state: event.target.value as SignalState })
                }
              >
                {SIGNAL_STATES.map((state) => (
                  <option key={state} value={state}>
                    {STATE_LABEL[state]}
                  </option>
                ))}
              </select>
              <Button
                size="sm"
                variant="ghost"
                aria-label={`Remove phase ${index + 1}`}
                onClick={() => onChange(schedule.filter((_, i) => i !== index))}
              >
                <Trash2 className="size-4" />
              </Button>
            </li>
          ))}
        </ul>
      )}

      {schedule.length > 0 && !isScheduleUsable(schedule) ? (
        <p role="alert" className="text-2xs text-destructive">
          Phases must be in non-decreasing time order.
        </p>
      ) : null}
      {schedule.length > 1 ? (
        <Button size="sm" variant="ghost" onClick={() => onChange(sortSchedule(schedule))}>
          Sort by time
        </Button>
      ) : null}
    </section>
  );
}

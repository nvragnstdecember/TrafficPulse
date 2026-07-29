import { Check } from 'lucide-react';

import { type ProcessingPhase, type WorkflowStage, stageStates } from '@/lib/job';
import { cn } from '@/lib/utils';

export interface WorkflowNavProps {
  phase: ProcessingPhase;
  /** True once an event is selected — review has progressed into evidence. */
  hasSelection: boolean;
  /** Jump to a reachable stage. */
  onNavigate?: (stage: WorkflowStage) => void;
}

const STAGES: Array<{ id: WorkflowStage; label: string }> = [
  { id: 'upload', label: 'Upload' },
  { id: 'processing', label: 'Processing' },
  { id: 'review', label: 'Review' },
  { id: 'evidence', label: 'Evidence' },
];

/**
 * Where the analyst is in the upload → processing → review → evidence flow.
 *
 * Stage states come from `lib/job.stageStates`, so this component holds no state
 * of its own. Completed stages are navigable; stages not yet reached are inert
 * rather than hidden, so the shape of the whole workflow is visible from the very
 * first screen instead of appearing piece by piece.
 */
export function WorkflowNav({ phase, hasSelection, onNavigate }: WorkflowNavProps) {
  const states = stageStates(phase, hasSelection);

  return (
    <nav aria-label="Review workflow">
      <ol className="flex flex-wrap items-center gap-1">
        {STAGES.map((stage, index) => {
          const state = states[stage.id];
          const reachable = state !== 'todo' && Boolean(onNavigate);
          return (
            <li key={stage.id} className="flex items-center gap-1">
              {index > 0 ? <span aria-hidden="true" className="h-px w-4 bg-border sm:w-6" /> : null}
              <button
                type="button"
                onClick={reachable ? () => onNavigate?.(stage.id) : undefined}
                disabled={!reachable}
                aria-current={state === 'current' ? 'step' : undefined}
                className={cn(
                  'flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors',
                  state === 'current' && 'border-primary bg-primary/10 font-medium text-foreground',
                  state === 'done' && 'border-transparent bg-muted text-muted-foreground',
                  state === 'todo' && 'border-dashed text-muted-foreground/60',
                  reachable && 'hover:bg-accent',
                )}
              >
                {state === 'done' ? (
                  <Check className="size-3" aria-hidden="true" />
                ) : (
                  <span
                    aria-hidden="true"
                    className={cn(
                      'size-1.5 rounded-full',
                      state === 'current' ? 'bg-primary' : 'bg-muted-foreground/40',
                    )}
                  />
                )}
                {stage.label}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

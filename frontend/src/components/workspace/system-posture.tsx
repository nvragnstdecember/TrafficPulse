import { ShieldAlert } from 'lucide-react';

import { type SystemPosture } from '@/api/types';
import { postureTone } from '@/lib/analysis';

import { StatusChip } from '../common/status-chip';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip';

export interface SystemPostureStripProps {
  posture: SystemPosture | undefined;
}

/**
 * The capability status strip: what this deployment can honestly claim.
 *
 * Placed where the run is watched rather than buried in a settings page, because its
 * whole purpose is to be read *next to* the output it qualifies. A viewer shown boxes
 * and helmet labels will read them as enforcement decisions unless something on the
 * same screen says otherwise, and the evidence does not support that reading.
 *
 * Two design rules, both deliberate.
 *
 * **It is never an alarm.** Every component here is behaving exactly as designed: the
 * capability guard is refusing an unsafe rule, not failing. The strip is muted and
 * informational, so a limitation reads as a stated boundary rather than as a broken
 * system — which is also the honest presentation, since nothing is broken.
 *
 * **The reason travels with the state.** A bare `UNAVAILABLE` invites a viewer to
 * assume a bug or an oversight. The server sends a complete sentence for each
 * component; it is one hover away on every row and printed in full for the one
 * component that governs what may be claimed.
 */
export function SystemPostureStrip({ posture }: SystemPostureStripProps) {
  if (!posture) return null;

  const enforcement = posture.components.find(
    (component) => component.component_id === 'helmet_enforcement',
  );

  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2 p-4 pb-2">
        <ShieldAlert className="size-4 text-muted-foreground" aria-hidden="true" />
        <CardTitle className="text-base">System capabilities</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 p-4 pt-2">
        <dl className="space-y-1.5">
          {posture.components.map((component) => (
            <div
              key={component.component_id}
              className="flex items-center justify-between gap-3"
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <dt className="cursor-help truncate border-b border-dotted border-muted-foreground/40 text-sm">
                    {component.label}
                  </dt>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">{component.detail}</TooltipContent>
              </Tooltip>
              <dd className="shrink-0">
                <StatusChip
                  tone={postureTone(component.state)}
                  label={component.state}
                  className="text-2xs"
                />
              </dd>
            </div>
          ))}
        </dl>

        {enforcement ? (
          <p className="border-t pt-3 text-xs leading-relaxed text-muted-foreground">
            {enforcement.detail}
          </p>
        ) : null}

        {posture.helmet_backend_labels.length > 0 ? (
          <p className="text-2xs text-muted-foreground">
            Helmet backend emits:{' '}
            <span className="font-mono">{posture.helmet_backend_labels.join(', ')}</span>
            {posture.turban_capable ? null : ' — no turban label, so no exemption can fire.'}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

import { ChevronDown } from 'lucide-react';
import { useId, useState } from 'react';

import { cn } from '@/lib/utils';

export interface CollapsibleSectionProps {
  title: string;
  /** Open on first render (uncontrolled). */
  defaultOpen?: boolean;
  /** Optional content shown at the right of the header (e.g. a copy button). */
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

/**
 * A lightweight, accessible collapsible section (H7E).
 *
 * A disclosure button (`aria-expanded` + `aria-controls`) toggling a labelled
 * region — no dependency, keyboard-operable, and used to make the detail panel's
 * sections collapsible without changing its layout.
 */
export function CollapsibleSection({
  title,
  defaultOpen = true,
  action,
  children,
  className,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const regionId = useId();

  return (
    <div className={cn('rounded-md border', className)}>
      <div className="flex items-center justify-between gap-2 pr-1.5">
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-controls={regionId}
          className="flex flex-1 items-center gap-2 px-3 py-2 text-left text-sm font-medium"
        >
          <ChevronDown
            className={cn('size-4 shrink-0 transition-transform', open ? '' : '-rotate-90')}
            aria-hidden="true"
          />
          {title}
        </button>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div id={regionId} hidden={!open} className="border-t px-3 py-2">
        {children}
      </div>
    </div>
  );
}

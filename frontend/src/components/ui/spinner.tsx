import { Loader2 } from 'lucide-react';

import { cn } from '@/lib/utils';

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Accessible label announced to screen readers. */
  label?: string;
  size?: number;
}

/** An accessible loading spinner (uses reduced-motion-aware CSS animation). */
export function Spinner({ label = 'Loading', size = 20, className, ...props }: SpinnerProps) {
  return (
    <div role="status" aria-live="polite" className={cn('inline-flex', className)} {...props}>
      <Loader2
        className="animate-spin text-muted-foreground"
        style={{ width: size, height: size }}
      />
      <span className="sr-only">{label}</span>
    </div>
  );
}

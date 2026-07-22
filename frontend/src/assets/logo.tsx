import { cn } from '@/lib/utils';

/** The TrafficPulse mark: a rounded tile with a stylized signal pulse. */
export function Logo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      role="img"
      aria-label="TrafficPulse"
      className={cn('text-primary', className)}
    >
      <rect x="1" y="1" width="30" height="30" rx="8" className="fill-current" />
      <path
        d="M7 16h4l2.5-6 5 12 2.5-6H25"
        fill="none"
        stroke="hsl(var(--primary-foreground))"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

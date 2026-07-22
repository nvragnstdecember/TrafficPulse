import { useHealth } from '@/hooks/use-system';
import { APP_VERSION } from '@/lib/app-info';
import { cn } from '@/lib/utils';

interface Indicator {
  tone: 'success' | 'warning' | 'error' | 'neutral';
  label: string;
}

const TONE_DOT: Record<Indicator['tone'], string> = {
  success: 'bg-success',
  warning: 'bg-warning',
  error: 'bg-destructive',
  neutral: 'bg-muted-foreground',
};

/** The status bar: live backend/engine health plus the app version. */
export function StatusFooter() {
  const { data, isLoading, isError } = useHealth();

  let backend: Indicator;
  let engine: Indicator | null = null;
  if (isLoading) {
    backend = { tone: 'neutral', label: 'Connecting…' };
  } else if (isError || !data) {
    backend = { tone: 'error', label: 'Backend unreachable' };
  } else {
    backend = { tone: 'success', label: `Backend ${data.status}` };
    engine = {
      tone: data.engine === 'ready' ? 'success' : 'warning',
      label: `Engine ${data.engine}`,
    };
  }

  return (
    <footer className="flex h-8 shrink-0 items-center justify-between border-t bg-background px-4 text-xs text-muted-foreground">
      <div className="flex items-center gap-4">
        <StatusItem indicator={backend} />
        {engine ? <StatusItem indicator={engine} /> : null}
      </div>
      <span className="tabular-nums">v{APP_VERSION}</span>
    </footer>
  );
}

function StatusItem({ indicator }: { indicator: Indicator }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={cn('size-1.5 rounded-full', TONE_DOT[indicator.tone])} aria-hidden="true" />
      {indicator.label}
    </span>
  );
}

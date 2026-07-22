import { Spinner } from '../ui/spinner';

/** A centered loading indicator for route/Suspense fallbacks. */
export function PageLoader({ label = 'Loading page' }: { label?: string }) {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <Spinner size={28} label={label} />
    </div>
  );
}

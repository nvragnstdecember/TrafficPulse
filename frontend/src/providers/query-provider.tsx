import { QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

import { createQueryClient } from '@/api/query-client';
import { toErrorMessage } from '@/api/errors';
import { notify } from '@/store/notifications-store';

/**
 * TanStack Query provider (H7B).
 *
 * Builds one client per app instance (kept stable via `useState`) and wires the
 * global error handler to the notifications store, so any unhandled query/
 * mutation failure surfaces as a toast without per-call boilerplate.
 */
export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() =>
    createQueryClient({
      onError: (error) =>
        notify({ title: 'Request failed', description: toErrorMessage(error), variant: 'error' }),
    }),
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

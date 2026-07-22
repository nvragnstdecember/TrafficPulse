import { RouterProvider } from 'react-router-dom';
import { useState } from 'react';

import { AppProviders } from '@/providers/app-providers';
import { createRouter } from '@/routes/router';

/** The root application: providers wrapping the browser router. */
export function App() {
  const [router] = useState(() => createRouter());
  return (
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>
  );
}

import { isRouteErrorResponse, useRouteError } from 'react-router-dom';

import { FullPageError } from '@/components/common/full-page-error';
import { toErrorMessage } from '@/api/errors';

/** Router-level `errorElement`: renders thrown/loader errors gracefully. */
export function RouteError() {
  const error = useRouteError();

  if (isRouteErrorResponse(error)) {
    return (
      <FullPageError
        title={`${error.status} ${error.statusText}`}
        message={error.data?.message ?? 'This route could not be loaded.'}
      />
    );
  }

  return <FullPageError message={toErrorMessage(error)} />;
}

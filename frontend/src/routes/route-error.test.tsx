import { render, screen } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { RouteError } from './route-error';

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

function renderWithError(child: { loader?: () => never; element?: React.ReactElement }) {
  const router = createMemoryRouter([
    {
      path: '/',
      loader: child.loader,
      element: child.element ?? <div>ok</div>,
      errorElement: <RouteError />,
    },
  ]);
  return render(<RouterProvider router={router} />);
}

function Throw(): never {
  throw new Error('render failed');
}

describe('RouteError', () => {
  it('renders a route error response (status + text)', async () => {
    renderWithError({
      loader: () => {
        throw new Response(null, { status: 503, statusText: 'Service Unavailable' });
      },
    });
    expect(await screen.findByText('503 Service Unavailable')).toBeInTheDocument();
  });

  it('renders a generic thrown error', async () => {
    renderWithError({ element: <Throw /> });
    expect(await screen.findByText('render failed')).toBeInTheDocument();
  });
});

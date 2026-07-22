import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ErrorBoundary } from './error-boundary';

function Boom(): never {
  throw new Error('render exploded');
}

beforeEach(() => {
  // Silence the expected React error log for the thrown render.
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ErrorBoundary', () => {
  it('renders the fallback when a child throws', () => {
    render(
      <MemoryRouter>
        <ErrorBoundary>
          <Boom />
        </ErrorBoundary>
      </MemoryRouter>,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('render exploded');
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('supports a custom fallback and reset', async () => {
    const user = userEvent.setup();

    function Harness() {
      const [failing, setFailing] = useState(true);
      return (
        <ErrorBoundary
          fallback={({ reset }) => (
            <button
              onClick={() => {
                setFailing(false);
                reset();
              }}
            >
              recover
            </button>
          )}
        >
          {failing ? <Boom /> : <span>recovered content</span>}
        </ErrorBoundary>
      );
    }

    render(<Harness />);
    await user.click(screen.getByRole('button', { name: 'recover' }));
    expect(screen.getByText('recovered content')).toBeInTheDocument();
  });
});

import { screen } from '@testing-library/react';
import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { App } from './app';

// Keep the shell's health poll deterministic and offline.
vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: vi.fn(async () => ({ status: 'ok', version: '0.1.0', engine: 'ready' })),
    getMetrics: vi.fn(),
  },
}));

describe('App', () => {
  it('boots the full provider + router stack and renders the dashboard', async () => {
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Evidence' })).toBeInTheDocument();
  });
});

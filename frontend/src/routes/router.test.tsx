import { screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderRoutesAt } from '@/test/utils';
import { useUiStore } from '@/store/ui-store';

// The shell footer polls health; keep it deterministic and offline.
vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: vi.fn(async () => ({ status: 'ok', version: '0.1.0', engine: 'ready' })),
    getMetrics: vi.fn(),
  },
}));

beforeEach(() => {
  localStorage.clear();
  useUiStore.setState({ sidebarCollapsed: false, mobileSidebarOpen: false });
});

describe('routing', () => {
  it('renders the dashboard at the index route inside the shell', async () => {
    renderRoutesAt('/');
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
    // The shell navigation is present alongside the page.
    expect(screen.getByRole('link', { name: 'Videos' })).toBeInTheDocument();
  });

  it('renders each top-level page', async () => {
    renderRoutesAt('/videos');
    expect(await screen.findByRole('heading', { name: 'Video workspace' })).toBeInTheDocument();
  });

  it('renders the functional settings page', async () => {
    renderRoutesAt('/settings');
    expect(await screen.findByRole('heading', { name: 'Settings' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Theme' })).toBeInTheDocument();
  });

  it('renders the 404 page for unknown routes', async () => {
    renderRoutesAt('/does-not-exist');
    expect(await screen.findByRole('heading', { name: /page not found/i })).toBeInTheDocument();
  });
});

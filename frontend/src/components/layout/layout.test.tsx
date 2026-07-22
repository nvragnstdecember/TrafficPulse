import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '@/test/utils';
import { useUiStore } from '@/store/ui-store';

import { Sidebar } from './sidebar';
import { StatusFooter } from './status-footer';
import { TopNav } from './top-nav';

const getHealth = vi.fn();
vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: () => getHealth(),
    getMetrics: vi.fn(),
  },
}));

beforeEach(() => {
  localStorage.clear();
  useUiStore.setState({ sidebarCollapsed: false, mobileSidebarOpen: false });
});

describe('Sidebar', () => {
  it('renders the primary navigation', () => {
    renderWithProviders(<Sidebar />);
    for (const label of ['Dashboard', 'Videos', 'Evidence', 'Analytics', 'Settings']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument();
    }
  });

  it('collapses via the toggle', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Sidebar />);
    await user.click(screen.getByRole('button', { name: /collapse sidebar/i }));
    expect(useUiStore.getState().sidebarCollapsed).toBe(true);
    expect(screen.getByRole('button', { name: /expand sidebar/i })).toBeInTheDocument();
  });
});

describe('TopNav', () => {
  it('opens the mobile navigation drawer', async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopNav />);
    await user.click(screen.getByRole('button', { name: /open navigation/i }));
    expect(useUiStore.getState().mobileSidebarOpen).toBe(true);
  });

  it('exposes the account menu', async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopNav />);
    await user.click(screen.getByRole('button', { name: /account menu/i }));
    expect(await screen.findByRole('menuitem', { name: /settings/i })).toBeInTheDocument();
  });
});

describe('StatusFooter', () => {
  it('shows healthy backend + engine status', async () => {
    getHealth.mockResolvedValue({ status: 'ok', version: '0.1.0', engine: 'ready' });
    renderWithProviders(<StatusFooter />);
    await waitFor(() => expect(screen.getByText('Backend ok')).toBeInTheDocument());
    expect(screen.getByText('Engine ready')).toBeInTheDocument();
    expect(screen.getByText('v0.1.0')).toBeInTheDocument();
  });

  it('shows an unreachable state on failure', async () => {
    getHealth.mockRejectedValue(new Error('down'));
    renderWithProviders(<StatusFooter />);
    await waitFor(() => expect(screen.getByText('Backend unreachable')).toBeInTheDocument());
  });
});

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';

import { renderWithProviders } from '@/test/utils';
import { useSettingsStore } from '@/store/settings-store';

import EvidencePage from './evidence-page';
import SettingsPage from './settings-page';

beforeEach(() => {
  localStorage.clear();
  useSettingsStore.getState().reset();
});

describe('placeholder pages', () => {
  // Analytics stopped being a placeholder in H15; it is covered, with its data
  // source mocked, in dashboard-page.test.tsx.
  it('render their headers', () => {
    renderWithProviders(<EvidencePage />);
    expect(screen.getByRole('heading', { name: 'Evidence' })).toBeInTheDocument();
  });
});

describe('SettingsPage', () => {
  it('updates density and page-size preferences', async () => {
    const user = userEvent.setup();
    renderWithProviders(<SettingsPage />);

    await user.click(screen.getByRole('button', { name: 'Compact' }));
    expect(useSettingsStore.getState().density).toBe('compact');

    await user.click(screen.getByRole('button', { name: '50' }));
    expect(useSettingsStore.getState().eventsPageSize).toBe(50);
  });

  it('switches the theme', async () => {
    const user = userEvent.setup();
    renderWithProviders(<SettingsPage />);
    await user.click(screen.getByRole('button', { name: 'Dark' }));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});

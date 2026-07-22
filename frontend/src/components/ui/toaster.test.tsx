import { act, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import { useNotificationsStore } from '@/store/notifications-store';

import { Toaster } from './toaster';

beforeEach(() => {
  useNotificationsStore.getState().clear();
});

describe('Toaster', () => {
  it('renders notifications from the store', () => {
    render(<Toaster />);
    act(() => {
      useNotificationsStore.getState().notify({
        title: 'Saved',
        description: 'All good',
        duration: null,
      });
    });
    expect(screen.getByText('Saved')).toBeInTheDocument();
    expect(screen.getByText('All good')).toBeInTheDocument();
  });

  it('removes a notification when dismissed', async () => {
    render(<Toaster />);
    let id = '';
    act(() => {
      id = useNotificationsStore.getState().notify({ title: 'Temp', duration: null });
    });
    expect(screen.getByText('Temp')).toBeInTheDocument();

    act(() => {
      useNotificationsStore.getState().dismiss(id);
    });
    await waitFor(() => expect(screen.queryByText('Temp')).not.toBeInTheDocument());
  });
});

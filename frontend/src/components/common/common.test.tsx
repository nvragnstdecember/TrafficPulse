import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Inbox } from 'lucide-react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '@/test/utils';

import { ConfirmDialog } from './confirm-dialog';
import { EmptyState } from './empty-state';
import { ErrorBanner } from './error-banner';
import { Pagination } from './pagination';
import { SearchInput } from './search-input';
import { StatusChip, engineTone, jobStatusTone } from './status-chip';
import { ThemeToggle } from './theme-toggle';

describe('StatusChip', () => {
  it('renders a label and maps domain statuses to tones', () => {
    renderWithProviders(<StatusChip tone="success" label="succeeded" />);
    expect(screen.getByText('succeeded')).toBeInTheDocument();
    expect(jobStatusTone('succeeded')).toBe('success');
    expect(jobStatusTone('running')).toBe('info');
    expect(jobStatusTone('pending')).toBe('neutral');
    expect(jobStatusTone('failed')).toBe('error');
    expect(engineTone('ready')).toBe('success');
    expect(engineTone('unconfigured')).toBe('warning');
    expect(engineTone('other')).toBe('neutral');
  });
});

describe('EmptyState', () => {
  it('renders title, description, and action', () => {
    renderWithProviders(
      <EmptyState
        icon={Inbox}
        title="Nothing here"
        description="Add something"
        action={<button>Add</button>}
      />,
    );
    expect(screen.getByText('Nothing here')).toBeInTheDocument();
    expect(screen.getByText('Add something')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add' })).toBeInTheDocument();
  });
});

describe('ErrorBanner', () => {
  it('shows a message and triggers retry', async () => {
    const onRetry = vi.fn();
    renderWithProviders(<ErrorBanner error={new Error('kaboom')} onRetry={onRetry} />);
    expect(screen.getByRole('alert')).toHaveTextContent('kaboom');
    await userEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

describe('SearchInput', () => {
  it('emits changes and clears', async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    const { rerender } = renderWithProviders(
      <SearchInput value="" onValueChange={onValueChange} />,
    );
    await user.type(screen.getByRole('searchbox'), 'a');
    expect(onValueChange).toHaveBeenCalledWith('a');

    rerender(<SearchInput value="abc" onValueChange={onValueChange} />);
    await user.click(screen.getByRole('button', { name: /clear search/i }));
    expect(onValueChange).toHaveBeenLastCalledWith('');
  });
});

describe('Pagination', () => {
  it('summarizes the range and disables edges', async () => {
    const user = userEvent.setup();
    const onOffsetChange = vi.fn();
    const { rerender } = renderWithProviders(
      <Pagination total={45} limit={10} offset={0} onOffsetChange={onOffsetChange} />,
    );
    expect(screen.getByText('1–10 of 45')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /previous page/i })).toBeDisabled();

    await user.click(screen.getByRole('button', { name: /next page/i }));
    expect(onOffsetChange).toHaveBeenCalledWith(10);

    rerender(<Pagination total={45} limit={10} offset={40} onOffsetChange={onOffsetChange} />);
    expect(screen.getByRole('button', { name: /next page/i })).toBeDisabled();
  });
});

describe('ThemeToggle', () => {
  it('changes theme from the menu', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ThemeToggle />);
    await user.click(screen.getByRole('button', { name: /change theme/i }));
    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Dark' }));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});

describe('ConfirmDialog', () => {
  it('confirms and closes', async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    const onOpenChange = vi.fn();
    renderWithProviders(
      <ConfirmDialog
        open
        onOpenChange={onOpenChange}
        title="Delete?"
        description="This cannot be undone"
        onConfirm={onConfirm}
        destructive
      />,
    );
    expect(screen.getByText('Delete?')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});

import { screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { makeAnalytics, makeEmptyAnalytics } from '@/test/fixtures';
import { renderWithProviders } from '@/test/utils';

vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: vi.fn(),
    getMetrics: vi.fn(),
    getAnalytics: vi.fn(),
  },
}));

const { systemService } = await import('@/services/system.service');

import AnalyticsPage from './analytics-page';
import DashboardPage from './dashboard-page';

beforeEach(() => {
  localStorage.clear();
  vi.mocked(systemService.getAnalytics).mockResolvedValue(makeAnalytics());
});

describe('DashboardPage (H15)', () => {
  it('renders the repository summary from one request', async () => {
    renderWithProviders(<DashboardPage />);

    // Headline KPIs, straight from the backend summary.
    expect(await screen.findByText('9')).toBeInTheDocument();
    expect(screen.getByText('10m 20s')).toBeInTheDocument();
    expect(screen.getByText('Repository health')).toBeInTheDocument();
    // Exactly one aggregation request serves the whole page.
    expect(systemService.getAnalytics).toHaveBeenCalledTimes(1);
  });

  it('charts the violation breakdown as labelled bars', async () => {
    renderWithProviders(<DashboardPage />);

    const chart = await screen.findByLabelText('Confirmed violations by type');
    expect(within(chart).getByText('No helmet')).toBeInTheDocument();
    expect(within(chart).getByText('Wrong way')).toBeInTheDocument();
    expect(within(chart).getByText('Red-light jumping')).toBeInTheDocument();
  });

  it('shows review and evidence progress as measured fractions', async () => {
    renderWithProviders(<DashboardPage />);

    const reviewed = await screen.findByRole('progressbar', { name: 'Reviewed' });
    expect(reviewed).toHaveAttribute('aria-valuenow', '33');
    const evidence = screen.getByRole('progressbar', { name: 'Evidence rendered' });
    expect(evidence).toHaveAttribute('aria-valuenow', '67');
  });

  it('lists recent activity with wall-clock times, never media time', async () => {
    renderWithProviders(<DashboardPage />);

    const feed = await screen.findByRole('list', { name: 'Recent activity' });
    expect(within(feed).getByText('Run succeeded with 3 violation(s)')).toBeInTheDocument();
    expect(within(feed).getByText('Uploaded clip.mp4')).toBeInTheDocument();
    // The H15 trap: nothing on this page may render the 1970 media epoch.
    expect(feed.textContent).not.toContain('1970');
  });

  it('invites an upload when the repository is empty, without crashing', async () => {
    vi.mocked(systemService.getAnalytics).mockResolvedValue(makeEmptyAnalytics());
    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText('This repository is empty')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to videos' })).toHaveAttribute(
      'href',
      '/videos',
    );
    // No chart or feed is rendered for a repository with nothing in it.
    expect(screen.queryByRole('list', { name: 'Recent activity' })).not.toBeInTheDocument();
  });

  it('declares a breakdown the backend reported as incomplete', async () => {
    vi.mocked(systemService.getAnalytics).mockResolvedValue(
      makeAnalytics({
        violations: {
          events_total: 9,
          by_type: [{ violation_type: 'no_helmet', count: 5 }],
          counted_jobs: 1,
          uncounted_jobs: 2,
        },
      }),
    );
    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText(/2 run\(s\) recorded before per-type counts/)).toBeInTheDocument();
  });

  it('surfaces a load failure with a retry', async () => {
    vi.mocked(systemService.getAnalytics).mockRejectedValue(
      new ApiError('analytics unavailable', {
        kind: 'http',
        status: 503,
        type: 'internal_error',
      }),
    );
    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText('Could not load analytics')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });
});

describe('AnalyticsPage (H15)', () => {
  it('renders the detailed breakdown from the same summary', async () => {
    renderWithProviders(<AnalyticsPage />);

    expect(await screen.findByRole('heading', { name: 'Analytics' })).toBeInTheDocument();
    expect(await screen.findByText('Violations by type')).toBeInTheDocument();
    expect(
      await screen.findByLabelText('Confirmed violations by type'),
    ).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Videos calibrated' })).toBeInTheDocument();
  });

  it('shows the latest run engine metrics verbatim', async () => {
    renderWithProviders(<AnalyticsPage />);
    expect(await screen.findByText('Latest run')).toBeInTheDocument();
    // Detections come straight from EngineMetrics through the existing mapping.
    expect(await screen.findByText('120')).toBeInTheDocument();
  });

  it('explains an empty repository instead of charting nothing', async () => {
    vi.mocked(systemService.getAnalytics).mockResolvedValue(makeEmptyAnalytics());
    renderWithProviders(<AnalyticsPage />);

    expect(await screen.findByText('No analytics yet')).toBeInTheDocument();
    await waitFor(() =>
      expect(
        screen.queryByLabelText('Confirmed violations by type'),
      ).not.toBeInTheDocument(),
    );
  });

  it('reports absent engine metrics rather than zeroes', async () => {
    vi.mocked(systemService.getAnalytics).mockResolvedValue(
      makeAnalytics({ latest_run: null }),
    );
    renderWithProviders(<AnalyticsPage />);
    expect(
      await screen.findByText('No run has recorded engine metrics yet.'),
    ).toBeInTheDocument();
  });
});

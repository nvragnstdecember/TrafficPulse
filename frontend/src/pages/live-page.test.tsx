import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { type LiveReadiness } from '@/api/types';
import { makeConfirmedEvent } from '@/test/fixtures';
import {
  FakeSocket,
  makeResultMessage,
  makeSessionMessage,
  stubCamera,
  stubCameraDenied,
  stubCanvasCapture,
  stubVideoFrames,
  stubWebSocket,
} from '@/test/live-doubles';
import { renderWithProviders } from '@/test/utils';

vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: vi.fn(),
    getMetrics: vi.fn(),
    getAnalytics: vi.fn(),
    getPosture: vi.fn(),
  },
}));
// Only the two HTTP calls are replaced; `LiveConnection` stays real, so the page
// drives the actual socket client against the WebSocket double.
vi.mock('@/services/live.service', async (importOriginal) => {
  const actual = (await importOriginal()) as {
    liveService: Record<string, unknown>;
  };
  return {
    ...actual,
    liveService: {
      ...actual.liveService,
      getStatus: vi.fn(),
      listSessions: vi.fn(),
    },
  };
});

const { systemService } = await import('@/services/system.service');
const { liveService } = await import('@/services/live.service');

import LivePage from './live-page';

function readiness(overrides: Partial<LiveReadiness> = {}): LiveReadiness {
  return {
    ready: true,
    detail: 'Live monitoring can start.',
    active_sessions: 0,
    max_sessions: 2,
    inference_configured: true,
    drawing_backend_available: true,
    helmet_classifier_configured: true,
    ...overrides,
  };
}

beforeEach(() => {
  stubWebSocket();
  stubVideoFrames();
  stubCanvasCapture();
  vi.mocked(liveService.getStatus).mockResolvedValue(readiness());
  vi.mocked(systemService.getPosture).mockRejectedValue(new Error('no posture in this test'));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Walk the demo path: start camera, start monitoring, open the session. */
async function beginMonitoring(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: /start camera/i }));
  const start = await screen.findByRole('button', { name: /start monitoring/i });
  await waitFor(() => expect(start).toBeEnabled());
  await user.click(start);

  const socket = FakeSocket.latest;
  if (!socket) throw new Error('no socket was opened');
  act(() => {
    socket.connect();
    socket.deliver(makeSessionMessage());
  });
  return socket;
}

describe('LivePage', () => {
  it('offers the camera before it offers monitoring', async () => {
    stubCamera();
    renderWithProviders(<LivePage />);

    expect(screen.getByRole('heading', { name: 'Live camera' })).toBeInTheDocument();
    expect(screen.getByText('camera off')).toBeInTheDocument();
    // Monitoring is refused until there is a camera to monitor.
    expect(screen.getByRole('button', { name: /start monitoring/i })).toBeDisabled();
  });

  it('runs the whole demo path and shows the analysed frame', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    const socket = await beginMonitoring(user);
    act(() => {
      socket.deliver(makeResultMessage());
    });

    expect(await screen.findByText('monitoring')).toBeInTheDocument();
    const annotated = await screen.findByAltText(/latest analysed frame/i);
    expect(annotated).toHaveAttribute('src', 'data:image/jpeg;base64,YW5ub3RhdGVk');
    // The measured inference rate is published, not the camera's.
    expect(screen.getByText('1.5 fps')).toBeInTheDocument();
    expect(screen.getByText('640 ms')).toBeInTheDocument();
  });

  it('says in words that the camera rate is not the analysis rate', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    const socket = await beginMonitoring(user);
    act(() => {
      socket.deliver(makeResultMessage());
    });

    expect(
      await screen.findByText(/AI inference processes about 1\.5 of those frames per second/i),
    ).toBeInTheDocument();
  });

  it('names the violations this camera is not being checked for', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    await beginMonitoring(user);

    // The reason an empty event list is readable at all.
    expect(await screen.findByText(/Wrong way — not evaluated/i)).toBeInTheDocument();
    expect(screen.getByText(/declares no legal travel direction/i)).toBeInTheDocument();
  });

  it('states multi-rider driver attribution as unresolved', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    const socket = await beginMonitoring(user);
    act(() => {
      socket.deliver(
        makeResultMessage({
          motorcycles: [
            { motorcycle_track_id: 'trk-1', rider_count: 3, driver_resolved: false },
          ],
          riders: [
            {
              rider_track_id: 'trk-2',
              motorcycle_track_id: 'trk-1',
              rider_count: 3,
              driver_resolved: false,
              helmet_label: 'no_helmet',
              helmet_confidence: 0.8,
              helmet_gated: false,
            },
          ],
        }),
      );
    });

    expect(await screen.findByText(/3 riders — DRIVER UNRESOLVED/)).toBeInTheDocument();
  });

  it('lists confirmed events as they arrive', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    const socket = await beginMonitoring(user);
    expect(screen.getByText(/No violation confirmed yet/i)).toBeInTheDocument();

    act(() => {
      socket.deliver({
        type: 'events',
        events: [makeConfirmedEvent({ event_id: 'evt-7', violation_type: 'triple_riding' })],
      });
    });

    // Scoped to the row, because the violation's name also appears in the
    // "evaluated on this camera" summary beneath the feed.
    const row = await screen.findByRole('listitem');
    expect(within(row).getByText('Triple riding')).toBeInTheDocument();
    expect(within(row).getByText(/Track t-1/)).toBeInTheDocument();
    expect(within(row).getByText(/sustained/)).toBeInTheDocument();
  });

  it('shows a denied camera permission without breaking the page', async () => {
    stubCameraDenied();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    await user.click(await screen.findByRole('button', { name: /start camera/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/allow camera access/i);
    // The page is still usable and the camera can be retried.
    expect(screen.getByRole('button', { name: /start camera/i })).toBeEnabled();
  });

  it('reports a lost connection and leaves the camera running', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    const socket = await beginMonitoring(user);
    act(() => {
      socket.serverClose(1006);
    });

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /connection to the analysis server was lost/i,
    );
    expect(screen.getByRole('button', { name: /stop camera/i })).toBeInTheDocument();
  });

  it('refuses monitoring on a deployment that cannot run it, and explains why', async () => {
    stubCamera();
    vi.mocked(liveService.getStatus).mockResolvedValue(
      readiness({
        ready: false,
        inference_configured: false,
        detail: 'No inference backend is configured, so there is nothing to run.',
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    expect(await screen.findByText(/Live monitoring is unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/No inference backend is configured/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /start camera/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /start monitoring/i })).toBeDisabled(),
    );
  });

  it('states the session’s ephemerality and its window bound', async () => {
    stubCamera();
    const user = userEvent.setup();
    renderWithProviders(<LivePage />);

    await beginMonitoring(user);

    expect(await screen.findByText(/never written to disk/i)).toBeInTheDocument();
    expect(screen.getByText(/restarts every 600 processed frames/i)).toBeInTheDocument();
  });
});

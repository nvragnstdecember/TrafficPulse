import { QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { type ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createTestQueryClient } from '@/test/utils';

import { useEvent, useEventsList, useEvidence } from './use-events';
import { useMediaQuery } from './use-media-query';
import { useJob, useStartProcessing, useUploadVideo } from './use-videos';

const list = vi.fn((..._args: unknown[]) =>
  Promise.resolve({ items: [], total: 0, limit: 25, offset: 0 }),
);
const getEvent = vi.fn((..._args: unknown[]) => Promise.resolve({ event_id: 'e1' }));
const getEvidence = vi.fn((..._args: unknown[]) => Promise.resolve({ event_id: 'e1' }));
vi.mock('@/services/events.service', () => ({
  eventsService: {
    list: (...args: unknown[]) => list(...args),
    get: (...args: unknown[]) => getEvent(...args),
    getEvidence: (...args: unknown[]) => getEvidence(...args),
  },
}));

const getJob = vi.fn((..._args: unknown[]) =>
  Promise.resolve({ job_id: 'j1', status: 'succeeded' }),
);
const upload = vi.fn((..._args: unknown[]) => Promise.resolve({ video_id: 'v1' }));
const startProcessing = vi.fn((..._args: unknown[]) => Promise.resolve({ job_id: 'j1' }));
vi.mock('@/services/videos.service', () => ({
  videosService: {
    getJob: (...args: unknown[]) => getJob(...args),
    upload: (...args: unknown[]) => upload(...args),
    startProcessing: (...args: unknown[]) => startProcessing(...args),
  },
}));

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={createTestQueryClient()}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('event hooks', () => {
  it('fetches an event list', async () => {
    const { result } = renderHook(() => useEventsList({ limit: 25, offset: 0 }), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.total).toBe(0);
  });

  it('is disabled until an id is provided', () => {
    const { result } = renderHook(() => useEvent(undefined), { wrapper });
    expect(result.current.fetchStatus).toBe('idle');
    expect(getEvent).not.toHaveBeenCalled();
  });

  it('fetches an event and its evidence when enabled', async () => {
    const { result: event } = renderHook(() => useEvent('e1'), { wrapper });
    const { result: evidence } = renderHook(() => useEvidence('e1'), { wrapper });
    await waitFor(() => expect(event.current.isSuccess).toBe(true));
    await waitFor(() => expect(evidence.current.isSuccess).toBe(true));
    expect(getEvent).toHaveBeenCalledWith('e1', expect.anything());
    expect(getEvidence).toHaveBeenCalledWith('e1', expect.anything());
  });
});

describe('video hooks', () => {
  it('fetches a job by id', async () => {
    const { result } = renderHook(() => useJob('j1'), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.status).toBe('succeeded');
  });

  it('uploads a video via mutation', async () => {
    const { result } = renderHook(() => useUploadVideo(), { wrapper });
    result.current.mutate({ file: new File(['x'], 'c.mp4') });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(upload).toHaveBeenCalled();
  });

  it('starts processing via mutation', async () => {
    const { result } = renderHook(() => useStartProcessing(), { wrapper });
    result.current.mutate({ videoId: 'v1' });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(startProcessing).toHaveBeenCalled();
  });
});

describe('useMediaQuery', () => {
  it('reflects the media query match', () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn((query: string) => ({
        matches: true,
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
        onchange: null,
      })),
    );
    const { result } = renderHook(() => useMediaQuery('(min-width: 768px)'));
    expect(result.current).toBe(true);
  });
});

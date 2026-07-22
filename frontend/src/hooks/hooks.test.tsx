import { QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { type ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createTestQueryClient } from '@/test/utils';

import { useDebouncedValue } from './use-debounced-value';
import { useHealth, useMetrics } from './use-system';

vi.mock('@/services/system.service', () => ({
  systemService: {
    getHealth: vi.fn(async () => ({ status: 'ok', version: '0.1.0', engine: 'ready' })),
    getMetrics: vi.fn(async () => ({
      jobs_total: 2,
      jobs_pending: 0,
      jobs_running: 0,
      jobs_succeeded: 2,
      jobs_failed: 0,
      events_total: 3,
      latest: null,
    })),
  },
}));

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={createTestQueryClient()}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('useHealth / useMetrics', () => {
  it('fetches health via the service', async () => {
    const { result } = renderHook(() => useHealth(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ status: 'ok', version: '0.1.0', engine: 'ready' });
  });

  it('fetches aggregate metrics via the service', async () => {
    const { result } = renderHook(() => useMetrics(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.events_total).toBe(3);
  });
});

describe('useDebouncedValue', () => {
  it('delays updates until the timeout elapses', () => {
    vi.useFakeTimers();
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, 300), {
      initialProps: { value: 'a' },
    });
    expect(result.current).toBe('a');

    rerender({ value: 'b' });
    expect(result.current).toBe('a'); // not yet

    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current).toBe('b');
  });
});

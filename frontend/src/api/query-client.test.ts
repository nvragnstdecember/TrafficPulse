import { describe, expect, it, vi } from 'vitest';

import { ApiError } from './errors';
import { createQueryClient } from './query-client';

describe('createQueryClient', () => {
  it('does not retry non-retryable ApiErrors (4xx)', () => {
    const client = createQueryClient();
    const retry = client.getDefaultOptions().queries?.retry as (n: number, e: unknown) => boolean;
    const notFound = new ApiError('nope', { kind: 'http', status: 404 });
    expect(retry(0, notFound)).toBe(false);
  });

  it('retries retryable errors up to the limit', () => {
    const client = createQueryClient();
    const retry = client.getDefaultOptions().queries?.retry as (n: number, e: unknown) => boolean;
    const serverError = new ApiError('boom', { kind: 'http', status: 503 });
    expect(retry(0, serverError)).toBe(true);
    expect(retry(1, serverError)).toBe(true);
    expect(retry(2, serverError)).toBe(false);
  });

  it('invokes onError for real failures but not for cancellations', () => {
    const onError = vi.fn();
    const client = createQueryClient({ onError });
    const cache = client.getQueryCache();

    cache.config.onError?.(new ApiError('canceled', { kind: 'canceled' }), {} as never);
    expect(onError).not.toHaveBeenCalled();

    cache.config.onError?.(new ApiError('boom', { kind: 'network' }), {} as never);
    expect(onError).toHaveBeenCalledTimes(1);
  });
});

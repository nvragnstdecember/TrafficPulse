import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiClient } from './client';
import { ApiError } from './errors';

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });
}

/** A fetch that never resolves until its signal aborts (for timeout/cancel tests). */
function hangingFetch() {
  return vi.fn(
    (_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'));
          return;
        }
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      }),
  );
}

function makeClient(overrides?: Partial<ConstructorParameters<typeof ApiClient>[0]>) {
  return new ApiClient({ baseUrl: '', timeoutMs: 1000, ...overrides });
}

afterEach(() => {
  vi.useRealTimers();
});

describe('ApiClient', () => {
  it('returns parsed JSON and sends an Accept header', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      jsonResponse({ ok: true }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await makeClient().get<{ ok: boolean }>('/api/health');

    expect(result).toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/health');
    expect(new Headers(init?.headers).get('Accept')).toBe('application/json');
  });

  it('appends query params and omits null/undefined', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => jsonResponse({}));
    vi.stubGlobal('fetch', fetchMock);

    await makeClient().get('/api/events', {
      query: { limit: 10, offset: 0, video_id: undefined, sort: null },
    });

    expect(fetchMock.mock.calls[0][0]).toBe('/api/events?limit=10&offset=0');
  });

  it('serializes a JSON body and injects the auth token', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => jsonResponse({ id: 'x' }));
    vi.stubGlobal('fetch', fetchMock);

    const client = makeClient({ getAuthToken: () => 'secret-token' });
    await client.post('/api/process', { video_id: 'v1' });

    const init = fetchMock.mock.calls[0][1];
    const headers = new Headers(init?.headers);
    expect(init?.method).toBe('POST');
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(headers.get('Authorization')).toBe('Bearer secret-token');
    expect(init?.body).toBe(JSON.stringify({ video_id: 'v1' }));
  });

  it('maps an error envelope to a typed ApiError', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ error: { type: 'video_not_found', message: 'no video' } }, { status: 404 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const error = await makeClient()
      .get('/api/videos/x')
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      kind: 'http',
      status: 404,
      type: 'video_not_found',
      message: 'no video',
    });
    expect((error as ApiError).isRetryable).toBe(false);
  });

  it('falls back to a status message when there is no envelope', async () => {
    const fetchMock = vi.fn(async () => new Response('boom', { status: 500 }));
    vi.stubGlobal('fetch', fetchMock);

    const error = (await makeClient()
      .get('/api/x')
      .catch((e: unknown) => e)) as ApiError;

    expect(error.kind).toBe('http');
    expect(error.status).toBe(500);
    expect(error.message).toContain('500');
    expect(error.isRetryable).toBe(true);
  });

  it('wraps a network failure', async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    });
    vi.stubGlobal('fetch', fetchMock);

    const error = (await makeClient()
      .get('/api/x')
      .catch((e: unknown) => e)) as ApiError;

    expect(error.kind).toBe('network');
    expect(error.isRetryable).toBe(true);
  });

  it('returns undefined for a 204 response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 204 })),
    );
    await expect(makeClient().delete('/api/x')).resolves.toBeUndefined();
  });

  it('returns text for a non-JSON response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('plain', { status: 200 })),
    );
    await expect(makeClient().get('/api/x')).resolves.toBe('plain');
  });

  it('times out after the configured deadline', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', hangingFetch());

    const promise = makeClient({ timeoutMs: 1000 })
      .get('/api/slow')
      .catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1000);
    const error = (await promise) as ApiError;

    expect(error.kind).toBe('timeout');
    expect(error.isRetryable).toBe(true);
  });

  it('supports external cancellation', async () => {
    vi.stubGlobal('fetch', hangingFetch());
    const controller = new AbortController();

    const promise = makeClient()
      .get('/api/slow', { signal: controller.signal })
      .catch((e: unknown) => e);
    controller.abort();
    const error = (await promise) as ApiError;

    expect(error.kind).toBe('canceled');
    expect(error.isCanceled).toBe(true);
    expect(error.isRetryable).toBe(false);
  });

  it('uploads FormData without forcing a JSON content-type', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      jsonResponse({ video_id: 'v1' }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const form = new FormData();
    form.append('file', new Blob(['data']), 'clip.mp4');
    await makeClient().upload('/api/video/upload', form);

    const init = fetchMock.mock.calls[0][1];
    expect(init?.body).toBeInstanceOf(FormData);
    expect(new Headers(init?.headers).get('Content-Type')).toBeNull();
  });
});

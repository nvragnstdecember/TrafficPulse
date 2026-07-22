import { env } from '@/lib/env';

import { ApiError, parseErrorBody } from './errors';

/** Supplies a bearer token per request; return null/undefined when unauthenticated. */
export type AuthTokenProvider = () => string | null | undefined;

export type QueryValue = string | number | boolean | null | undefined;

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  /** JSON-serialized (object) or sent as-is (FormData). */
  body?: unknown;
  headers?: Record<string, string>;
  /** Query params; null/undefined entries are omitted. */
  query?: Record<string, QueryValue>;
  /** Caller-owned cancellation, combined with the per-request timeout. */
  signal?: AbortSignal;
  /** Overrides the client default timeout for this request. */
  timeoutMs?: number;
}

export interface ApiClientConfig {
  baseUrl: string;
  timeoutMs: number;
  getAuthToken?: AuthTokenProvider;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  /** 0..1 fraction uploaded. */
  ratio: number;
}

export interface UploadRequestOptions {
  onProgress?: (progress: UploadProgress) => void;
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

function buildUrl(baseUrl: string, path: string, query?: Record<string, QueryValue>): string {
  const url = `${baseUrl}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== null && value !== undefined) params.append(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

/**
 * The typed HTTP client for the H7A API (H7B).
 *
 * One place owns base-URL joining, JSON (de)serialization, bearer-token
 * injection, per-request timeouts, external cancellation, and the translation of
 * every failure into a typed {@link ApiError}. Pages never call it directly —
 * services wrap it, hooks call services — so no endpoint logic leaks into UI.
 */
export class ApiClient {
  private baseUrl: string;
  private timeoutMs: number;
  private getAuthToken?: AuthTokenProvider;

  constructor(config: ApiClientConfig) {
    this.baseUrl = config.baseUrl;
    this.timeoutMs = config.timeoutMs;
    this.getAuthToken = config.getAuthToken;
  }

  /** Register (or clear) the auth-token provider — keeps the client auth-ready. */
  setAuthTokenProvider(provider: AuthTokenProvider | undefined): void {
    this.getAuthToken = provider;
  }

  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const controller = new AbortController();
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    const external = options.signal;
    const onExternalAbort = () => controller.abort();
    if (external) {
      if (external.aborted) controller.abort();
      else external.addEventListener('abort', onExternalAbort, { once: true });
    }

    try {
      const headers = new Headers(options.headers);
      if (!headers.has('Accept')) headers.set('Accept', 'application/json');

      const token = this.getAuthToken?.();
      if (token) headers.set('Authorization', `Bearer ${token}`);

      let body: BodyInit | undefined;
      if (options.body instanceof FormData) {
        body = options.body;
      } else if (options.body !== undefined) {
        headers.set('Content-Type', 'application/json');
        body = JSON.stringify(options.body);
      }

      let response: Response;
      try {
        response = await fetch(buildUrl(this.baseUrl, path, options.query), {
          method: options.method ?? 'GET',
          headers,
          body,
          signal: controller.signal,
        });
      } catch (cause) {
        if (controller.signal.aborted) {
          throw timedOut
            ? new ApiError(`Request timed out after ${timeoutMs}ms`, { kind: 'timeout', cause })
            : new ApiError('Request was canceled', { kind: 'canceled', cause });
        }
        throw new ApiError('Network request failed', { kind: 'network', cause });
      }

      return await this.parseResponse<T>(response);
    } finally {
      clearTimeout(timer);
      external?.removeEventListener('abort', onExternalAbort);
    }
  }

  private async parseResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') ?? '';
    const isJson = contentType.includes('application/json');

    let payload: unknown;
    if (response.status !== 204) {
      try {
        payload = isJson ? await response.json() : await response.text();
      } catch (cause) {
        if (response.ok) {
          throw new ApiError('Failed to parse the server response', { kind: 'parse', cause });
        }
        payload = undefined;
      }
    }

    if (!response.ok) {
      const envelope = parseErrorBody(payload);
      throw new ApiError(envelope?.message ?? `Request failed with status ${response.status}`, {
        kind: 'http',
        status: response.status,
        type: envelope?.type ?? null,
      });
    }

    return payload as T;
  }

  get<T>(path: string, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
    return this.request<T>(path, { ...options, method: 'GET' });
  }

  post<T>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method'>): Promise<T> {
    return this.request<T>(path, { ...options, method: 'POST', body });
  }

  put<T>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method'>): Promise<T> {
    return this.request<T>(path, { ...options, method: 'PUT', body });
  }

  patch<T>(path: string, body?: unknown, options?: Omit<RequestOptions, 'method'>): Promise<T> {
    return this.request<T>(path, { ...options, method: 'PATCH', body });
  }

  delete<T>(path: string, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
    return this.request<T>(path, { ...options, method: 'DELETE' });
  }

  /** Multipart upload (FormData); the browser sets the multipart boundary. */
  upload<T>(
    path: string,
    formData: FormData,
    options?: Omit<RequestOptions, 'method' | 'body'>,
  ): Promise<T> {
    return this.request<T>(path, { ...options, method: 'POST', body: formData });
  }

  /**
   * Multipart upload with real progress reporting (XHR-backed).
   *
   * `fetch` cannot report upload progress, so this path uses `XMLHttpRequest`
   * to drive `onProgress`, while keeping the same timeout, cancellation, and
   * typed-error contract as {@link request}. Still a client-layer concern — UI
   * reaches it only through services/hooks.
   */
  uploadWithProgress<T>(
    path: string,
    formData: FormData,
    options: UploadRequestOptions = {},
  ): Promise<T> {
    const { onProgress, signal, headers } = options;
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;

    return new Promise<T>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new ApiError('Request was canceled', { kind: 'canceled' }));
        return;
      }

      const xhr = new XMLHttpRequest();
      xhr.open('POST', buildUrl(this.baseUrl, path));
      xhr.responseType = 'text';
      xhr.timeout = timeoutMs;

      const merged = new Headers(headers);
      if (!merged.has('Accept')) merged.set('Accept', 'application/json');
      const token = this.getAuthToken?.();
      if (token) merged.set('Authorization', `Bearer ${token}`);
      merged.forEach((value, key) => xhr.setRequestHeader(key, value));

      const onAbort = () => xhr.abort();
      const cleanup = () => signal?.removeEventListener('abort', onAbort);
      signal?.addEventListener('abort', onAbort, { once: true });

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress({
            loaded: event.loaded,
            total: event.total,
            ratio: event.total > 0 ? event.loaded / event.total : 0,
          });
        }
      };

      xhr.onload = () => {
        cleanup();
        const contentType = xhr.getResponseHeader('content-type') ?? '';
        const isJson = contentType.includes('application/json');
        let payload: unknown;
        try {
          payload = xhr.responseText && isJson ? JSON.parse(xhr.responseText) : xhr.responseText;
        } catch (cause) {
          if (xhr.status >= 200 && xhr.status < 300) {
            reject(new ApiError('Failed to parse the server response', { kind: 'parse', cause }));
            return;
          }
          payload = undefined;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve((payload || undefined) as T);
          return;
        }
        const envelope = parseErrorBody(payload);
        reject(
          new ApiError(envelope?.message ?? `Request failed with status ${xhr.status}`, {
            kind: 'http',
            status: xhr.status,
            type: envelope?.type ?? null,
          }),
        );
      };
      xhr.onerror = () => {
        cleanup();
        reject(new ApiError('Network request failed', { kind: 'network' }));
      };
      xhr.ontimeout = () => {
        cleanup();
        reject(new ApiError(`Request timed out after ${timeoutMs}ms`, { kind: 'timeout' }));
      };
      xhr.onabort = () => {
        cleanup();
        reject(new ApiError('Request was canceled', { kind: 'canceled' }));
      };

      xhr.send(formData);
    });
  }
}

/** The shared, env-configured client instance the services use. */
export const apiClient = new ApiClient({
  baseUrl: env.apiBaseUrl,
  timeoutMs: env.apiTimeoutMs,
});

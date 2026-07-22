import { type ApiErrorBody } from './types';

/**
 * A normalized API failure. Every non-2xx response, network failure, timeout, or
 * cancellation surfaces as an `ApiError`, so callers depend on one error type
 * with a stable `kind` rather than raw fetch/DOM exceptions.
 */
export type ApiErrorKind = 'http' | 'network' | 'timeout' | 'canceled' | 'parse';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  /** HTTP status when `kind === 'http'`, else 0. */
  readonly status: number;
  /** The server's stable error slug (from the H7A envelope) when available. */
  readonly type: string | null;
  /** Existing video id carried by a `duplicate_video` conflict, else null (H7D). */
  readonly videoId: string | null;

  constructor(
    message: string,
    options: {
      kind: ApiErrorKind;
      status?: number;
      type?: string | null;
      videoId?: string | null;
      cause?: unknown;
    },
  ) {
    super(message, { cause: options.cause });
    this.name = 'ApiError';
    this.kind = options.kind;
    this.status = options.status ?? 0;
    this.type = options.type ?? null;
    this.videoId = options.videoId ?? null;
  }

  /** True for a duplicate-video conflict (409), which carries the existing id. */
  get isDuplicate(): boolean {
    return this.type === 'duplicate_video';
  }

  /** True for transient failures worth a retry (network/timeout/5xx). */
  get isRetryable(): boolean {
    if (this.kind === 'network' || this.kind === 'timeout') return true;
    return this.kind === 'http' && this.status >= 500;
  }

  get isCanceled(): boolean {
    return this.kind === 'canceled';
  }
}

/** Best-effort extraction of the H7A error envelope from a parsed body. */
export function parseErrorBody(body: unknown): ApiErrorBody['error'] | null {
  if (
    typeof body === 'object' &&
    body !== null &&
    'error' in body &&
    typeof (body as ApiErrorBody).error === 'object' &&
    (body as ApiErrorBody).error !== null
  ) {
    const { type, message, video_id } = (body as ApiErrorBody).error;
    if (typeof type === 'string' && typeof message === 'string') {
      return typeof video_id === 'string' ? { type, message, video_id } : { type, message };
    }
  }
  return null;
}

/** A human-facing message for any thrown value (used by toasts/banners). */
export function toErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

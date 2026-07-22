/**
 * Runtime environment configuration (H7B).
 *
 * The single place `import.meta.env` is read. The API base URL is configurable
 * via `VITE_API_BASE_URL`; it defaults to an empty string so requests hit the
 * same origin (the dev server proxies `/api` to the backend, and a production
 * deployment serving the SPA behind the API needs no override). Nothing is
 * hardcoded to a machine or OS path.
 */
export interface AppEnv {
  /** Base URL prepended to every API path. Empty = same-origin. */
  apiBaseUrl: string;
  /** Default per-request timeout, in milliseconds. */
  apiTimeoutMs: number;
  /** Maximum accepted upload size, in bytes. */
  maxUploadBytes: number;
  /** Accepted video container extensions (lower-case, leading dot). */
  acceptedVideoFormats: string[];
  /** True in production builds. */
  isProduction: boolean;
}

const DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024; // 512 MiB (mirrors the H7A default)
const DEFAULT_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v'];

function readNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function readFormats(value: string | undefined, fallback: string[]): string[] {
  if (!value) return fallback;
  const parsed = value
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean)
    .map((entry) => (entry.startsWith('.') ? entry : `.${entry}`));
  return parsed.length > 0 ? parsed : fallback;
}

export const env: AppEnv = {
  apiBaseUrl: (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, ''),
  apiTimeoutMs: readNumber(import.meta.env.VITE_API_TIMEOUT_MS, 30_000),
  maxUploadBytes: readNumber(import.meta.env.VITE_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
  acceptedVideoFormats: readFormats(
    import.meta.env.VITE_ACCEPTED_VIDEO_FORMATS,
    DEFAULT_VIDEO_FORMATS,
  ),
  isProduction: import.meta.env.PROD,
};

/**
 * Pure display formatters (H7B).
 *
 * Deterministic, framework-free helpers used across components so number/size/
 * time presentation is consistent and never re-implemented per feature.
 */

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

/** Human-readable byte size, e.g. `formatBytes(1536)` -> "1.5 KB". */
export function formatBytes(bytes: number, fractionDigits = 1): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), BYTE_UNITS.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(exponent === 0 ? 0 : fractionDigits)} ${BYTE_UNITS[exponent]}`;
}

/** Compact duration from seconds, e.g. `formatDuration(95)` -> "1m 35s". */
export function formatDuration(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || !Number.isFinite(totalSeconds) || totalSeconds < 0) return '—';
  const seconds = Math.floor(totalSeconds % 60);
  const minutes = Math.floor((totalSeconds / 60) % 60);
  const hours = Math.floor(totalSeconds / 3600);
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

/** Locale date-time, tolerant of null/invalid input (returns an em dash). */
export function formatDateTime(value: string | number | Date | null | undefined): string {
  if (value == null) return '—';
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

/** Percentage from a 0..1 fraction, or an em dash when unknown. */
export function formatPercent(fraction: number | null | undefined, fractionDigits = 0): string {
  if (fraction == null || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(fractionDigits)}%`;
}

/** Grouped integer/number, e.g. `formatNumber(12345)` -> "12,345". */
export function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat().format(value);
}

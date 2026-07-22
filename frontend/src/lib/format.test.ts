import { describe, expect, it } from 'vitest';

import { formatBytes, formatDateTime, formatDuration, formatNumber, formatPercent } from './format';

describe('formatBytes', () => {
  it('handles zero and negatives', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(-5)).toBe('0 B');
  });

  it('formats across units', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB');
    expect(formatBytes(1024 ** 3)).toBe('1.0 GB');
  });
});

describe('formatDuration', () => {
  it('returns an em dash for invalid input', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(undefined)).toBe('—');
    expect(formatDuration(-1)).toBe('—');
  });

  it('formats seconds, minutes, and hours', () => {
    expect(formatDuration(45)).toBe('45s');
    expect(formatDuration(95)).toBe('1m 35s');
    expect(formatDuration(3720)).toBe('1h 2m');
  });
});

describe('formatDateTime', () => {
  it('returns an em dash for invalid input', () => {
    expect(formatDateTime(null)).toBe('—');
    expect(formatDateTime('not-a-date')).toBe('—');
  });

  it('formats a valid date', () => {
    expect(formatDateTime('2026-01-02T03:04:05Z')).not.toBe('—');
  });
});

describe('formatPercent', () => {
  it('formats fractions and handles nullish', () => {
    expect(formatPercent(0.5)).toBe('50%');
    expect(formatPercent(0.1234, 1)).toBe('12.3%');
    expect(formatPercent(null)).toBe('—');
  });
});

describe('formatNumber', () => {
  it('groups digits and handles nullish', () => {
    expect(formatNumber(12345)).toBe('12,345');
    expect(formatNumber(null)).toBe('—');
  });
});

import { describe, expect, it } from 'vitest';

import { makeJob } from '@/test/fixtures';

import { overlayVideoSource } from './overlay-source';

describe('overlayVideoSource', () => {
  it('returns null while the job is not yet succeeded', () => {
    expect(overlayVideoSource(makeJob({ status: 'running', overlay_available: false }))).toBeNull();
    expect(overlayVideoSource(undefined)).toBeNull();
    expect(overlayVideoSource(null)).toBeNull();
  });

  it('returns null when succeeded but no overlay was produced', () => {
    expect(
      overlayVideoSource(makeJob({ status: 'succeeded', overlay_available: false })),
    ).toBeNull();
  });

  it('returns the overlay endpoint URL once succeeded and available', () => {
    const url = overlayVideoSource(
      makeJob({ job_id: 'job-xyz', status: 'succeeded', overlay_available: true }),
    );
    expect(url).toBe('/api/process/job-xyz/overlay');
  });

  it('encodes the job id into the path', () => {
    const url = overlayVideoSource(
      makeJob({ job_id: 'job/../x', status: 'succeeded', overlay_available: true }),
    );
    expect(url).toBe('/api/process/job%2F..%2Fx/overlay');
  });
});

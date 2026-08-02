import { describe, expect, it } from 'vitest';

import { makeJob } from '@/test/fixtures';

import { videoMediaSource, workspaceVideoSource } from './video-source';

describe('videoMediaSource', () => {
  it('addresses the stored upload', () => {
    expect(videoMediaSource('vid-1')).toBe('/api/videos/vid-1/media');
  });

  it('encodes the id and refuses a missing one', () => {
    expect(videoMediaSource('vid/odd id')).toBe('/api/videos/vid%2Fodd%20id/media');
    expect(videoMediaSource(null)).toBeNull();
    expect(videoMediaSource(undefined)).toBeNull();
  });
});

describe('workspaceVideoSource', () => {
  const overlaid = makeJob({ status: 'succeeded', overlay_available: true, overlay_status: 'ready' });

  it('prefers the annotated overlay over every other source', () => {
    // Showing the raw clip when boxes exist hides the reasoning under review.
    expect(
      workspaceVideoSource({ job: overlaid, objectUrl: 'blob:local', videoId: 'vid-1' }),
    ).toBe('/api/process/job-1/overlay');
  });

  it('prefers a local file over a download when there is no overlay', () => {
    expect(
      workspaceVideoSource({ job: makeJob(), objectUrl: 'blob:local', videoId: 'vid-1' }),
    ).toBe('blob:local');
  });

  it('falls back to the stored video for a session that never held the file', () => {
    // The library case: nothing was picked in this browser, so without this the
    // player would have no source at all.
    expect(workspaceVideoSource({ job: makeJob(), objectUrl: null, videoId: 'vid-1' })).toBe(
      '/api/videos/vid-1/media',
    );
  });

  it('reports no source when the stored file is gone and nothing local exists', () => {
    expect(
      workspaceVideoSource({
        job: makeJob(),
        objectUrl: null,
        videoId: 'vid-1',
        mediaAvailable: false,
      }),
    ).toBeNull();
    expect(workspaceVideoSource({ job: undefined, objectUrl: null, videoId: null })).toBeNull();
  });

  it('still plays the overlay for a video whose source file was deleted', () => {
    expect(
      workspaceVideoSource({
        job: overlaid,
        objectUrl: null,
        videoId: 'vid-1',
        mediaAvailable: false,
      }),
    ).toBe('/api/process/job-1/overlay');
  });
});

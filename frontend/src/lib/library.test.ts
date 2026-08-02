import { describe, expect, it } from 'vitest';

import { makeVideoSummary } from '@/test/fixtures';

import {
  canOpen,
  filterVideos,
  hasAnalysis,
  libraryStatusLabel,
  libraryStatusTone,
  reviewProgressLabel,
} from './library';

describe('libraryStatusLabel / libraryStatusTone', () => {
  it('distinguishes never-processed from every job state', () => {
    // Null is not "pending": nothing was ever submitted for this video, and telling
    // an analyst it is queued would have them waiting for a job that does not exist.
    expect(libraryStatusLabel(makeVideoSummary({ status: null }))).toBe('Not processed');
    expect(libraryStatusLabel(makeVideoSummary({ status: 'pending' }))).toBe('Queued');
    expect(libraryStatusLabel(makeVideoSummary({ status: 'running' }))).toBe('Processing');
    expect(libraryStatusLabel(makeVideoSummary({ status: 'succeeded' }))).toBe('Analysed');
    expect(libraryStatusLabel(makeVideoSummary({ status: 'failed' }))).toBe('Failed');
    expect(libraryStatusLabel(makeVideoSummary({ status: 'cancelled' }))).toBe('Cancelled');
  });

  it('tones a failure apart from a cancellation', () => {
    expect(libraryStatusTone(makeVideoSummary({ status: 'failed' }))).toBe('error');
    expect(libraryStatusTone(makeVideoSummary({ status: 'cancelled' }))).toBe('neutral');
    expect(libraryStatusTone(makeVideoSummary({ status: 'succeeded' }))).toBe('success');
    expect(libraryStatusTone(makeVideoSummary({ status: null }))).toBe('neutral');
  });
});

describe('reviewProgressLabel', () => {
  it('says nothing for a video with no events to review', () => {
    expect(reviewProgressLabel(makeVideoSummary({ event_count: 0 }))).toBeNull();
  });

  it('reports untouched, partial, and complete review progress', () => {
    expect(reviewProgressLabel(makeVideoSummary({ event_count: 4, events_reviewed: 0 }))).toBe(
      '4 to review',
    );
    expect(reviewProgressLabel(makeVideoSummary({ event_count: 4, events_reviewed: 1 }))).toBe(
      '1 of 4 reviewed',
    );
    expect(reviewProgressLabel(makeVideoSummary({ event_count: 4, events_reviewed: 4 }))).toBe(
      'All reviewed',
    );
  });
});

describe('hasAnalysis / canOpen', () => {
  it('treats only a succeeded run as an analysis to reopen', () => {
    expect(hasAnalysis(makeVideoSummary({ status: 'succeeded' }))).toBe(true);
    expect(hasAnalysis(makeVideoSummary({ status: 'failed' }))).toBe(false);
    expect(hasAnalysis(makeVideoSummary({ status: null }))).toBe(false);
  });

  it('refuses to open a video with neither stored media nor an overlay', () => {
    // Its events and review history are still valid, so the row stays — but opening
    // it would strand the analyst on an empty player.
    const gone = makeVideoSummary({ media_available: false, overlay_available: false });
    expect(canOpen(gone)).toBe(false);
    expect(canOpen(makeVideoSummary({ media_available: false, overlay_available: true }))).toBe(
      true,
    );
    expect(canOpen(makeVideoSummary())).toBe(true);
  });
});

describe('filterVideos', () => {
  const videos = [
    makeVideoSummary({ video_id: 'vid-aaa', filename: 'Junction North.mp4' }),
    makeVideoSummary({ video_id: 'vid-bbb', filename: 'highway.mp4' }),
  ];

  it('returns everything for an empty or whitespace query', () => {
    expect(filterVideos(videos, '')).toHaveLength(2);
    expect(filterVideos(videos, '   ')).toHaveLength(2);
  });

  it('matches filename and id case-insensitively', () => {
    expect(filterVideos(videos, 'north').map((v) => v.video_id)).toEqual(['vid-aaa']);
    expect(filterVideos(videos, 'VID-BBB').map((v) => v.video_id)).toEqual(['vid-bbb']);
    expect(filterVideos(videos, 'nothing')).toEqual([]);
  });
});

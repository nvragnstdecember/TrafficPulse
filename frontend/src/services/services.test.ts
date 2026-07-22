import { beforeEach, describe, expect, it, vi } from 'vitest';

import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';

import { eventsService } from './events.service';
import { systemService } from './system.service';
import { videosService } from './videos.service';

vi.mock('@/api/client', () => ({
  apiClient: {
    get: vi.fn(async () => ({})),
    post: vi.fn(async () => ({})),
    upload: vi.fn(async () => ({})),
  },
}));

const mockedClient = vi.mocked(apiClient);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('systemService', () => {
  it('requests health and metrics', async () => {
    await systemService.getHealth();
    await systemService.getMetrics();
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.health, { signal: undefined });
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.metrics, { signal: undefined });
  });
});

describe('videosService', () => {
  it('uploads a file as multipart form data', async () => {
    const file = new File(['x'], 'clip.mp4', { type: 'video/mp4' });
    await videosService.upload({ file });
    const [path, form] = mockedClient.upload.mock.calls[0];
    expect(path).toBe(endpoints.videoUpload);
    expect(form).toBeInstanceOf(FormData);
    expect((form as FormData).get('file')).toBe(file);
  });

  it('starts processing with the video id and null rules by default', async () => {
    await videosService.startProcessing({ videoId: 'v1' });
    expect(mockedClient.post).toHaveBeenCalledWith(
      endpoints.process,
      { video_id: 'v1', rules: null },
      { signal: undefined },
    );
  });

  it('fetches a job by id', async () => {
    await videosService.getJob('job-1');
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.job('job-1'), { signal: undefined });
  });
});

describe('eventsService', () => {
  it('lists events with query params', async () => {
    await eventsService.list({ videoId: 'v1', limit: 10, offset: 0, sort: '-trigger_at' });
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.events, {
      query: { video_id: 'v1', limit: 10, offset: 0, sort: '-trigger_at' },
      signal: undefined,
    });
  });

  it('fetches event detail and evidence', async () => {
    await eventsService.get('e1');
    await eventsService.getEvidence('e1');
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.event('e1'), { signal: undefined });
    expect(mockedClient.get).toHaveBeenCalledWith(endpoints.evidence('e1'), { signal: undefined });
  });
});

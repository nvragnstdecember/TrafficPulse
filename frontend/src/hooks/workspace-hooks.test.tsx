import { QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { type ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '@/api/errors';
import { type UploadVideoInput } from '@/services/videos.service';
import { useProcessingStore } from '@/store/processing-store';
import { useSelectionStore } from '@/store/selection-store';
import { useUploadStore } from '@/store/upload-store';
import {
  makeConfirmedEvent,
  makeEventSummary,
  makeEvidence,
  makeFile,
  makeJob,
  makeVideo,
  mediaSeconds,
} from '@/test/fixtures';
import { createTestQueryClient } from '@/test/utils';

import { useProcessing } from './use-processing';
import { useWorkspaceEvents } from './use-workspace-events';

vi.mock('@/services/videos.service', () => ({
  videosService: {
    upload: vi.fn(),
    startProcessing: vi.fn(),
    getJob: vi.fn(),
  },
}));

vi.mock('@/services/events.service', () => ({
  eventsService: {
    list: vi.fn(),
    get: vi.fn(),
    getEvidence: vi.fn(),
  },
}));

const { videosService } = await import('@/services/videos.service');
const { eventsService } = await import('@/services/events.service');

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={createTestQueryClient()}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useUploadStore.getState().reset();
    useProcessingStore.getState().reset();
    useSelectionStore.getState().clearSelection();
  });
  vi.mocked(videosService.upload).mockResolvedValue(makeVideo());
  vi.mocked(videosService.startProcessing).mockResolvedValue({
    job_id: 'job-1',
    video_id: 'vid-1',
    status: 'pending',
  });
  vi.mocked(videosService.getJob).mockResolvedValue(makeJob({ status: 'succeeded', progress: 1 }));
  vi.mocked(eventsService.list).mockResolvedValue({
    items: [makeEventSummary()],
    total: 1,
    limit: 200,
    offset: 0,
  });
  vi.mocked(eventsService.get).mockResolvedValue(makeConfirmedEvent());
  vi.mocked(eventsService.getEvidence).mockResolvedValue(makeEvidence());
});

describe('useProcessing', () => {
  it('rejects an invalid file before touching the network', async () => {
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('notes.txt')));

    expect(videosService.upload).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.error).toMatch(/Unsupported format/));
    expect(useProcessingStore.getState().logs.at(-1)?.level).toBe('error');
  });

  it('uploads, reports progress, and starts processing', async () => {
    vi.mocked(videosService.upload).mockImplementation(async (input: UploadVideoInput) => {
      input.onProgress?.(0.5);
      return makeVideo();
    });
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));

    await waitFor(() => expect(result.current.video?.video_id).toBe('vid-1'));
    expect(useUploadStore.getState().progress).toBe(1);
    expect(videosService.startProcessing).toHaveBeenCalledWith({ videoId: 'vid-1' });
    await waitFor(() => expect(useProcessingStore.getState().jobId).toBe('job-1'));
    expect(useSelectionStore.getState().currentVideoId).toBe('vid-1');
    expect(result.current.logs.map((entry) => entry.message)).toContain('Uploaded junction.mp4.');
  });

  it('drives the phase and activity log from the polled job status', async () => {
    vi.mocked(videosService.getJob).mockResolvedValue(
      makeJob({ status: 'succeeded', progress: 1, event_count: 3 }),
    );
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));

    await waitFor(() => expect(result.current.phase).toBe('completed'));
    expect(result.current.progressRatio).toBe(1);
    expect(result.current.logs.at(-1)?.message).toContain('3 event(s)');
  });

  it('records a failed job with its error', async () => {
    vi.mocked(videosService.getJob).mockResolvedValue(
      makeJob({ status: 'failed', progress: null, error: 'decoder crashed' }),
    );
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));

    await waitFor(() => expect(result.current.phase).toBe('failed'));
    expect(result.current.error).toBe('decoder crashed');
  });

  it('surfaces an upload failure', async () => {
    vi.mocked(videosService.upload).mockRejectedValue(
      new ApiError('Upload rejected', { kind: 'http', status: 413 }),
    );
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));

    await waitFor(() => expect(result.current.phase).toBe('failed'));
    expect(result.current.error).toBe('Upload rejected');
    expect(videosService.startProcessing).not.toHaveBeenCalled();
  });

  it('cancels an in-flight upload and clears the workspace', async () => {
    vi.mocked(videosService.upload).mockImplementation(
      (input: UploadVideoInput) =>
        new Promise((_resolve, reject) => {
          input.signal?.addEventListener('abort', () =>
            reject(new ApiError('Request canceled', { kind: 'canceled' })),
          );
        }),
    );
    const { result } = renderHook(() => useProcessing(), { wrapper });

    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));
    await waitFor(() => expect(result.current.phase).toBe('uploading'));

    act(() => result.current.actions.cancelUpload());

    await waitFor(() => expect(useUploadStore.getState().phase).toBe('idle'));
    expect(useProcessingStore.getState().phase).toBe('idle');
  });

  it('removes the video and clears every workspace store', async () => {
    const { result } = renderHook(() => useProcessing(), { wrapper });
    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));
    await waitFor(() => expect(result.current.video).not.toBeNull());

    act(() => result.current.actions.remove());

    expect(useUploadStore.getState().video).toBeNull();
    expect(useProcessingStore.getState().phase).toBe('idle');
    expect(useSelectionStore.getState().currentVideoId).toBeNull();
  });

  it('retries processing for an already-uploaded video', async () => {
    const { result } = renderHook(() => useProcessing(), { wrapper });
    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));
    await waitFor(() => expect(result.current.video).not.toBeNull());

    act(() => result.current.actions.retry());

    await waitFor(() => expect(videosService.startProcessing).toHaveBeenCalledTimes(2));
    expect(videosService.upload).toHaveBeenCalledTimes(1);
  });

  it('replaces the video by uploading the new file', async () => {
    const { result } = renderHook(() => useProcessing(), { wrapper });
    act(() => result.current.actions.selectAndUpload(makeFile('clip.mp4')));
    await waitFor(() => expect(result.current.video).not.toBeNull());

    act(() => result.current.actions.replace(makeFile('second.mp4')));

    await waitFor(() => expect(videosService.upload).toHaveBeenCalledTimes(2));
  });
});

describe('useWorkspaceEvents', () => {
  it('returns view-models for the video events', async () => {
    vi.mocked(eventsService.list).mockResolvedValue({
      items: [
        makeEventSummary({ event_id: 'a', trigger_at: mediaSeconds(4) }),
        makeEventSummary({ event_id: 'b', trigger_at: mediaSeconds(30) }),
      ],
      total: 2,
      limit: 200,
      offset: 0,
    });
    const { result } = renderHook(() => useWorkspaceEvents('vid-1'), { wrapper });

    await waitFor(() => expect(result.current.events).toHaveLength(2));
    expect(result.current.events.map((event) => event.mediaSeconds)).toEqual([4, 30]);
    expect(result.current.total).toBe(2);
    expect(eventsService.list).toHaveBeenCalledWith(
      { videoId: 'vid-1', limit: 200, offset: 0, sort: 'trigger_at' },
      expect.anything(),
    );
  });

  it('enriches the selected event with its detail and evidence', async () => {
    const { result } = renderHook(() => useWorkspaceEvents('vid-1'), { wrapper });
    await waitFor(() => expect(result.current.events).toHaveLength(1));

    act(() => result.current.select('evt-1'));

    await waitFor(() => expect(result.current.selectedEvent?.confidence).toBe(0.91));
    expect(result.current.selectedDetail?.event_id).toBe('evt-1');
    expect(result.current.evidence?.evidence_package_id).toBe('pkg-1');
    expect(eventsService.get).toHaveBeenCalledWith('evt-1', expect.anything());
  });

  it('reports list failures for the caller to surface', async () => {
    vi.mocked(eventsService.list).mockRejectedValue(
      new ApiError('Events unavailable', { kind: 'http', status: 500 }),
    );
    const { result } = renderHook(() => useWorkspaceEvents('vid-1'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.events).toEqual([]);
  });

  it('does not fetch detail while nothing is selected', async () => {
    const { result } = renderHook(() => useWorkspaceEvents('vid-1'), { wrapper });
    await waitFor(() => expect(result.current.events).toHaveLength(1));

    expect(result.current.selectedEvent).toBeNull();
    expect(result.current.isDetailLoading).toBe(false);
    expect(eventsService.get).not.toHaveBeenCalled();
  });
});

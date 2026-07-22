import { act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { makeFile, makeVideo } from '@/test/fixtures';

import * as storeBarrel from './index';
import { useProcessingStore } from './processing-store';
import { useSelectionStore } from './selection-store';
import { useUploadStore } from './upload-store';

beforeEach(() => {
  localStorage.clear();
  act(() => {
    useUploadStore.getState().reset();
    useProcessingStore.getState().reset();
    useSelectionStore.getState().clearSelection();
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('store barrel', () => {
  it('re-exports the H7C stores', () => {
    expect(storeBarrel.useUploadStore).toBeTypeOf('function');
    expect(storeBarrel.useProcessingStore).toBeTypeOf('function');
  });
});

describe('upload store', () => {
  it('walks the select → upload → uploaded lifecycle', () => {
    const file = makeFile('clip.mp4');
    act(() => useUploadStore.getState().selectFile(file));
    expect(useUploadStore.getState()).toMatchObject({ phase: 'selected', file, progress: 0 });

    act(() => useUploadStore.getState().markUploading());
    act(() => useUploadStore.getState().setProgress(0.4));
    expect(useUploadStore.getState()).toMatchObject({ phase: 'uploading', progress: 0.4 });

    const video = makeVideo();
    act(() => useUploadStore.getState().markUploaded(video));
    expect(useUploadStore.getState()).toMatchObject({ phase: 'uploaded', progress: 1, video });
  });

  it('clamps progress into 0..1', () => {
    act(() => useUploadStore.getState().setProgress(2));
    expect(useUploadStore.getState().progress).toBe(1);
    act(() => useUploadStore.getState().setProgress(-1));
    expect(useUploadStore.getState().progress).toBe(0);
  });

  it('records an error without losing the selected file', () => {
    const file = makeFile('clip.mp4');
    act(() => {
      useUploadStore.getState().selectFile(file);
      useUploadStore.getState().markError('Upload failed.');
    });
    expect(useUploadStore.getState()).toMatchObject({ phase: 'error', error: 'Upload failed.' });
    expect(useUploadStore.getState().file).toBe(file);
  });

  it('creates and revokes object URLs around selection and reset', () => {
    const createObjectURL = vi.fn(() => 'blob:clip');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });

    act(() => useUploadStore.getState().selectFile(makeFile('a.mp4')));
    expect(useUploadStore.getState().objectUrl).toBe('blob:clip');

    act(() => useUploadStore.getState().selectFile(makeFile('b.mp4')));
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:clip');

    act(() => useUploadStore.getState().reset());
    expect(useUploadStore.getState()).toMatchObject({ phase: 'idle', objectUrl: null, file: null });
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it('persists only the server record', () => {
    act(() => {
      useUploadStore.getState().selectFile(makeFile('clip.mp4'));
      useUploadStore.getState().markUploaded(makeVideo());
    });
    const persisted = JSON.parse(localStorage.getItem('trafficpulse-upload') ?? '{}');
    expect(Object.keys(persisted.state)).toEqual(['video']);
  });
});

describe('processing store', () => {
  it('moves from upload to a queued job and logs the workflow', () => {
    act(() => useProcessingStore.getState().beginUpload());
    expect(useProcessingStore.getState().phase).toBe('uploading');
    expect(useProcessingStore.getState().logs).toHaveLength(1);
    const startedAt = useProcessingStore.getState().startedAt;

    act(() => useProcessingStore.getState().attachJob('vid-1', 'job-1'));
    expect(useProcessingStore.getState()).toMatchObject({
      videoId: 'vid-1',
      jobId: 'job-1',
      phase: 'queued',
      startedAt,
    });

    act(() => useProcessingStore.getState().setPhase('running'));
    expect(useProcessingStore.getState().phase).toBe('running');
  });

  it('caps the activity log', () => {
    act(() => {
      for (let index = 0; index < 60; index += 1) {
        useProcessingStore.getState().addLog('info', `entry ${index}`);
      }
    });
    const { logs } = useProcessingStore.getState();
    expect(logs).toHaveLength(50);
    expect(logs[logs.length - 1].message).toBe('entry 59');
  });

  it('gives every log entry a unique id', () => {
    act(() => {
      useProcessingStore.getState().addLog('info', 'one');
      useProcessingStore.getState().addLog('error', 'two');
    });
    const ids = useProcessingStore.getState().logs.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('persists the ids and recovery hints needed to reconnect', () => {
    act(() => {
      useProcessingStore.getState().beginUpload();
      useProcessingStore.getState().attachJob('vid-1', 'job-1');
      useProcessingStore.getState().rememberSelection('evt-1');
      useProcessingStore.getState().rememberPlayback(12.9);
    });
    const persisted = JSON.parse(localStorage.getItem('trafficpulse-processing') ?? '{}');
    expect(persisted.state).toEqual({
      videoId: 'vid-1',
      jobId: 'job-1',
      selectedEventId: 'evt-1',
      playbackSeconds: 12,
    });
  });

  it('resets to idle', () => {
    act(() => {
      useProcessingStore.getState().beginUpload();
      useProcessingStore.getState().reset();
    });
    expect(useProcessingStore.getState()).toMatchObject({
      phase: 'idle',
      jobId: null,
      videoId: null,
      startedAt: null,
    });
    expect(useProcessingStore.getState().logs).toHaveLength(0);
  });
});

describe('selection store (workspace extension)', () => {
  it('selects and clears the focused event', () => {
    act(() => useSelectionStore.getState().selectEvent('evt-1'));
    expect(useSelectionStore.getState().selectedEventId).toBe('evt-1');
    act(() => useSelectionStore.getState().clearEventSelection());
    expect(useSelectionStore.getState().selectedEventId).toBeNull();
  });

  it('clears the event selection when the video selection is cleared', () => {
    act(() => {
      useSelectionStore.getState().selectVideo('vid-1');
      useSelectionStore.getState().selectEvent('evt-1');
      useSelectionStore.getState().clearSelection();
    });
    expect(useSelectionStore.getState()).toMatchObject({
      currentVideoId: null,
      selectedEventId: null,
    });
  });
});

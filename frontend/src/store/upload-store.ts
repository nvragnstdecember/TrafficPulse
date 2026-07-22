import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { type VideoUploadResponse } from '@/api/types';

export type UploadPhase = 'idle' | 'selected' | 'uploading' | 'uploaded' | 'error';

interface UploadState {
  /** The locally-selected File (session-only; enables in-browser playback). */
  file: File | null;
  /** Object URL for the selected file, or null when unavailable. */
  objectUrl: string | null;
  phase: UploadPhase;
  /** 0..1 upload progress. */
  progress: number;
  error: string | null;
  /** The server record once the upload completes (persisted for reconnect). */
  video: VideoUploadResponse | null;

  selectFile: (file: File) => void;
  markUploading: () => void;
  setProgress: (ratio: number) => void;
  markUploaded: (video: VideoUploadResponse) => void;
  markError: (message: string) => void;
  reset: () => void;
}

function createObjectUrl(file: File): string | null {
  if (typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function') {
    try {
      return URL.createObjectURL(file);
    } catch {
      return null;
    }
  }
  return null;
}

function revokeObjectUrl(url: string | null): void {
  if (url && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
    try {
      URL.revokeObjectURL(url);
    } catch {
      /* ignore */
    }
  }
}

/**
 * Upload state (H7C): the selected file + object URL, live upload progress, and
 * the resulting server record. Only the server `video` record is persisted (JSON)
 * so the workspace can reconnect after a refresh; the File/object URL are
 * session-only. Object URLs are always revoked when replaced or reset.
 */
export const useUploadStore = create<UploadState>()(
  persist(
    (set, get) => ({
      file: null,
      objectUrl: null,
      phase: 'idle',
      progress: 0,
      error: null,
      video: null,

      selectFile: (file) => {
        revokeObjectUrl(get().objectUrl);
        set({
          file,
          objectUrl: createObjectUrl(file),
          phase: 'selected',
          progress: 0,
          error: null,
          video: null,
        });
      },
      markUploading: () => set({ phase: 'uploading', progress: 0, error: null }),
      setProgress: (ratio) => set({ progress: Math.min(1, Math.max(0, ratio)) }),
      markUploaded: (video) => set({ phase: 'uploaded', progress: 1, video, error: null }),
      markError: (message) => set({ phase: 'error', error: message }),
      reset: () => {
        revokeObjectUrl(get().objectUrl);
        set({
          file: null,
          objectUrl: null,
          phase: 'idle',
          progress: 0,
          error: null,
          video: null,
        });
      },
    }),
    {
      name: 'trafficpulse-upload',
      partialize: (state) => ({ video: state.video }),
      onRehydrateStorage: () => (state) => {
        if (state?.video) state.phase = 'uploaded';
      },
    },
  ),
);

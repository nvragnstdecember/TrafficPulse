import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { type ProcessingPhase } from '@/lib/job';

export type LogLevel = 'info' | 'success' | 'error';

export interface LogEntry {
  id: string;
  at: number;
  level: LogLevel;
  message: string;
}

interface ProcessingState {
  videoId: string | null;
  jobId: string | null;
  phase: ProcessingPhase;
  logs: LogEntry[];
  startedAt: number | null;

  /** Begin the client-side upload phase (before a job exists). */
  beginUpload: () => void;
  /** A job was created for the video → move to the queued phase. */
  attachJob: (videoId: string, jobId: string) => void;
  setPhase: (phase: ProcessingPhase) => void;
  addLog: (level: LogLevel, message: string) => void;
  reset: () => void;
}

let logCounter = 0;
function nextLogId(): string {
  logCounter += 1;
  return `log-${logCounter}`;
}

const MAX_LOGS = 50;

/**
 * Processing state (H7C): the active video/job, lifecycle phase, and a rolling
 * activity log (latest messages). Only `videoId`/`jobId` are persisted so a page
 * refresh can reconnect to an in-flight job; the phase is re-derived by polling
 * and the logs restart for the session.
 */
export const useProcessingStore = create<ProcessingState>()(
  persist(
    (set) => ({
      videoId: null,
      jobId: null,
      phase: 'idle',
      logs: [],
      startedAt: null,

      beginUpload: () =>
        set({
          videoId: null,
          jobId: null,
          phase: 'uploading',
          startedAt: Date.now(),
          logs: [{ id: nextLogId(), at: Date.now(), level: 'info', message: 'Uploading video…' }],
        }),
      attachJob: (videoId, jobId) =>
        set((state) => ({
          videoId,
          jobId,
          phase: 'queued',
          startedAt: state.startedAt ?? Date.now(),
        })),
      setPhase: (phase) => set({ phase }),
      addLog: (level, message) =>
        set((state) => ({
          logs: [
            ...state.logs.slice(-(MAX_LOGS - 1)),
            { id: nextLogId(), at: Date.now(), level, message },
          ],
        })),
      reset: () => set({ videoId: null, jobId: null, phase: 'idle', logs: [], startedAt: null }),
    }),
    {
      name: 'trafficpulse-processing',
      partialize: (state) => ({ videoId: state.videoId, jobId: state.jobId }),
    },
  ),
);

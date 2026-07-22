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
  /** Last selected event id — persisted so a refresh restores the selection (H7D). */
  selectedEventId: string | null;
  /** Last playback position in seconds — persisted for refresh recovery (H7D). */
  playbackSeconds: number;

  /** Begin the client-side upload phase (before a job exists). */
  beginUpload: () => void;
  /** A job was created for the video → move to the queued phase. */
  attachJob: (videoId: string, jobId: string) => void;
  setPhase: (phase: ProcessingPhase) => void;
  addLog: (level: LogLevel, message: string) => void;
  /** Record the selected event for recovery (no-op if unchanged). */
  rememberSelection: (eventId: string | null) => void;
  /** Record the playback position for recovery (whole-second granularity). */
  rememberPlayback: (seconds: number) => void;
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
    (set, get) => ({
      videoId: null,
      jobId: null,
      phase: 'idle',
      logs: [],
      startedAt: null,
      selectedEventId: null,
      playbackSeconds: 0,

      beginUpload: () =>
        set({
          videoId: null,
          jobId: null,
          phase: 'uploading',
          startedAt: Date.now(),
          selectedEventId: null,
          playbackSeconds: 0,
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
      rememberSelection: (eventId) => {
        if (get().selectedEventId !== eventId) set({ selectedEventId: eventId });
      },
      rememberPlayback: (seconds) => {
        const whole = Math.max(0, Math.floor(seconds));
        if (get().playbackSeconds !== whole) set({ playbackSeconds: whole });
      },
      reset: () =>
        set({
          videoId: null,
          jobId: null,
          phase: 'idle',
          logs: [],
          startedAt: null,
          selectedEventId: null,
          playbackSeconds: 0,
        }),
    }),
    {
      name: 'trafficpulse-processing',
      partialize: (state) => ({
        videoId: state.videoId,
        jobId: state.jobId,
        selectedEventId: state.selectedEventId,
        playbackSeconds: state.playbackSeconds,
      }),
    },
  ),
);

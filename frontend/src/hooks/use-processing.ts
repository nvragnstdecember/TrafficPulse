import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';

import { queryKeys } from '@/api/query-keys';
import { type JobStatusResponse, type VideoUploadResponse } from '@/api/types';
import { ApiError, toErrorMessage } from '@/api/errors';
import { type ProcessingPhase, isActivePhase, jobProgressRatio, jobStatusToPhase } from '@/lib/job';
import { validateUploadFile } from '@/lib/upload';
import { type LogEntry, useProcessingStore } from '@/store/processing-store';
import { useSelectionStore } from '@/store/selection-store';
import { useUploadStore } from '@/store/upload-store';

import { useJob, useStartProcessing, useUploadVideo } from './use-videos';

export interface ProcessingActions {
  selectAndUpload: (file: File) => void;
  startProcessing: () => void;
  cancelUpload: () => void;
  retry: () => void;
  remove: () => void;
  replace: (file: File) => void;
}

export interface ProcessingController {
  phase: ProcessingPhase;
  job: JobStatusResponse | undefined;
  video: VideoUploadResponse | null;
  /** Upload progress while uploading, else job progress; 0..1 or null. */
  progressRatio: number | null;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  logs: LogEntry[];
  error: string | null;
  isBusy: boolean;
  actions: ProcessingActions;
}

/** A 1s ticker while `active`, so elapsed time updates without a component timer. */
function useTicker(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

/**
 * Processing orchestration (H7C): the single hook the workspace uses to drive
 * upload → process → poll, plus cancel/retry/remove/replace and reconnect.
 *
 * It composes the existing mutation hooks (`useUploadVideo`, `useStartProcessing`)
 * and the polling `useJob` (all Query-layer) with the upload/processing/selection
 * stores. No component performs polling or network calls — they call `actions`.
 */
export function useProcessing(): ProcessingController {
  const queryClient = useQueryClient();
  const uploadStore = useUploadStore();
  const processing = useProcessingStore();
  const selectVideo = useSelectionStore((s) => s.selectVideo);
  const clearSelection = useSelectionStore((s) => s.clearSelection);

  const uploadMutation = useUploadVideo();
  const startMutation = useStartProcessing();
  const jobQuery = useJob(processing.jobId ?? undefined, { poll: true });

  const abortRef = useRef<AbortController | null>(null);
  const lastStatusRef = useRef<string | null>(null);
  const reconnectedRef = useRef(false);

  // Reconnect: a persisted jobId with no session logs means we resumed after a
  // refresh — announce it once so the activity log explains the state.
  useEffect(() => {
    if (
      !reconnectedRef.current &&
      processing.jobId &&
      processing.logs.length === 0 &&
      processing.phase === 'idle'
    ) {
      reconnectedRef.current = true;
      processing.addLog('info', 'Reconnected to processing job.');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startProcessingFor = useCallback(
    (videoId: string) => {
      startMutation.mutate(
        { videoId },
        {
          onSuccess: (res) => {
            useProcessingStore.getState().attachJob(videoId, res.job_id);
            useProcessingStore.getState().addLog('info', 'Queued for processing.');
          },
          onError: (error) => {
            useProcessingStore.getState().setPhase('failed');
            useProcessingStore.getState().addLog('error', toErrorMessage(error));
          },
        },
      );
    },
    [startMutation],
  );

  const runUpload = useCallback(
    (file: File) => {
      const validation = validateUploadFile(file);
      if (!validation.ok) {
        useUploadStore.getState().markError(validation.message);
        useProcessingStore.getState().addLog('error', validation.message);
        return;
      }
      useUploadStore.getState().selectFile(file);
      useUploadStore.getState().markUploading();
      useProcessingStore.getState().reset();
      useProcessingStore.getState().beginUpload();

      const controller = new AbortController();
      abortRef.current = controller;
      uploadMutation.mutate(
        {
          file,
          signal: controller.signal,
          onProgress: (ratio) => useUploadStore.getState().setProgress(ratio),
        },
        {
          onSuccess: (video) => {
            useUploadStore.getState().markUploaded(video);
            selectVideo(video.video_id);
            useProcessingStore.getState().addLog('success', `Uploaded ${video.filename}.`);
            startProcessingFor(video.video_id);
          },
          onError: (error) => {
            if (error instanceof ApiError && error.isCanceled) {
              useUploadStore.getState().reset();
              useProcessingStore.getState().reset();
              return;
            }
            const message = toErrorMessage(error);
            useUploadStore.getState().markError(message);
            useProcessingStore.getState().setPhase('failed');
            useProcessingStore.getState().addLog('error', message);
          },
        },
      );
    },
    [uploadMutation, selectVideo, startProcessingFor],
  );

  // Drive phase + activity log from the polled job status (Query layer).
  useEffect(() => {
    const job = jobQuery.data;
    if (!job) return;
    if (lastStatusRef.current === job.status) return;
    lastStatusRef.current = job.status;

    const phase = jobStatusToPhase(job.status);
    const store = useProcessingStore.getState();
    store.setPhase(phase);
    if (phase === 'running') store.addLog('info', 'Processing video…');
    if (phase === 'completed') {
      store.addLog('success', `Completed — ${job.event_count} event(s) detected.`);
      void queryClient.invalidateQueries({ queryKey: queryKeys.events.all });
    }
    if (phase === 'failed') store.addLog('error', job.error ?? 'Processing failed.');
  }, [jobQuery.data, queryClient]);

  const actions: ProcessingActions = {
    selectAndUpload: runUpload,
    startProcessing: () => {
      const video = useUploadStore.getState().video;
      if (video) startProcessingFor(video.video_id);
    },
    cancelUpload: () => abortRef.current?.abort(),
    retry: () => {
      const upload = useUploadStore.getState();
      lastStatusRef.current = null;
      if (upload.video) startProcessingFor(upload.video.video_id);
      else if (upload.file) runUpload(upload.file);
    },
    remove: () => {
      abortRef.current?.abort();
      lastStatusRef.current = null;
      useUploadStore.getState().reset();
      useProcessingStore.getState().reset();
      clearSelection();
    },
    replace: (file) => {
      abortRef.current?.abort();
      lastStatusRef.current = null;
      runUpload(file);
    },
  };

  const busy =
    isActivePhase(processing.phase) || uploadMutation.isPending || startMutation.isPending;
  const ticked = useTicker(busy);
  const elapsedSeconds =
    processing.startedAt != null ? Math.max(0, (ticked - processing.startedAt) / 1000) : null;

  const progressRatio =
    processing.phase === 'uploading' ? uploadStore.progress : jobProgressRatio(jobQuery.data);

  return {
    phase: processing.phase,
    job: jobQuery.data,
    video: uploadStore.video,
    progressRatio,
    elapsedSeconds,
    etaSeconds: jobQuery.data?.estimated_remaining_seconds ?? null,
    logs: processing.logs,
    error: uploadStore.error ?? jobQuery.data?.error ?? null,
    isBusy: busy,
    actions,
  };
}

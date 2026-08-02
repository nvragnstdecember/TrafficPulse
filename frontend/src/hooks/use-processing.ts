import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';

import { queryKeys } from '@/api/query-keys';
import {
  type JobStatusResponse,
  type VideoSummary,
  type VideoUploadResponse,
} from '@/api/types';
import { ApiError, toErrorMessage } from '@/api/errors';
import {
  type ProcessingPhase,
  derivePhase,
  isActivePhase,
  isCancellablePhase,
  jobProgressRatio,
} from '@/lib/job';
import { validateUploadFile } from '@/lib/upload';
import { type LogEntry, useProcessingStore } from '@/store/processing-store';
import { useSelectionStore } from '@/store/selection-store';
import { useUploadStore } from '@/store/upload-store';

import { useCancelJob, useJob, useStartProcessing, useUploadVideo } from './use-videos';

export interface ProcessingActions {
  selectAndUpload: (file: File) => void;
  /** Open a stored video from the library — its finished run, or start one (H11). */
  openVideo: (video: VideoSummary) => void;
  startProcessing: () => void;
  /**
   * Re-run the open video with an explicit rule set (H13).
   *
   * Calibrating a scene changes what the video supports, so the rules a run should
   * request are known only after the scene is saved. Passing them explicitly is what
   * lets a newly-drawn junction actually be analysed without a server default that
   * could not know about it.
   */
  reprocessWith: (rules: Array<Record<string, unknown>>) => void;
  /** Cancel the in-flight upload (client abort) or the running job (API), as apt. */
  cancel: () => void;
  /** Abort an in-flight upload (client-side only). Kept for the uploading phase. */
  cancelUpload: () => void;
  retry: () => void;
  remove: () => void;
  replace: (file: File) => void;
  /** Re-poll the job now (e.g. after a transient backend outage). */
  reconnect: () => void;
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
  /** A cancellation request is in flight (the job has not yet observed it). */
  isCancelling: boolean;
  /** The job poll is currently failing (backend unavailable / transient outage). */
  connectionError: unknown;
  actions: ProcessingActions;
}

/**
 * Present a library row as the upload record the workspace already understands.
 *
 * The two shapes overlap on everything the workspace reads; `status` is the
 * upload-receipt vocabulary ("stored"), and `frame_count` is not carried by a
 * library row, so it stays null rather than being guessed from fps × duration.
 */
function toUploadRecord(video: VideoSummary): VideoUploadResponse {
  return {
    video_id: video.video_id,
    filename: video.filename,
    status: 'stored',
    size_bytes: video.size_bytes,
    width: video.width,
    height: video.height,
    fps: video.fps,
    frame_count: null,
    duration_seconds: video.duration_seconds,
    codec: video.codec,
  };
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
  const cancelMutation = useCancelJob();
  const jobQuery = useJob(processing.jobId ?? undefined, { poll: true });

  const abortRef = useRef<AbortController | null>(null);
  const lastStatusRef = useRef<string | null>(null);
  const lastOverlayRef = useRef<string | null>(null);
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
    (videoId: string, rules?: Array<Record<string, unknown>>) => {
      startMutation.mutate(
        { videoId, rules },
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
            // Duplicate: this exact video is already stored. Adopt the existing
            // record (we still hold the local file for playback) and process it,
            // instead of dead-ending the upload.
            if (error instanceof ApiError && error.isDuplicate && error.videoId) {
              const existing: VideoUploadResponse = {
                video_id: error.videoId,
                filename: file.name,
                status: 'stored',
                size_bytes: file.size,
                width: null,
                height: null,
                fps: null,
                frame_count: null,
                duration_seconds: null,
                codec: '',
              };
              useUploadStore.getState().markUploaded(existing);
              selectVideo(existing.video_id);
              useProcessingStore
                .getState()
                .addLog('info', `Already uploaded — opening ${file.name}.`);
              startProcessingFor(existing.video_id);
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

  // Drive phase + activity log from the polled job status (Query layer). The
  // phase is re-derived on every poll (so initializing→running→finalizing track
  // the live frame counters), while the activity log fires only on backend
  // status transitions.
  useEffect(() => {
    const job = jobQuery.data;
    if (!job) return;
    const store = useProcessingStore.getState();

    const phase = derivePhase(job);
    if (phase !== store.phase) store.setPhase(phase);

    // The annotated video renders after the run finishes, so its transitions are
    // logged on their own axis — otherwise the analyst sees the player silently
    // swap sources with no explanation of where the boxes came from.
    if (lastOverlayRef.current !== job.overlay_status) {
      lastOverlayRef.current = job.overlay_status;
      if (job.overlay_status === 'pending') {
        store.addLog('info', 'Rendering explainable overlay…');
      } else if (job.overlay_status === 'ready') {
        store.addLog('success', 'Annotated video ready — now showing AI reasoning.');
      } else if (job.overlay_status === 'failed') {
        store.addLog('error', 'Overlay render failed; showing the original video.');
      }
    }

    if (lastStatusRef.current === job.status) return;
    lastStatusRef.current = job.status;
    if (job.status === 'running') {
      store.addLog('info', 'Processing video…');
    } else if (job.status === 'succeeded') {
      store.addLog('success', `Completed — ${job.event_count} event(s) detected.`);
      void queryClient.invalidateQueries({ queryKey: queryKeys.events.all });
    } else if (job.status === 'failed') {
      store.addLog('error', job.error ?? 'Processing failed.');
    } else if (job.status === 'cancelled') {
      store.addLog('info', 'Processing cancelled.');
    }
  }, [jobQuery.data, queryClient]);

  /**
   * Open a stored video from the historical library (H11).
   *
   * The whole point of the milestone: a video that was processed in some earlier
   * session becomes the workspace's current video without being re-uploaded. It
   * seeds exactly the state a completed upload would have left behind — the server
   * record and the job id — so every downstream surface (player, timeline, event
   * list, review panel) works unchanged and cannot tell the two paths apart.
   *
   * A video with no run has nothing to reopen, so opening it starts one, which is
   * what the upload path does after storing a file.
   */
  const openVideo = useCallback(
    (summary: VideoSummary) => {
      abortRef.current?.abort();
      lastStatusRef.current = null;
      lastOverlayRef.current = null;
      // Drop any local file/object URL first: it belongs to a different video, and
      // leaving it would play the wrong clip under the right metadata.
      useUploadStore.getState().reset();
      useUploadStore.getState().markUploaded(toUploadRecord(summary));
      selectVideo(summary.video_id);
      if (summary.job_id) {
        useProcessingStore.getState().adoptJob(summary.video_id, summary.job_id);
      } else {
        useProcessingStore.getState().reset();
        startProcessingFor(summary.video_id);
      }
    },
    [selectVideo, startProcessingFor],
  );

  const cancelUpload = useCallback(() => abortRef.current?.abort(), []);

  const cancelJob = useCallback(() => {
    const store = useProcessingStore.getState();
    if (store.phase === 'uploading') {
      cancelUpload();
      return;
    }
    const jobId = store.jobId;
    if (jobId && isCancellablePhase(store.phase)) {
      store.addLog('info', 'Cancelling…');
      cancelMutation.mutate(jobId);
    }
  }, [cancelUpload, cancelMutation]);

  const actions: ProcessingActions = {
    selectAndUpload: runUpload,
    openVideo,
    startProcessing: () => {
      const video = useUploadStore.getState().video;
      if (video) startProcessingFor(video.video_id);
    },
    reprocessWith: (rules) => {
      const video = useUploadStore.getState().video;
      if (!video) return;
      lastStatusRef.current = null;
      lastOverlayRef.current = null;
      startProcessingFor(video.video_id, rules);
    },
    cancel: cancelJob,
    cancelUpload,
    retry: () => {
      const upload = useUploadStore.getState();
      lastStatusRef.current = null;
      lastOverlayRef.current = null;
      if (upload.video) startProcessingFor(upload.video.video_id);
      else if (upload.file) runUpload(upload.file);
    },
    remove: () => {
      abortRef.current?.abort();
      lastStatusRef.current = null;
      lastOverlayRef.current = null;
      useUploadStore.getState().reset();
      useProcessingStore.getState().reset();
      clearSelection();
    },
    replace: (file) => {
      abortRef.current?.abort();
      lastStatusRef.current = null;
      lastOverlayRef.current = null;
      runUpload(file);
    },
    reconnect: () => void jobQuery.refetch(),
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
    isCancelling: cancelMutation.isPending,
    connectionError: jobQuery.isError ? jobQuery.error : null,
    actions,
  };
}

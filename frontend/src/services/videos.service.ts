import { apiClient } from '@/api/client';
import { endpoints } from '@/api/endpoints';
import {
  type JobStatusResponse,
  type ProcessResponse,
  type VideoUploadResponse,
} from '@/api/types';

export interface UploadVideoInput {
  file: File;
  signal?: AbortSignal;
  /** When provided, upload reports 0..1 progress (XHR-backed). */
  onProgress?: (ratio: number) => void;
}

export interface StartProcessingInput {
  videoId: string;
  /** Optional H7A rule declarations; omitted lets the server use its defaults. */
  rules?: Array<Record<string, unknown>>;
  signal?: AbortSignal;
}

/**
 * Videos service (H7B): upload, start processing, poll job status.
 *
 * Feature UIs will call these through hooks; the service itself carries no React
 * or view logic.
 */
export const videosService = {
  upload({ file, signal, onProgress }: UploadVideoInput): Promise<VideoUploadResponse> {
    const form = new FormData();
    form.append('file', file);
    if (onProgress) {
      return apiClient.uploadWithProgress<VideoUploadResponse>(endpoints.videoUpload, form, {
        signal,
        onProgress: (progress) => onProgress(progress.ratio),
      });
    }
    return apiClient.upload<VideoUploadResponse>(endpoints.videoUpload, form, { signal });
  },
  startProcessing({ videoId, rules, signal }: StartProcessingInput): Promise<ProcessResponse> {
    return apiClient.post<ProcessResponse>(
      endpoints.process,
      { video_id: videoId, rules: rules ?? null },
      { signal },
    );
  },
  getJob(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
    return apiClient.get<JobStatusResponse>(endpoints.job(jobId), { signal });
  },
  cancelJob(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
    return apiClient.post<JobStatusResponse>(endpoints.cancelJob(jobId), undefined, { signal });
  },
};

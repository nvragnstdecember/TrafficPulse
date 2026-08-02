/**
 * API endpoint paths (H7B).
 *
 * The single registry of backend routes, so no path string is duplicated across
 * services. Mirrors the H7A router layout under `/api`.
 */
export const endpoints = {
  health: '/api/health',
  metrics: '/api/metrics',
  videoUpload: '/api/video/upload',
  videos: '/api/videos',
  video: (videoId: string) => `/api/videos/${encodeURIComponent(videoId)}`,
  /** The stored source video — playable for a video this session did not upload. */
  videoMedia: (videoId: string) => `/api/videos/${encodeURIComponent(videoId)}/media`,
  process: '/api/process',
  job: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}`,
  cancelJob: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}/cancel`,
  jobOverlay: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}/overlay`,
  events: '/api/events',
  event: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}`,
  evidence: (eventId: string) => `/api/evidence/${encodeURIComponent(eventId)}`,
  review: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}/review`,
} as const;

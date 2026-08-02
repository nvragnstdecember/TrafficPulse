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
  /** A stored scene revision, addressed by the hash events carry (H12). */
  scene: (sceneHash: string) => `/api/scenes/${encodeURIComponent(sceneHash)}`,
  /** A video's calibrated scene: GET reads the binding, PUT authors it. */
  videoScene: (videoId: string) => `/api/videos/${encodeURIComponent(videoId)}/scene`,
  videoSceneValidate: (videoId: string) =>
    `/api/videos/${encodeURIComponent(videoId)}/scene/validate`,
  process: '/api/process',
  job: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}`,
  cancelJob: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}/cancel`,
  jobOverlay: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}/overlay`,
  events: '/api/events',
  event: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}`,
  evidence: (eventId: string) => `/api/evidence/${encodeURIComponent(eventId)}`,
  review: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}/review`,
} as const;

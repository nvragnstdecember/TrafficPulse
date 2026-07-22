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
  process: '/api/process',
  job: (jobId: string) => `/api/process/${encodeURIComponent(jobId)}`,
  events: '/api/events',
  event: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}`,
  evidence: (eventId: string) => `/api/evidence/${encodeURIComponent(eventId)}`,
} as const;

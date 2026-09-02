/**
 * API endpoint paths (H7B).
 *
 * The single registry of backend routes, so no path string is duplicated across
 * services. Mirrors the H7A router layout under `/api`.
 */
export const endpoints = {
  health: '/api/health',
  metrics: '/api/metrics',
  /** The whole dashboard in one aggregated response (H15). */
  analyticsSummary: '/api/analytics/summary',
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
  /** A finished run's helmet analysis (perception only; never a violation). */
  helmetAnalysis: (jobId: string) =>
    `/api/process/${encodeURIComponent(jobId)}/helmet-analysis`,
  /** What this deployment can honestly claim, per capability. */
  posture: '/api/system/posture',
  events: '/api/events',
  event: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}`,
  evidence: (eventId: string) => `/api/evidence/${encodeURIComponent(eventId)}`,
  /** One backend-rendered evidence frame (H14) — the only source of evidence pixels. */
  evidenceArtifact: (eventId: string, kind: string) =>
    `/api/evidence/${encodeURIComponent(eventId)}/artifacts/${encodeURIComponent(kind)}`,
  /** The downloadable ZIP evidence package for an event (H14). */
  evidencePackage: (eventId: string) =>
    `/api/evidence/${encodeURIComponent(eventId)}/package`,
  review: (eventId: string) => `/api/events/${encodeURIComponent(eventId)}/review`,
  /** Whether this deployment can monitor a live camera at all (pre-flight). */
  liveStatus: '/api/live/status',
  /** Live sessions currently running in the server process. */
  liveSessions: '/api/live/sessions',
  /**
   * The live monitoring socket. A path, like every other entry: the ws:// or
   * wss:// origin is resolved at connect time from the page, so nothing here is
   * hardcoded to a host.
   */
  liveSocket: '/api/live/ws',
} as const;

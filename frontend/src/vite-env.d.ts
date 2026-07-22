/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_TIMEOUT_MS?: string;
  readonly VITE_MAX_UPLOAD_BYTES?: string;
  readonly VITE_ACCEPTED_VIDEO_FORMATS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

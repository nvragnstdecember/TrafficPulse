/**
 * Client-side video frame grabbing for review thumbnails (Phase 2).
 *
 * The backend renders no still images — evidence artifacts are typed *references*,
 * and adding a server-side frame endpoint would mean decoding video per request.
 * The browser already has the footage, so a thumbnail is produced where the pixels
 * already are: seek a detached `<video>`, draw one frame to a canvas, keep the data
 * URL. No new endpoint, no new dependency, no extra bytes over the network.
 *
 * Three properties this has to have, all of them about not degrading the page:
 *
 * * **serialized** — one decoder at a time. A dashboard of twenty cards asking for
 *   twenty simultaneous seeks would thrash the media stack; requests queue instead.
 * * **cached** — by `(src, second)`, so scrolling a list back and forth costs
 *   nothing and a re-render never re-decodes.
 * * **failure is normal** — a codec the canvas cannot read, a cross-origin source
 *   that taints it, or an environment with no real media stack (jsdom) all resolve
 *   to `null` rather than throwing. Every caller must have a non-image fallback.
 */

const CACHE = new Map<string, string | null>();
const MAX_CACHE_ENTRIES = 240;

/** How long to wait for a seek before giving up on a frame. */
export const THUMBNAIL_TIMEOUT_MS = 4000;

/** Rounding applied to the requested second, so near-identical asks share a cache slot. */
export const THUMBNAIL_TIME_PRECISION = 2;

let queue: Promise<unknown> = Promise.resolve();

export function thumbnailCacheKey(src: string, seconds: number): string {
  return `${src}@${seconds.toFixed(THUMBNAIL_TIME_PRECISION)}`;
}

/** Test seam: forget every cached frame. */
export function clearThumbnailCache(): void {
  CACHE.clear();
}

function remember(key: string, value: string | null): string | null {
  if (CACHE.size >= MAX_CACHE_ENTRIES) {
    const oldest = CACHE.keys().next();
    if (!oldest.done) CACHE.delete(oldest.value);
  }
  CACHE.set(key, value);
  return value;
}

function captureOnce(src: string, seconds: number, width: number): Promise<string | null> {
  return new Promise((resolve) => {
    if (typeof document === 'undefined') {
      resolve(null);
      return;
    }

    const video = document.createElement('video');
    let settled = false;

    const finish = (value: string | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      video.removeAttribute('src');
      video.load?.();
      resolve(value);
    };

    const onSeeked = () => {
      try {
        const ratio = video.videoHeight > 0 ? video.videoHeight / video.videoWidth : 9 / 16;
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = Math.max(1, Math.round(width * ratio));
        const context = canvas.getContext('2d');
        if (!context) {
          finish(null);
          return;
        }
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Throws a SecurityError on a tainted canvas (cross-origin media).
        finish(canvas.toDataURL('image/jpeg', 0.7));
      } catch {
        finish(null);
      }
    };

    const onError = () => finish(null);
    const timer = setTimeout(() => finish(null), THUMBNAIL_TIMEOUT_MS);

    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    video.preload = 'auto';
    video.muted = true;
    video.crossOrigin = 'anonymous';
    try {
      video.src = src;
      video.currentTime = Math.max(0, seconds);
    } catch {
      finish(null);
    }
  });
}

/**
 * A JPEG data URL for one frame, or `null` when a frame cannot be produced.
 *
 * Repeated calls for the same `(src, seconds)` share one decode and one result.
 */
export function captureThumbnail(
  src: string,
  seconds: number,
  width = 160,
): Promise<string | null> {
  const key = thumbnailCacheKey(src, seconds);
  const cached = CACHE.get(key);
  if (cached !== undefined) return Promise.resolve(cached);

  const run = queue.then(() => captureOnce(src, seconds, width).then((value) => remember(key, value)));
  // Keep the chain alive even if one capture rejects unexpectedly.
  queue = run.catch(() => null);
  return run.catch(() => remember(key, null));
}
